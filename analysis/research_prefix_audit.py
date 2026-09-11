"""Retrospective prefix-feasibility audit; NOT a confirmatory Gate C score.

No GPU, downloads, tuning, or edits to frozen classifiers. Stage 2 is already
published/exposed. Fixed logistic C=1, train-only scaling, seed=20260905.
Baseline is ONLY first-token logprob, not the missing as-of-prefix LP baseline.
Fractional checkpoints are oracle-length retrospective observations, not an
online stopping policy. Labels retain the historical judged-file ingestion.

Run: python -m analysis.research_prefix_audit --out out/research_20260905/prefix.json
Smoke: add --smoke (first 300 usable rows per stage; not representative).
"""
from __future__ import annotations

import argparse
from collections import Counter
from copy import deepcopy
import hashlib
import json
from pathlib import Path

import numpy as np

SCALARS = ("ignition_frac", "ignition_depth", "mean_log_rank_answer",
           "band_agreement", "mean_entropy", "best_hedge_rank_log")
FAMILIES = ("first_lp", "onset", "mid", "end")


def available_features(row: dict, checkpoint: str) -> list[float]:
    """Never consult full-answer LP, length, text, label, or a later checkpoint."""
    first = [float(row["logprob_features"]["bl_first_token_logprob"])]
    if checkpoint == "first_lp":
        return first
    onset = row["onset_workspace_features"]
    base = [float(onset[k]) for k in SCALARS]
    if checkpoint == "onset":
        return first + base
    frac = {"mid": 0.5, "end": 1.0}[checkpoint]
    current = next(p for p in row["prefix_workspace_features"] if p["frac"] == frac)
    return first + base + [float(current[k]) - float(onset[k]) for k in SCALARS]


def load_stage(path: Path, limit: int | None = None) -> tuple[list[dict], dict]:
    rows, counts, digest = [], Counter(), hashlib.sha256()
    with path.open("rb") as stream:
        for line in stream:
            digest.update(line)
            if not line.strip():
                continue
            counts["raw_rows"] += 1
            r = json.loads(line)
            correct = r.get("deterministic_grade", {}).get("correct")
            if correct is None:
                counts["unresolved"] += 1
                continue
            try:
                matrices = {f: available_features(r, f) for f in FAMILIES}
                p = {v["frac"]: v["token_index"] for v in r["prefix_workspace_features"]}
                if not all(np.isfinite(v).all() for v in matrices.values()):
                    raise ValueError("nonfinite")
                rows.append({"id": r["example_id"], "group": r["split_group"],
                             "source": r["source_dataset"], "y": int(not correct),
                             "length": r["token_count"], "indices": [p[0], p[0.5], p[1]],
                             "features": matrices})
            except (KeyError, ValueError, StopIteration):
                counts["missing_or_nonfinite_features"] += 1
            if limit and len(rows) >= limit:
                break
    counts["usable_rows"] = len(rows)
    return rows, {"path": str(path.resolve()), "sha256": digest.hexdigest(),
                  "hash_scope": "read_prefix" if limit else "whole_file", **counts}


def describe(rows: list[dict]) -> dict:
    lengths = np.array([r["length"] for r in rows])
    return {"n": len(rows), "groups": len({r["group"] for r in rows}),
            "errors": sum(r["y"] for r in rows),
            "median_tokens": float(np.median(lengths)),
            "length_le_2": int((lengths <= 2).sum()),
            "mid_equals_onset": sum(r["indices"][1] == r["indices"][0] for r in rows),
            "mid_equals_end": sum(r["indices"][1] == r["indices"][2] for r in rows),
            "three_distinct_checkpoints": sum(len(set(r["indices"])) == 3 for r in rows)}


def score(train: list[dict], test: list[dict]) -> dict:
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import roc_auc_score
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler

    excluded_groups = {r["group"] for r in test}
    excluded_ids = {r["id"] for r in test}
    tr = [r for r in train if r["group"] not in excluded_groups and r["id"] not in excluded_ids]
    out = {"train_n": len(tr), "train_overlap_removed": len(train) - len(tr),
           "test_n": len(test), "auc": {}, "auc_length_ge_3": {}}
    ytr, yte = np.array([r["y"] for r in tr]), np.array([r["y"] for r in test])
    if len(set(ytr)) < 2 or len(set(yte)) < 2:
        out["status"] = "single_class_not_evaluable"
        return out
    long = np.array([r["length"] >= 3 for r in test])
    for family in FAMILIES:
        model = make_pipeline(StandardScaler(), LogisticRegression(
            C=1.0, max_iter=2000, class_weight="balanced", random_state=20260905))
        model.fit(np.array([r["features"][family] for r in tr]), ytr)
        pred = model.predict_proba(np.array([r["features"][family] for r in test]))[:, 1]
        out["auc"][family] = float(roc_auc_score(yte, pred))
        out["auc_length_ge_3"][family] = (float(roc_auc_score(yte[long], pred[long]))
            if len(set(yte[long])) == 2 else None)
    out["status"] = "exploratory"
    return out


def conformance_negative_controls() -> dict:
    """Record current harness behavior on invalid/incomplete capture pairs."""
    from sidecar.conformance import compare_captures
    ref = {"input_token_ids": [1, 2], "layers": [
        {"layer": i, "residual": [1., 0.], "top_ids": [1, 2], "top_probs": [.7, .2]}
        for i in (2, 3)], "final": {"next_token_id": 1}}
    candidates = {}
    missing_layer = deepcopy(ref)
    missing_layer["layers"].pop()
    candidates["missing_layer"] = (ref, missing_layer)
    missing_residual = deepcopy(ref)
    for layer in missing_residual["layers"]:
        layer.pop("residual")
    candidates["missing_residual"] = (ref, missing_residual)
    no_final = deepcopy(ref)
    no_final.pop("final")
    candidates["both_missing_final"] = (no_final, deepcopy(no_final))
    scaled = deepcopy(ref)
    for layer in scaled["layers"]:
        layer["residual"] = [10., 0.]
    candidates["tenfold_residual_norm"] = (ref, scaled)
    return {name: compare_captures(a, b)["passed"] for name, (a, b) in candidates.items()}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage1", type=Path, default=Path("out/campaign/stage1_judged.jsonl"))
    parser.add_argument("--stage2", type=Path, default=Path("out/campaign/stage2_judged.jsonl"))
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args()
    if args.out.exists():
        raise FileExistsError(f"Refusing to overwrite {args.out}; choose a new output.")
    stages, provenance = {}, {}
    for stage, path in (("stage1", args.stage1), ("stage2", args.stage2)):
        stages[stage], provenance[stage] = load_stage(path, 300 if args.smoke else None)
    report = {"status": "EXPLORATORY_NOT_GATE_C", "provenance": provenance,
              "features": {f: len(stages["stage1"][0]["features"][f]) for f in FAMILIES},
              "checkpoint_audit": {}, "stage1_lodo": {}, "stage1_to_stage2": {},
              "conformance_invalid_pairs_accepted": conformance_negative_controls()}
    for stage, rows in stages.items():
        report["checkpoint_audit"][stage] = {"all": describe(rows), "per_source": {
            src: describe([r for r in rows if r["source"] == src])
            for src in sorted({r["source"] for r in rows})}}
    for stage, key in (("stage1", "stage1_lodo"), ("stage2", "stage1_to_stage2")):
        for src in sorted({r["source"] for r in stages[stage]}):
            test = [r for r in stages[stage] if r["source"] == src]
            train = [r for r in stages["stage1"] if stage != "stage1" or r["source"] != src]
            result = score(train, test)
            report[key][src] = result
            print(stage, src, json.dumps(result), flush=True)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2, allow_nan=False), encoding="utf-8")
    print("Wrote", args.out, args.out.stat().st_size, "bytes")


if __name__ == "__main__":
    main()
