"""Score Gate C: does a mid-answer prefix preserve the full-answer increment?

Registered criterion (campaign/PREREG_STAGE1.md):

    the risk score from the 50 percent answer-prefix preserves at least 90
    percent of the full-answer LODO AUROC increment. Falsified if the 50 percent
    prefix retains < 90 percent.

"Increment" throughout the campaign means workspace-over-logprob, so the ratio is
computed between two self-consistent pairs:

    full   increment = AUROC(full LP + full workspace)      - AUROC(full LP)
    prefix increment = AUROC(prefix LP + workspace @ 50%)   - AUROC(prefix LP)
    ratio            = prefix increment / full increment

The prefix arm uses ONLY pfxlp_* features, which aggregate tokens 0..k. The
original campaign had no such features -- it fed whole-answer logprobs and the
final answer length to every checkpoint -- which is why this gate could not be
scored from the existing traces.

Everything is leave-one-dataset-out. Pooled numbers are not reported: Stage 1
measured identity leakage of 0.811 vs a 0.185 majority, and the registered
interpretation says pooled numbers are not to be trusted.

Usage:
    python -m campaign.score_gate_c --input out/campaign/gatec_features.jsonl \
        --out out/campaign/gate_c.json
"""

import argparse
import collections
import io
import json
import os

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

SEED = 20260910
WS_SCALARS = ("ignition_frac", "ignition_depth", "mean_log_rank_answer",
              "band_agreement", "mean_entropy", "best_hedge_rank_log")
PFX_LP = ("pfxlp_first", "pfxlp_last", "pfxlp_mean", "pfxlp_min", "pfxlp_tokens_so_far")
FULL_LP = ("bl_first_token_logprob", "bl_mean_logprob", "bl_min_logprob", "bl_answer_len")
MIN_TEST_ROWS = 30
MIN_MINORITY = 5


def model():
    return make_pipeline(
        StandardScaler(),
        LogisticRegression(C=1.0, max_iter=2000, class_weight="balanced",
                           random_state=SEED),
    )


def arms(row: dict) -> dict:
    """Four feature vectors per row, named by arm."""
    prefix = row["prefix_workspace_features"]
    mid = next((p for p in prefix if abs(float(p.get("frac", -1)) - 0.5) < 1e-9), None)
    end = next((p for p in prefix if abs(float(p.get("frac", -1)) - 1.0) < 1e-9), None)
    if mid is None or end is None:
        return {}
    lp_full = row["logprob_features"]
    return {
        "prefix_lp": [float(mid[k]) for k in PFX_LP],
        "prefix_lp_ws": [float(mid[k]) for k in PFX_LP] + [float(mid[k]) for k in WS_SCALARS],
        "full_lp": [float(lp_full[k]) for k in FULL_LP],
        "full_lp_ws": [float(lp_full[k]) for k in FULL_LP] + [float(end[k]) for k in WS_SCALARS],
    }


def lodo(rows: list, arm: str) -> dict:
    by_source = collections.defaultdict(list)
    for row in rows:
        by_source[row["source_dataset"]].append(row)
    out, skipped = {}, {}
    for held in sorted(by_source):
        test = by_source[held]
        train = [r for s, rs in by_source.items() if s != held for r in rs]
        y_test = np.array([int(r["_label"]) for r in test])
        if len(test) < MIN_TEST_ROWS or min(int(y_test.sum()), int((1 - y_test).sum())) < MIN_MINORITY:
            skipped[held] = (f"n={len(test)} pos={int(y_test.sum())} "
                             f"neg={int((1 - y_test).sum())} below floor")
            continue
        X_train = np.array([r["_arms"][arm] for r in train])
        y_train = np.array([int(r["_label"]) for r in train])
        X_test = np.array([r["_arms"][arm] for r in test])
        clf = model().fit(X_train, y_train)
        out[held] = float(roc_auc_score(y_test, clf.predict_proba(X_test)[:, 1]))
    return {"per_source": out, "mean": float(np.mean(list(out.values()))) if out else None,
            "skipped": skipped}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    rows = []
    dropped = collections.Counter()
    with io.open(args.input, encoding="utf-8") as fh:
        for line in fh:
            if not line.strip():
                continue
            row = json.loads(line)
            # Label from the FRESH grade of THIS run's answer. The manifest also
            # carries metadata.label_is_error from the July campaign, but greedy
            # regeneration reproduced only ~1/30 of those answers (the Modal image
            # pinned neither transformers nor a model revision), so a July label
            # does not describe a September answer. Using it would silently score
            # one run's features against another run's correctness.
            grade = row.get("deterministic_grade") or {}
            if grade.get("correct") is None:
                dropped["unlabeled_needs_judge"] += 1
                continue
            label_is_error = not bool(grade["correct"])
            a = arms(row)
            if not a:
                dropped["missing_checkpoint"] += 1
                continue
            row["_label"] = bool(label_is_error)
            row["_arms"] = a
            rows.append(row)

    results = {arm: lodo(rows, arm) for arm in ("prefix_lp", "prefix_lp_ws", "full_lp", "full_lp_ws")}
    full_inc = results["full_lp_ws"]["mean"] - results["full_lp"]["mean"]
    prefix_inc = results["prefix_lp_ws"]["mean"] - results["prefix_lp"]["mean"]
    ratio = (prefix_inc / full_inc) if full_inc and full_inc > 0 else None

    verdict = "UNDECIDABLE"
    note = ""
    if full_inc is not None and full_inc <= 0:
        note = ("full-answer increment is <= 0 on this long-answer subset, so the "
                "registered ratio has no denominator; Gate C cannot be scored here")
    elif ratio is not None:
        verdict = "HIT" if ratio >= 0.90 else "MISS"

    payload = {
        "gate": "C",
        "criterion": "50%-prefix increment >= 90% of full-answer increment (LODO mean)",
        "n_rows": len(rows),
        "n_sources_scored": len(results["full_lp_ws"]["per_source"]),
        "dropped": dict(dropped),
        "arms": results,
        "full_answer_increment": full_inc,
        "prefix_increment": prefix_inc,
        "retention_ratio": ratio,
        "verdict": verdict,
        "note": note,
        "caveats": [
            "LODO only; pooled numbers untrusted per registered identity-leakage rule.",
            "Restricted to answers >= 32 tokens: early warning is undefined where "
            "there is no room to warn (trivia_qa median ~4 tokens, legal never reaches 8).",
            "Logistic regression, no tuning, seed fixed. No bootstrap CIs here.",
            "Sources below the size/balance floor are skipped and listed, not imputed.",
            "Labels are fresh deterministic grades of this run's own answers. The "
            "July campaign's generations are not reproducible from the current "
            "image, so this is a new internally-consistent collection, not a "
            "re-analysis of the published traces.",
        ],
    }
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    with io.open(args.out, "w", encoding="utf-8") as fh:
        json.dump(payload, fh, indent=2)

    print(f"rows={len(rows)} dropped={dict(dropped)}")
    for arm, res in results.items():
        print(f"  {arm:14s} LODO mean {res['mean']} over {len(res['per_source'])} sources")
        if res["skipped"]:
            print(f"                 skipped: {res['skipped']}")
    print(f"full increment   {full_inc}")
    print(f"prefix increment {prefix_inc}")
    print(f"retention ratio  {ratio}  -> GATE C {verdict} {note}")


if __name__ == "__main__":
    main()
