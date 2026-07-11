"""Flatten graded campaign traces into a feature table for the bakeoff.

Emits one row per response with:
  features      deployable flat dict (NO dataset/domain identity columns)
  label         1 = eventual_response_error, 0 = correct, absent if unresolved
  split_group, source_dataset, domain, upstream_group   (for splitting/reporting
                only; never fed to a deployable classifier)
  layer_entropies_onset, prefix_entropy_traj            (kept raw for temporal)

Deployable feature families (handoff 9.2):
  logprob   : first/mean/min answer logprob, answer length
  workspace : onset ignition/rank/agreement/entropy/hedge + shape-mass means
  entropy_trajectory : early/mid/late means, slope, std, max layer jump
  prefix    : onset->mid->end deltas of rank/entropy/ignition, max-risk prefix

Usage:
    python -m campaign.build_feature_table --input out/campaign/pilot_graded.jsonl \
        --out out/campaign/pilot_features.jsonl
"""

from __future__ import annotations

import argparse
import json

import numpy as np

WS_SCALARS = ["ignition_frac", "ignition_depth", "mean_log_rank_answer",
              "band_agreement", "mean_entropy", "best_hedge_rank_log"]


def _traj_summary(entropies: list[float]) -> dict:
    e = np.array(entropies, dtype=float)
    n = len(e)
    third = max(1, n // 3)
    early, mid, late = e[:third].mean(), e[third:2 * third].mean(), e[-third:].mean()
    slope = float(np.polyfit(np.arange(n), e, 1)[0]) if n > 1 else 0.0
    jumps = np.abs(np.diff(e)) if n > 1 else np.array([0.0])
    return {
        "ent_early": float(early), "ent_mid": float(mid), "ent_late": float(late),
        "ent_slope": slope, "ent_std": float(e.std()),
        "ent_max_jump": float(jumps.max()), "ent_n_dir_changes":
            float(int(np.sum(np.diff(np.sign(np.diff(e))) != 0))) if n > 2 else 0.0,
    }


def _shape_means(shape: dict) -> dict:
    out = {}
    for k in ("top1_p", "rival_mass", "tail_mass", "eff_k20"):
        v = shape.get(k, [])
        out[f"shape_{k}_mean"] = float(np.mean(v)) if v else 0.0
        out[f"shape_{k}_late"] = float(v[-1]) if v else 0.0
    return out


def build_features(row: dict) -> dict:
    f = {}
    f.update({k: float(v) for k, v in row["logprob_features"].items()})
    onset = row["onset_workspace_features"]
    for k in WS_SCALARS:
        f[f"ws_{k}"] = float(onset[k])
    f.update(_shape_means(onset.get("shape", {})))
    f.update(_traj_summary(onset["layer_entropies"]))

    # prefix evolution: onset -> mid -> end
    prefix = row.get("prefix_workspace_features", [])
    if len(prefix) >= 2:
        risk = [p["mean_log_rank_answer"] for p in prefix]   # higher = riskier
        ent = [p["mean_entropy"] for p in prefix]
        ign = [p["ignition_frac"] for p in prefix]
        f["pfx_rank_delta"] = float(risk[-1] - risk[0])
        f["pfx_rank_max"] = float(max(risk))
        f["pfx_ent_delta"] = float(ent[-1] - ent[0])
        f["pfx_ent_max"] = float(max(ent))
        f["pfx_ign_delta"] = float(ign[-1] - ign[0])
        f["pfx_ign_min"] = float(min(ign))
    else:
        for k in ("pfx_rank_delta", "pfx_rank_max", "pfx_ent_delta",
                  "pfx_ent_max", "pfx_ign_delta", "pfx_ign_min"):
            f[k] = float(row["onset_workspace_features"][
                "mean_log_rank_answer" if "rank" in k else
                "mean_entropy" if "ent" in k else "ignition_frac"])
    return f


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    rows = [json.loads(l) for l in open(args.input, encoding="utf-8") if l.strip()]
    n_lab, n_unres = 0, 0
    with open(args.out, "w", encoding="utf-8") as fo:
        for r in rows:
            grade = r.get("deterministic_grade", {})
            correct = grade.get("correct")
            out = {
                "example_id": r["example_id"],
                "split_group": r["split_group"],
                "source_dataset": r["source_dataset"],
                "domain": r["domain"],
                "upstream_group": r["upstream_group"],
                "features": build_features(r),
                "layer_entropies_onset": r["onset_workspace_features"]["layer_entropies"],
                "prefix_entropy_traj": [p["mean_entropy"]
                                        for p in r.get("prefix_workspace_features", [])],
                "abstained": grade.get("abstained", False),
            }
            if correct is not None:
                out["label"] = int(not correct)   # 1 = error
                n_lab += 1
            else:
                n_unres += 1
            fo.write(json.dumps(out) + "\n")
    print(f"{len(rows)} rows -> {args.out}")
    print(f"  labeled: {n_lab}  unresolved(need judge): {n_unres}")
    n_feat = len(build_features(rows[0]))
    print(f"  {n_feat} deployable features/row")


if __name__ == "__main__":
    main()
