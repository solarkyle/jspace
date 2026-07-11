"""Score the pre-registered gates (campaign/PREREG_STAGE1.md) from a bakeoff
report. Gate A is the decision: does combined beat logprob-only on mean LODO
AUROC by >= +0.02, with the increment positive on >= 70% of held-out datasets,
on LightGBM (the intended production model)?

    python -m campaign.score_gates --bakeoff out/campaign/stage1_bakeoff.json
"""

from __future__ import annotations

import argparse
import json


def score_gate_a(report, model="lightgbm"):
    lodo = report["lodo"]
    per = report["lodo_per"]
    comb = lodo[f"{model}:combined"]
    lp = lodo[f"{model}:logprob"]
    mean_inc = comb - lp
    per_c = per[f"{model}:combined"]
    per_l = per[f"{model}:logprob"]
    datasets = [d for d in per_c if d in per_l
                and per_c[d] == per_c[d] and per_l[d] == per_l[d]]  # drop NaN
    pos = [d for d in datasets if per_c[d] - per_l[d] > 0]
    breadth = len(pos) / len(datasets) if datasets else 0.0
    if mean_inc >= 0.02 and breadth >= 0.70:
        verdict = "HIT"
    elif mean_inc >= 0.02:
        verdict = "PARTIAL (mean passes, breadth fails)"
    else:
        verdict = "MISS"
    return {
        "model": model, "mean_lodo_increment": round(mean_inc, 4),
        "breadth_positive": round(breadth, 3),
        "datasets_positive": f"{len(pos)}/{len(datasets)}",
        "per_dataset_increment": {d: round(per_c[d] - per_l[d], 3) for d in datasets},
        "verdict": verdict,
    }


def workspace_only_note(report, model="lightgbm"):
    """The pilot found workspace-only can beat combined cross-dataset."""
    lodo = report["lodo"]
    return {
        "logprob_lodo": lodo[f"{model}:logprob"],
        "workspace_lodo": lodo[f"{model}:workspace"],
        "combined_lodo": lodo[f"{model}:combined"],
        "workspace_over_logprob": round(
            lodo[f"{model}:workspace"] - lodo[f"{model}:logprob"], 4),
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--bakeoff", required=True)
    args = ap.parse_args()
    report = json.load(open(args.bakeoff))
    print("== Gate A (pre-registered) ==")
    for model in ("lightgbm", "logistic"):
        if f"{model}:combined" in report["lodo"]:
            g = score_gate_a(report, model)
            print(f"\n[{model}] verdict: {g['verdict']}")
            print(f"  mean LODO increment: {g['mean_lodo_increment']:+.4f} (need >= +0.02)")
            print(f"  breadth positive: {g['datasets_positive']} = {g['breadth_positive']} (need >= 0.70)")
            print(f"  per-dataset: {json.dumps(g['per_dataset_increment'])}")
            print(f"  workspace-only cross-dataset: {json.dumps(workspace_only_note(report, model))}")


if __name__ == "__main__":
    main()
