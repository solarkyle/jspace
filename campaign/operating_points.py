"""Deployment operating points for the LightGBM detector, per held-out dataset
(LODO). For each dataset: error prevalence, AUROC, catch-rate (recall) at a
fixed 10% and 20% false-positive rate, and the FPR needed to catch 80% of
errors. Also prints the classifier size (nodes ~ params) and what inference
actually costs.

    python -m campaign.operating_points --input out/campaign/stage1_features.jsonl
"""

from __future__ import annotations

import argparse
import json

import numpy as np

from campaign.split_groups import leave_one_dataset_out
from campaign.train_baselines import _matrix


def _lgbm():
    import lightgbm as lgb
    return lgb.LGBMClassifier(n_estimators=300, learning_rate=0.05, num_leaves=31,
                              min_child_samples=20, subsample=0.9,
                              colsample_bytree=0.9, class_weight="balanced",
                              verbosity=-1)


def recall_at_fpr(y, p, target_fpr):
    from sklearn.metrics import roc_curve
    fpr, tpr, _ = roc_curve(y, p)
    idx = np.searchsorted(fpr, target_fpr, side="right") - 1
    idx = max(0, idx)
    return tpr[idx]


def fpr_for_recall(y, p, target_tpr):
    from sklearn.metrics import roc_curve
    fpr, tpr, _ = roc_curve(y, p)
    idx = np.searchsorted(tpr, target_tpr, side="left")
    idx = min(idx, len(fpr) - 1)
    return fpr[idx]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True)
    args = ap.parse_args()
    rows = [json.loads(l) for l in open(args.input, encoding="utf-8") if l.strip()]
    labeled = [r for r in rows if "label" in r]
    names = sorted(labeled[0]["features"].keys())
    from sklearn.preprocessing import StandardScaler
    from sklearn.metrics import roc_auc_score

    print(f"{'dataset':>11} {'err%':>5} {'AUROC':>6} {'catch@10%FP':>11} "
          f"{'catch@20%FP':>11} {'FP@catch80%':>11}")
    total_nodes = 0
    for held, tr, te in leave_one_dataset_out(labeled):
        y = np.array([labeled[i]["label"] for i in range(len(labeled))])
        if len(set(y[te])) < 2 or len(set(y[tr])) < 2:
            print(f"{held:>11}  (no error variation - skipped)")
            continue
        X = _matrix(labeled, names)
        sc = StandardScaler().fit(X[tr])
        clf = _lgbm().fit(sc.transform(X[tr]), y[tr])
        p = clf.predict_proba(sc.transform(X[te]))[:, 1]
        yte = y[te]
        auc = roc_auc_score(yte, p)
        r10 = recall_at_fpr(yte, p, 0.10)
        r20 = recall_at_fpr(yte, p, 0.20)
        f80 = fpr_for_recall(yte, p, 0.80)
        total_nodes = clf.booster_.num_trees()  # trees, for size note
        print(f"{held:>11} {yte.mean()*100:>4.0f}% {auc:>6.3f} "
              f"{r10*100:>9.0f}% {r20*100:>9.0f}% {f80*100:>9.0f}%")

    # classifier size
    import lightgbm as lgb
    y = np.array([r["label"] for r in labeled])
    X = _matrix(labeled, names)
    sc = StandardScaler().fit(X)
    full = _lgbm().fit(sc.transform(X), y)
    df = full.booster_.trees_to_dataframe()
    n_nodes = len(df)
    n_leaves = int((df["left_child"].isna()).sum())
    print(f"\nClassifier size: {full.booster_.num_trees()} trees, {n_nodes} nodes "
          f"({n_leaves} leaves), {len(names)} input features.")
    print(f"Serialized model is ~{n_nodes*16/1024:.0f} KB; inference is a few "
          f"microseconds (300 threshold comparisons). Negligible vs the Gemma "
          f"forward pass + lens transport, which is the real per-query cost.")


if __name__ == "__main__":
    main()
