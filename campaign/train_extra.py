"""Control classifiers: CatBoost and a small MLP (handoff 12.3, 12.5).

Both are expected to LOSE or tie, and that expectation is the point:
  - CatBoost's edge is categorical handling, but we ban dataset/domain
    categoricals from deployable features, so it should ~= LightGBM.
  - A tabular MLP at this scale should not beat gradient-boosted trees; the MLP
    mainly exists as the distillation student later.

Same feature families, folds, and LODO metric as train_baselines.py.

    python -m campaign.train_extra --input out/campaign/stage1_features.jsonl
"""

from __future__ import annotations

import argparse
import json

import numpy as np

from campaign.split_groups import (grouped_kfold, leave_one_dataset_out,
                                    assert_no_group_leak)
from campaign.train_baselines import _split_families, _matrix, _auc


def _fit_predict(Xtr, ytr, Xte, kind):
    from sklearn.preprocessing import StandardScaler
    sc = StandardScaler().fit(Xtr)
    Xtr, Xte = sc.transform(Xtr), sc.transform(Xte)
    if kind == "catboost":
        from catboost import CatBoostClassifier
        clf = CatBoostClassifier(iterations=500, depth=6, learning_rate=0.05,
                                 l2_leaf_reg=3.0, auto_class_weights="Balanced",
                                 verbose=False, random_seed=0)
        clf.fit(Xtr, ytr)
        return clf.predict_proba(Xte)[:, 1]
    from sklearn.neural_network import MLPClassifier
    clf = MLPClassifier(hidden_layer_sizes=(128, 64), alpha=1e-3,
                        max_iter=500, early_stopping=True, random_state=0)
    clf.fit(Xtr, ytr)
    return clf.predict_proba(Xte)[:, 1]


def pooled_cv(rows, names, kind, k=5):
    y = np.array([r["label"] for r in rows]); X = _matrix(rows, names)
    oof = np.full(len(y), np.nan)
    for tr, te in grouped_kfold(rows, k):
        assert_no_group_leak(rows, tr, te)
        oof[te] = _fit_predict(X[tr], y[tr], X[te], kind)
    return _auc(y, oof)


def lodo(rows, names, kind):
    y = np.array([r["label"] for r in rows]); X = _matrix(rows, names)
    per = {}
    for held, tr, te in leave_one_dataset_out(rows):
        if len(set(y[te])) < 2 or len(set(y[tr])) < 2:
            per[held] = float("nan"); continue
        per[held] = _auc(y[te], _fit_predict(X[tr], y[tr], X[te], kind))
    vals = [v for v in per.values() if not np.isnan(v)]
    return (float(np.mean(vals)) if vals else float("nan")), per


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True)
    ap.add_argument("--out", default="")
    args = ap.parse_args()
    rows = [json.loads(l) for l in open(args.input, encoding="utf-8") if l.strip()]
    labeled = [r for r in rows if "label" in r]
    names = sorted(labeled[0]["features"].keys())
    fams = _split_families(names)
    report = {"pooled": {}, "lodo": {}}
    for kind in ("catboost", "mlp"):
        print(f"\n== {kind} ==")
        print(f"{'family':>10} {'pooled_grpCV':>13} {'LODO_mean':>10}")
        for fam, fn in fams.items():
            pc = pooled_cv(labeled, fn, kind)
            lm, _ = lodo(labeled, fn, kind)
            report["pooled"][f"{kind}:{fam}"] = round(pc, 4)
            report["lodo"][f"{kind}:{fam}"] = round(lm, 4)
            print(f"{fam:>10} {pc:>13.4f} {lm:>10.4f}")
        inc = report["lodo"][f"{kind}:combined"] - report["lodo"][f"{kind}:logprob"]
        print(f"  workspace increment LODO: {inc:+.4f}")
    if args.out:
        json.dump(report, open(args.out, "w"), indent=1); print("wrote", args.out)


if __name__ == "__main__":
    main()
