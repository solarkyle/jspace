"""Static classifier bakeoff on the campaign feature table (handoff 12).

The pilot question is Gate A in miniature: does workspace add value over output
confidence, both pooled-grouped and leave-one-dataset-out? Runs three feature
families (logprob / workspace / combined) through logistic and LightGBM, under
pooled grouped 5-fold AND LODO. Reports AUROC and the workspace increment.

TabFM is scored separately (its own venv); this covers the free local models.

Usage:
    python -m campaign.train_baselines --input out/campaign/pilot_features.jsonl
"""

from __future__ import annotations

import argparse
import json

import numpy as np

from campaign.split_groups import (grouped_kfold, leave_one_dataset_out,
                                    assert_no_group_leak, dataset_identity_leakage)

LOGPROB = ["bl_first_token_logprob", "bl_mean_logprob", "bl_min_logprob", "bl_answer_len"]


def _split_families(feat_names):
    logprob = [f for f in feat_names if f in LOGPROB]
    workspace = [f for f in feat_names if f not in LOGPROB]
    return {"logprob": logprob, "workspace": workspace, "combined": feat_names}


def _matrix(rows, names):
    return np.array([[r["features"][n] for n in names] for r in rows])


def _auc(y, p):
    from sklearn.metrics import roc_auc_score
    return roc_auc_score(y, p) if len(set(y)) > 1 else float("nan")


def _fit_predict(Xtr, ytr, Xte, kind):
    from sklearn.preprocessing import StandardScaler
    sc = StandardScaler().fit(Xtr)
    Xtr, Xte = sc.transform(Xtr), sc.transform(Xte)
    if kind == "logistic":
        from sklearn.linear_model import LogisticRegression
        clf = LogisticRegression(max_iter=2000, class_weight="balanced").fit(Xtr, ytr)
        return clf.predict_proba(Xte)[:, 1]
    import lightgbm as lgb
    clf = lgb.LGBMClassifier(n_estimators=300, learning_rate=0.05, num_leaves=31,
                             min_child_samples=20, subsample=0.9, colsample_bytree=0.9,
                             class_weight="balanced", verbosity=-1).fit(Xtr, ytr)
    return clf.predict_proba(Xte)[:, 1]


def pooled_cv(rows, names, kind, k=5):
    y = np.array([r["label"] for r in rows])
    X = _matrix(rows, names)
    oof = np.full(len(y), np.nan)
    for tr, te in grouped_kfold(rows, k):
        assert_no_group_leak(rows, tr, te)
        oof[te] = _fit_predict(X[tr], y[tr], X[te], kind)
    return _auc(y, oof)


def lodo(rows, names, kind):
    y = np.array([r["label"] for r in rows])
    X = _matrix(rows, names)
    per = {}
    for held, tr, te in leave_one_dataset_out(rows):
        if len(set(y[te])) < 2 or len(set(y[tr])) < 2:
            per[held] = float("nan"); continue
        p = _fit_predict(X[tr], y[tr], X[te], kind)
        per[held] = _auc(y[te], p)
    vals = [v for v in per.values() if not np.isnan(v)]
    return (float(np.mean(vals)) if vals else float("nan")), per


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True)
    ap.add_argument("--out", default="")
    args = ap.parse_args()

    rows = [json.loads(l) for l in open(args.input, encoding="utf-8") if l.strip()]
    labeled = [r for r in rows if "label" in r]
    print(f"{len(rows)} rows, {len(labeled)} labeled; "
          f"error rate {np.mean([r['label'] for r in labeled]):.3f}")
    by_src = {}
    for r in labeled:
        by_src.setdefault(r["source_dataset"], []).append(r["label"])
    for s, ys in sorted(by_src.items()):
        print(f"  {s:>12}: n={len(ys):>3} err={np.mean(ys):.2f}")

    names = sorted(labeled[0]["features"].keys())
    fams = _split_families(names)
    report = {"n_labeled": len(labeled), "pooled": {}, "lodo": {}, "lodo_per": {}}

    for kind in ("logistic", "lightgbm"):
        print(f"\n== {kind} ==")
        print(f"{'family':>10} {'pooled_grpCV':>13} {'LODO_mean':>10}")
        for fam, fn in fams.items():
            pc = pooled_cv(labeled, fn, kind)
            lm, per = lodo(labeled, fn, kind)
            report["pooled"][f"{kind}:{fam}"] = round(pc, 4)
            report["lodo"][f"{kind}:{fam}"] = round(lm, 4)
            report["lodo_per"][f"{kind}:{fam}"] = {k: round(v, 3) for k, v in per.items()}
            print(f"{fam:>10} {pc:>13.4f} {lm:>10.4f}")
        inc_p = report["pooled"][f"{kind}:combined"] - report["pooled"][f"{kind}:logprob"]
        inc_l = report["lodo"][f"{kind}:combined"] - report["lodo"][f"{kind}:logprob"]
        print(f"  workspace increment: pooled {inc_p:+.4f}   LODO {inc_l:+.4f}")

    leak = dataset_identity_leakage(labeled)
    report["identity_leakage"] = leak
    print("\ndataset-identity leakage:", json.dumps(leak))

    if args.out:
        json.dump(report, open(args.out, "w"), indent=1)
        print("wrote", args.out)


if __name__ == "__main__":
    main()
