"""TabFM leg of the campaign bakeoff (prereg registered comparison).

Same feature families, folds, and metrics as train_baselines.py, but with the
frozen TabFM config (n_estimators=32). Registered expectation (PREREG_STAGE1):
TabFM's edge over LightGBM shrinks as rows grow; a clear TabFM LODO win at Stage
1 scale is a reportable surprise, not a silent adoption.

Must run from the TabFM venv:
    C:/Users/18632/Desktop/stuff/ufc_bet/.venv-tabfm/Scripts/python.exe \
        -m campaign.train_tabfm --input out/campaign/pilot_features.jsonl
"""

from __future__ import annotations

import argparse
import json

import numpy as np

from campaign.split_groups import (grouped_kfold, leave_one_dataset_out,
                                    assert_no_group_leak)
from campaign.train_baselines import LOGPROB, _split_families, _matrix, _auc


def make_tabfm():
    import torch
    from safetensors.torch import load_file
    from tabfm import TabFMClassifier
    from tabfm.src.pytorch.model import TabFM
    from tabfm.src.pytorch.tabfm_v1_0_0 import ClassificationConfig
    m = TabFM(**ClassificationConfig().to_dict())
    m.load_state_dict(load_file(
        "C:/Users/18632/Desktop/tabfm/classification/model.safetensors"), strict=False)
    m = m.to("cuda" if torch.cuda.is_available() else "cpu").eval()
    return lambda: TabFMClassifier(model=m, n_estimators=32, random_state=0)


def _fit_predict_tabfm(make_clf, Xtr, ytr, Xte):
    from sklearn.preprocessing import StandardScaler
    sc = StandardScaler().fit(Xtr)
    clf = make_clf()
    clf.fit(sc.transform(Xtr), ytr)
    return clf.predict_proba(sc.transform(Xte))[:, 1]


def pooled_cv(rows, names, make_clf, k=5):
    y = np.array([r["label"] for r in rows])
    X = _matrix(rows, names)
    oof = np.full(len(y), np.nan)
    for tr, te in grouped_kfold(rows, k):
        assert_no_group_leak(rows, tr, te)
        oof[te] = _fit_predict_tabfm(make_clf, X[tr], y[tr], X[te])
    return _auc(y, oof)


def lodo(rows, names, make_clf):
    y = np.array([r["label"] for r in rows])
    X = _matrix(rows, names)
    per = {}
    for held, tr, te in leave_one_dataset_out(rows):
        if len(set(y[te])) < 2 or len(set(y[tr])) < 2:
            per[held] = float("nan"); continue
        p = _fit_predict_tabfm(make_clf, X[tr], y[tr], X[te])
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
    print(f"{len(labeled)} labeled rows")
    names = sorted(labeled[0]["features"].keys())
    fams = _split_families(names)
    make_clf = make_tabfm()

    report = {"pooled": {}, "lodo": {}, "lodo_per": {}}
    print(f"\n== TabFM (n_estimators=32) ==")
    print(f"{'family':>10} {'pooled_grpCV':>13} {'LODO_mean':>10}")
    for fam, fn in fams.items():
        pc = pooled_cv(labeled, fn, make_clf)
        lm, per = lodo(labeled, fn, make_clf)
        report["pooled"][fam] = round(pc, 4)
        report["lodo"][fam] = round(lm, 4)
        report["lodo_per"][fam] = {k: round(v, 3) for k, v in per.items()}
        print(f"{fam:>10} {pc:>13.4f} {lm:>10.4f}")
    inc_p = report["pooled"]["combined"] - report["pooled"]["logprob"]
    inc_l = report["lodo"]["combined"] - report["lodo"]["logprob"]
    print(f"  workspace increment: pooled {inc_p:+.4f}   LODO {inc_l:+.4f}")

    if args.out:
        json.dump(report, open(args.out, "w"), indent=1)
        print("wrote", args.out)


if __name__ == "__main__":
    main()
