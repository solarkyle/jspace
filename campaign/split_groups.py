"""Grouped splits and leakage tests (handoff section 11).

Three fold generators, all group-safe (no split_group or example_id spans
train and test):

  grouped_kfold(rows, k)         pooled grouped CV
  leave_one_dataset_out(rows)    train on all-but-one source, test on it
  leave_one_domain_out(rows)     same at the domain level

Plus dataset_identity_leakage(): trains a classifier to predict source_dataset
from the DEPLOYABLE feature set. High accuracy means the features encode
dataset style, which would inflate a pooled classifier. This is a guardrail,
not a model.

Prefixes and multiple generations inherit their parent example_id as the group
key, so they never straddle a split.

Usage (sanity check on a feature table):
    python -m campaign.split_groups --input out/campaign/pilot_features.jsonl
"""

from __future__ import annotations

import argparse
import json

import numpy as np


def _groups(rows):
    return np.array([r["split_group"] for r in rows])


def grouped_kfold(rows, k=5, seed=0):
    """Yield (train_idx, test_idx) with whole groups held out, label-balanced.
    StratifiedGroupKFold guarantees no split_group spans train and test."""
    from sklearn.model_selection import StratifiedGroupKFold
    y = np.array([int(r.get("label", 0)) for r in rows])
    g = _groups(rows)
    X = np.zeros((len(rows), 1))
    for tr, te in StratifiedGroupKFold(k, shuffle=True, random_state=seed).split(X, y, g):
        yield tr, te


def leave_one_dataset_out(rows):
    src = np.array([r["source_dataset"] for r in rows])
    for held in sorted(set(src)):
        te = np.where(src == held)[0]
        tr = np.where(src != held)[0]
        yield held, tr, te


def leave_one_domain_out(rows):
    dom = np.array([r["domain"] for r in rows])
    for held in sorted(set(dom)):
        te = np.where(dom == held)[0]
        tr = np.where(dom != held)[0]
        yield held, tr, te


def assert_no_group_leak(rows, tr, te):
    gtr = {rows[i]["split_group"] for i in tr}
    gte = {rows[i]["split_group"] for i in te}
    overlap = gtr & gte
    assert not overlap, f"group leak: {len(overlap)} groups in both splits"


def dataset_identity_leakage(rows, feature_key="features", k=5):
    """Can the deployable features predict which dataset a row came from?
    Reports multiclass accuracy vs the majority-class baseline. Big gap = leak."""
    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import StandardScaler
    from sklearn.model_selection import StratifiedKFold

    feats = sorted(rows[0][feature_key].keys())
    X = np.array([[r[feature_key][f] for f in feats] for r in rows])
    src = np.array([r["source_dataset"] for r in rows])
    classes, y = np.unique(src, return_inverse=True)
    baseline = np.bincount(y).max() / len(y)
    correct = 0
    for tr, te in StratifiedKFold(k, shuffle=True, random_state=0).split(X, y):
        sc = StandardScaler().fit(X[tr])
        clf = LogisticRegression(max_iter=2000).fit(sc.transform(X[tr]), y[tr])
        correct += (clf.predict(sc.transform(X[te])) == y[te]).sum()
    acc = correct / len(y)
    return {"identity_acc": round(float(acc), 3),
            "majority_baseline": round(float(baseline), 3),
            "n_datasets": len(classes),
            "leak_margin": round(float(acc - baseline), 3)}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True)
    args = ap.parse_args()
    rows = [json.loads(l) for l in open(args.input, encoding="utf-8") if l.strip()]

    print(f"{len(rows)} rows, {len({r['split_group'] for r in rows})} groups, "
          f"{len({r['source_dataset'] for r in rows})} datasets")
    for tr, te in grouped_kfold(rows, 5):
        assert_no_group_leak(rows, tr, te)
    print("grouped 5-fold: no group leak across any fold OK")
    for held, tr, te in leave_one_dataset_out(rows):
        print(f"  LODO hold={held:>12}: train={len(tr)} test={len(te)}")
    if "features" in rows[0]:
        print("identity leakage:", json.dumps(dataset_identity_leakage(rows)))


if __name__ == "__main__":
    main()
