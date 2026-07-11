"""Freeze the Stage 1 production classifiers for prospective validation.

Trains the registered LightGBM config on ALL corrected Stage 1 labeled rows,
once per feature family (combined / logprob / workspace -- the increment
comparison needs all three frozen), and exports an immutable artifact set:

    out/campaign/frozen/lgbm_stage1_<family>.txt   LightGBM boosters
    out/campaign/frozen/frozen_meta.json           feature schema per family,
                                                   scaler mean/scale, training
                                                   row counts, sha256 per model

Stage 2 scores these artifacts zero-shot on datasets they have never seen.
Nothing here may change after the Stage 2 prereg records the hashes.

    python -m campaign.freeze_classifier --input out/campaign/stage1_features.jsonl
    python -m campaign.freeze_classifier --check   # verify hashes + reload
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os

import numpy as np

from campaign.train_baselines import LOGPROB, _matrix

HERE = os.path.dirname(__file__)
FROZEN_DIR = os.path.join(HERE, "..", "out", "campaign", "frozen")
META_PATH = os.path.join(FROZEN_DIR, "frozen_meta.json")
FAMILIES = ("combined", "logprob", "workspace")


def _model_path(family):
    return os.path.join(FROZEN_DIR, f"lgbm_stage1_{family}.txt")


def _lgbm():
    import lightgbm as lgb
    return lgb.LGBMClassifier(n_estimators=300, learning_rate=0.05,
                              num_leaves=31, min_child_samples=20,
                              subsample=0.9, colsample_bytree=0.9,
                              class_weight="balanced", verbosity=-1,
                              random_state=0, deterministic=True,
                              force_row_wise=True)


def _sha256(path):
    return hashlib.sha256(open(path, "rb").read()).hexdigest()


def _family_names(all_names):
    logprob = [n for n in all_names if n in LOGPROB]
    workspace = [n for n in all_names if n not in LOGPROB]
    return {"combined": list(all_names), "logprob": logprob,
            "workspace": workspace}


def freeze(args):
    rows = [json.loads(l) for l in open(args.input, encoding="utf-8") if l.strip()]
    labeled = [r for r in rows if "label" in r]
    all_names = sorted(labeled[0]["features"].keys())
    y = np.array([r["label"] for r in labeled])

    os.makedirs(FROZEN_DIR, exist_ok=True)
    for fam in FAMILIES:
        if os.path.exists(_model_path(fam)) and not args.force:
            raise SystemExit(f"{_model_path(fam)} already exists; refusing to "
                             f"overwrite a frozen artifact (--force only "
                             f"pre-prereg)")

    from sklearn.preprocessing import StandardScaler
    by_ds = {}
    for r in labeled:
        by_ds[r["source_dataset"]] = by_ds.get(r["source_dataset"], 0) + 1
    meta = {
        "model": "lightgbm",
        "config": {"n_estimators": 300, "learning_rate": 0.05, "num_leaves": 31,
                   "min_child_samples": 20, "subsample": 0.9,
                   "colsample_bytree": 0.9, "class_weight": "balanced",
                   "random_state": 0, "deterministic": True},
        "n_train_rows": int(len(labeled)),
        "train_rows_by_dataset": by_ds,
        "train_error_rate": float(y.mean()),
        "train_input": os.path.basename(args.input),
        "families": {},
    }
    fam_names = _family_names(all_names)
    for fam in FAMILIES:
        names = fam_names[fam]
        X = _matrix(labeled, names)
        sc = StandardScaler().fit(X)
        clf = _lgbm().fit(sc.transform(X), y)
        clf.booster_.save_model(_model_path(fam))
        meta["families"][fam] = {
            "feature_names": names,
            "scaler_mean": sc.mean_.tolist(),
            "scaler_scale": sc.scale_.tolist(),
            "model_sha256": _sha256(_model_path(fam)),
        }
        print(f"frozen [{fam:>9}]: {len(names)} features, "
              f"sha256 {meta['families'][fam]['model_sha256'][:16]}...")

    json.dump(meta, open(META_PATH, "w", encoding="utf-8"), indent=1)
    print(f"trained on {len(labeled)} rows, error rate {y.mean():.3f}")
    print(f"meta: {META_PATH}")


def load_frozen(family="combined"):
    """Load a frozen artifact; returns (booster, family_meta). Verifies hash."""
    import lightgbm as lgb
    meta = json.load(open(META_PATH, encoding="utf-8"))
    fm = meta["families"][family]
    h = _sha256(_model_path(family))
    if h != fm["model_sha256"]:
        raise RuntimeError(f"frozen model hash mismatch for {family}: {h} != "
                           f"{fm['model_sha256']} - artifact was modified")
    return lgb.Booster(model_file=_model_path(family)), fm


def score_rows(rows, family="combined"):
    """Score feature-table rows with a frozen artifact. Returns np.array of
    error probabilities. Missing features are an error (schema is frozen)."""
    booster, fm = load_frozen(family)
    names = fm["feature_names"]
    missing = [n for n in names if n not in rows[0]["features"]]
    if missing:
        raise RuntimeError(f"rows lack frozen features: {missing}")
    X = _matrix(rows, names)
    Xs = (X - np.array(fm["scaler_mean"])) / np.array(fm["scaler_scale"])
    return booster.predict(Xs)


def check(_args):
    meta = json.load(open(META_PATH, encoding="utf-8"))
    for fam in FAMILIES:
        booster, fm = load_frozen(fam)
        print(f"[{fam:>9}] hash OK {fm['model_sha256'][:16]}... "
              f"{booster.num_trees()} trees, {len(fm['feature_names'])} features")
    print(f"trained on {meta['n_train_rows']} rows "
          f"({json.dumps(meta['train_rows_by_dataset'])})")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", default="out/campaign/stage1_features.jsonl")
    ap.add_argument("--check", action="store_true")
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args()
    (check if args.check else freeze)(args)


if __name__ == "__main__":
    main()
