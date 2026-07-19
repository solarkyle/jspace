"""TabFM (Google's tabular foundation model) on the workspace traces.

Question: how much signal do the hand-picked scalar features leave behind?
Feed TabFM the FULL layerwise entropy trajectory (resampled to 16 points) +
baselines + workspace scalars, predict WRONG, 5-fold CV, compare against
logistic regression on the same folds and feature sets.

Requires a venv with the TabFM dependencies installed:
    python analysis/analyze_tabfm.py

Loads TabFM from local safetensors.
"""

import json
import os
import time

import numpy as np

WEIGHTS = os.environ.get(
    "TABFM_WEIGHTS", "C:/Users/18632/Desktop/tabfm/classification/model.safetensors")
TRACE = "data/uncertainty_trivia_{}.jsonl"
MODELS = ["gemma-4-e4b-it", "gemma-4-12b-it", "huihui-gemma-4-12b-it-abliterated",
          "gemma-4-26b-a4b-it", "qwen3.6-27b"]
N_TRAJ = 16

BASELINE = ["bl_first_token_logprob", "bl_mean_logprob", "bl_min_logprob", "bl_answer_len"]
WS_SCALARS = ["ignition_frac", "ignition_depth", "mean_log_rank_answer",
              "band_agreement", "mean_entropy", "best_hedge_rank_log"]


def load_rows(model: str) -> list[dict]:
    return [json.loads(l) for l in open(TRACE.format(model), encoding="utf-8") if l.strip()]


def featurize(rows: list[dict]):
    def traj(r):
        e = np.asarray(r["layer_entropies"], dtype=float)
        x = np.linspace(0, 1, len(e))
        return np.interp(np.linspace(0, 1, N_TRAJ), x, e)

    X_bl = np.array([[r[k] for k in BASELINE] for r in rows])
    X_ws = np.hstack([
        np.array([[r[k] for k in WS_SCALARS] for r in rows]),
        np.array([traj(r) for r in rows]),
    ])
    y = np.array([0 if r["correct"] else 1 for r in rows])
    return {"logprob": X_bl, "workspace": X_ws,
            "combined": np.hstack([X_bl, X_ws])}, y


def make_tabfm():
    import torch
    from safetensors.torch import load_file
    from tabfm import TabFMClassifier
    from tabfm.src.pytorch.model import TabFM
    from tabfm.src.pytorch.tabfm_v1_0_0 import ClassificationConfig

    model = TabFM(**ClassificationConfig().to_dict())
    miss, unexp = model.load_state_dict(load_file(WEIGHTS), strict=False)
    assert not miss and not unexp, "state_dict mismatch"
    device = "cuda" if torch.cuda.is_available() else "cpu"
    # keep VRAM tiny; a local probe run may own the card
    try:
        free = torch.cuda.mem_get_info()[0] / 1e9 if device == "cuda" else 0
        if device == "cuda" and free < 3.0:
            device = "cpu"
    except Exception:  # noqa: BLE001
        device = "cpu"
    model = model.to(device).eval()
    return lambda: TabFMClassifier(model=model, n_estimators=8, random_state=0)


def cv_auc_tabfm(make_clf, X, y, k=5, seed=0):
    from sklearn.metrics import roc_auc_score
    from sklearn.model_selection import StratifiedKFold

    oof = np.zeros(len(y))
    for tr, te in StratifiedKFold(k, shuffle=True, random_state=seed).split(X, y):
        clf = make_clf()
        clf.fit(X[tr], y[tr])
        oof[te] = clf.predict_proba(X[te])[:, 1]
    return roc_auc_score(y, oof), oof


def cv_auc_logistic(X, y, k=5, seed=0):
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import roc_auc_score
    from sklearn.model_selection import StratifiedKFold
    from sklearn.preprocessing import StandardScaler

    oof = np.zeros(len(y))
    for tr, te in StratifiedKFold(k, shuffle=True, random_state=seed).split(X, y):
        sc = StandardScaler().fit(X[tr])
        clf = LogisticRegression(max_iter=2000).fit(sc.transform(X[tr]), y[tr])
        oof[te] = clf.predict_proba(sc.transform(X[te]))[:, 1]
    return roc_auc_score(y, oof), oof


def main() -> None:
    make_clf = make_tabfm()
    print(f"{'model':>36} {'set':>10} {'logistic':>9} {'TabFM':>7} {'delta':>7}")
    results = {}
    for m in MODELS:
        try:
            rows = load_rows(m)
        except FileNotFoundError:
            continue
        sets, y = featurize(rows)
        for name, X in sets.items():
            t0 = time.time()
            auc_lr, _ = cv_auc_logistic(X, y)
            auc_tf, oof = cv_auc_tabfm(make_clf, X, y)
            results[(m, name)] = {"logistic": auc_lr, "tabfm": auc_tf}
            print(f"{m:>36} {name:>10} {auc_lr:>9.3f} {auc_tf:>7.3f} "
                  f"{auc_tf - auc_lr:>+7.3f}  ({time.time()-t0:.0f}s)")
    with open("out/tabfm_results.json", "w", encoding="utf-8") as f:
        json.dump({f"{m}|{s}": v for (m, s), v in results.items()}, f, indent=1)
    print("wrote out/tabfm_results.json")


if __name__ == "__main__":
    main()
