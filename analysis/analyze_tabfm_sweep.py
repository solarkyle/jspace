"""Custom TabFM configuration sweep for the workspace-wrongness problem.

Not the UFC settings: this sweeps the knobs that matter HERE.
  - n_estimators: 8 / 32 / 64
  - trajectory: native band resolution vs resampled-16
  - scaling: raw features (TabFM does its own normalization) vs z-scored
  - features: base (scalars + trajectory) vs extended (+ adjacent-layer
    entropy deltas, collapse depth/index, min/max positions)

Protocol: 5-fold CV per model on the 5 existing models, combined feature set
only (the deployment set). The winning config gets FROZEN before the 31B
trace lands, which then serves as the untouched held-out model.

Requires a venv with the TabFM dependencies installed.
"""

import itertools
import json
import os

import numpy as np

WEIGHTS = os.environ.get(
    "TABFM_WEIGHTS", "tabfm/classification/model.safetensors")
TRACE = "data/uncertainty_trivia_{}.jsonl"
MODELS = ["gemma-4-e4b-it", "gemma-4-12b-it", "huihui-gemma-4-12b-it-abliterated",
          "gemma-4-26b-a4b-it", "qwen3.6-27b"]

BASELINE = ["bl_first_token_logprob", "bl_mean_logprob", "bl_min_logprob", "bl_answer_len"]
WS_SCALARS = ["ignition_frac", "ignition_depth", "mean_log_rank_answer",
              "band_agreement", "mean_entropy", "best_hedge_rank_log"]


def build_X(rows, traj_mode: str, extended: bool):
    n_native = len(rows[0]["layer_entropies"])

    def traj(r):
        e = np.asarray(r["layer_entropies"], dtype=float)
        if traj_mode == "native":
            return e
        return np.interp(np.linspace(0, 1, 16), np.linspace(0, 1, len(e)), e)

    cols = [np.array([[r[k] for k in BASELINE + WS_SCALARS] for r in rows]),
            np.array([traj(r) for r in rows])]
    if extended:
        E = np.array([np.asarray(r["layer_entropies"], dtype=float) for r in rows])
        deltas = np.diff(E, axis=1)                      # adjacent-layer change
        collapse = E.argmin(1, keepdims=True) / n_native  # where entropy bottoms
        peak = E.argmax(1, keepdims=True) / n_native
        cols += [deltas, collapse, peak,
                 E.min(1, keepdims=True), E.max(1, keepdims=True),
                 np.abs(deltas).sum(1, keepdims=True)]   # total churn
    X = np.hstack(cols)
    y = np.array([0 if r["correct"] else 1 for r in rows])
    return X, y


def make_tabfm():
    import torch
    from safetensors.torch import load_file
    from tabfm import TabFMClassifier
    from tabfm.src.pytorch.model import TabFM
    from tabfm.src.pytorch.tabfm_v1_0_0 import ClassificationConfig

    model = TabFM(**ClassificationConfig().to_dict())
    miss, unexp = model.load_state_dict(load_file(WEIGHTS), strict=False)
    assert not miss and not unexp
    device = "cuda" if torch.cuda.is_available() else "cpu"
    try:
        if device == "cuda" and torch.cuda.mem_get_info()[0] / 1e9 < 3.0:
            device = "cpu"
    except Exception:  # noqa: BLE001
        device = "cpu"
    model = model.to(device).eval()

    def factory(n_est):
        return TabFMClassifier(model=model, n_estimators=n_est, random_state=0)
    return factory


def main() -> None:
    from sklearn.metrics import roc_auc_score
    from sklearn.model_selection import StratifiedKFold
    from sklearn.preprocessing import StandardScaler

    factory = make_tabfm()
    raw = {m: [json.loads(l) for l in open(TRACE.format(m), encoding="utf-8") if l.strip()]
           for m in MODELS}

    grid = list(itertools.product([8, 32, 64], ["native", "resampled"],
                                  [False, True], [False, True]))
    # (n_est, traj, zscore, extended)
    results = {}
    short = {"gemma-4-e4b-it": "e4b", "gemma-4-12b-it": "12b",
             "huihui-gemma-4-12b-it-abliterated": "ablit",
             "gemma-4-26b-a4b-it": "moe", "qwen3.6-27b": "qwen"}
    print(f"{'n_est':>5} {'traj':>10} {'zscore':>6} {'ext':>5} " +
          " ".join(f"{short[m]:>7}" for m in MODELS) + "   mean")
    for n_est, traj_mode, zscore, extended in grid:
        aucs = []
        for m in MODELS:
            X, y = build_X(raw[m], traj_mode, extended)
            oof = np.zeros(len(y))
            for tr, te in StratifiedKFold(5, shuffle=True, random_state=0).split(X, y):
                Xtr, Xte = X[tr], X[te]
                if zscore:
                    sc = StandardScaler().fit(Xtr)
                    Xtr, Xte = sc.transform(Xtr), sc.transform(Xte)
                clf = factory(n_est)
                clf.fit(Xtr, y[tr])
                oof[te] = clf.predict_proba(Xte)[:, 1]
            aucs.append(roc_auc_score(y, oof))
        key = f"{n_est}|{traj_mode}|z{int(zscore)}|ext{int(extended)}"
        results[key] = {"per_model": dict(zip(MODELS, aucs)),
                        "mean": float(np.mean(aucs))}
        print(f"{n_est:>5} {traj_mode:>10} {str(zscore):>6} {str(extended):>5} " +
              " ".join(f"{a:>7.3f}" for a in aucs) + f"  {np.mean(aucs):>6.3f}")

    best = max(results, key=lambda k: results[k]["mean"])
    print(f"\nBEST (by mean AUC): {best}  mean={results[best]['mean']:.3f}")
    print("FREEZE this config before the 31B trace lands.")
    with open("out/tabfm_sweep.json", "w", encoding="utf-8") as f:
        json.dump({"results": results, "best": best}, f, indent=1)
    print("wrote out/tabfm_sweep.json")


if __name__ == "__main__":
    main()
