"""PopQA: scoring the registered predictions in docs/POPQA_PREREG.md.

Rule set, fixed before any PopQA analysis ran:
  - Features and CV identical to analyze_tabfm_31b_blind.py: logprob = 4
    baselines; workspace = 6 scalars + native layer entropies; combined =
    both. StratifiedKFold(5, shuffle, seed 0), per-fold StandardScaler.
  - Models: TabFM frozen config (n_estimators=32) plus logistic reference.
  - P1 scored on mean TabFM combined AUC across the 7 models vs 0.86, and
    per-model comparison vs the TriviaQA reference values below.
  - P2 scored on TabFM logprob-only drops (5 of 7) and workspace-only
    stability within 0.05 (5 of 7) vs the same references.
  - P3 is scored separately after LLM grading (not in this script).

Run with the ufc_bet tabfm venv.
"""

import json
import os

import numpy as np

BASE = ["bl_first_token_logprob", "bl_mean_logprob", "bl_min_logprob", "bl_answer_len"]
WS = ["ignition_frac", "ignition_depth", "mean_log_rank_answer",
      "band_agreement", "mean_entropy", "best_hedge_rank_log"]

# TriviaQA TabFM reference values (frozen config), from committed analyses
REF = {
    "gemma-4-e4b-it":                     {"logprob": 0.772, "workspace": 0.816, "combined": 0.838},
    "gemma-4-12b-it":                     {"logprob": 0.779, "workspace": 0.827, "combined": 0.845},
    "huihui-gemma-4-12b-it-abliterated":  {"logprob": 0.775, "workspace": 0.842, "combined": 0.853},
    "gemma-4-26b-a4b-it":                 {"logprob": 0.741, "workspace": 0.758, "combined": 0.811},
    "qwen3.6-27b":                        {"logprob": 0.849, "workspace": 0.774, "combined": 0.849},
    "gemma-4-31b-it":                     {"logprob": 0.720, "workspace": 0.797, "combined": 0.819},
    "mistral-small-24b-instruct-2501":    {"logprob": 0.843, "workspace": 0.692, "combined": 0.850},
}


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


def cv(X, y, make_clf, kind):
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import roc_auc_score
    from sklearn.model_selection import StratifiedKFold
    from sklearn.preprocessing import StandardScaler
    oof = np.zeros(len(y))
    for tr, te in StratifiedKFold(5, shuffle=True, random_state=0).split(X, y):
        sc = StandardScaler().fit(X[tr])
        Xtr, Xte = sc.transform(X[tr]), sc.transform(X[te])
        if kind == "tabfm":
            clf = make_clf()
            clf.fit(Xtr, y[tr])
            oof[te] = clf.predict_proba(Xte)[:, 1]
        else:
            clf = LogisticRegression(max_iter=2000).fit(Xtr, y[tr])
            oof[te] = clf.predict_proba(Xte)[:, 1]
    return roc_auc_score(y, oof)


def main() -> None:
    make_clf = make_tabfm()
    results = {}
    print(f"{'model':>34} {'acc':>5} | {'lp':>6} {'ws':>6} {'comb':>6} | vs TriviaQA (lp/ws/comb deltas)")
    for slug in REF:
        path = f"data/uncertainty_popqa_{slug}.jsonl"
        if not os.path.exists(path):
            print(f"{slug:>34} (missing)")
            continue
        rows = [json.loads(l) for l in open(path, encoding="utf-8") if l.strip()]
        E = np.array([r["layer_entropies"] for r in rows])
        Xb = np.array([[r[k] for k in BASE] for r in rows])
        Xw = np.hstack([np.array([[r[k] for k in WS] for r in rows]), E])
        y = np.array([0 if r["correct"] else 1 for r in rows])
        acc = 1 - y.mean()
        res = {}
        for name, X in [("logprob", Xb), ("workspace", Xw),
                        ("combined", np.hstack([Xb, Xw]))]:
            res[name] = cv(X, y, make_clf, "tabfm")
        results[slug] = res
        d = {k: res[k] - REF[slug][k] for k in res}
        print(f"{slug:>34} {acc:>5.2f} | {res['logprob']:>6.3f} {res['workspace']:>6.3f} "
              f"{res['combined']:>6.3f} | {d['logprob']:+.3f}/{d['workspace']:+.3f}/{d['combined']:+.3f}")

    if len(results) < 4:
        print("\n(too few traces to score predictions)")
        return
    print("\n== scoring the registered predictions ==")
    comb = np.array([r["combined"] for r in results.values()])
    beat = sum(results[s]["combined"] > REF[s]["combined"] for s in results)
    print(f"P1 ceiling: mean combined {comb.mean():.3f} (predicted > 0.86); "
          f"{beat}/{len(results)} models beat their TriviaQA combined."
          f"  -> {'HIT' if comb.mean() > 0.86 and beat >= 4 else 'PARTIAL' if comb.mean() > 0.85 else 'MISS'}")
    lp_drop = sum(results[s]["logprob"] < REF[s]["logprob"] for s in results)
    ws_hold = sum(abs(results[s]["workspace"] - REF[s]["workspace"]) <= 0.05 for s in results)
    print(f"P2 miscalibration: logprob dropped on {lp_drop}/{len(results)} "
          f"(predicted >=5/7); workspace held within .05 on {ws_hold}/{len(results)} "
          f"(predicted >=5/7)."
          f"  -> {'HIT' if lp_drop >= 5 and ws_hold >= 5 else 'PARTIAL' if lp_drop >= 5 or ws_hold >= 5 else 'MISS'}")
    with open("out/popqa_results.json", "w", encoding="utf-8") as f:
        json.dump(results, f, indent=1)
    print("wrote out/popqa_results.json")


if __name__ == "__main__":
    main()
