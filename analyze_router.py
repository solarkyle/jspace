"""The classifier proof layer: do workspace features beat/boost output confidence?

Protocol (pre-committed):
  - Features from existing traces only (no new GPU): output-confidence baselines
    + workspace stats derived from the per-layer entropy trajectory.
  - Logistic regression (plain numpy GD), predicting WRONG.
  - Within-model 5-fold CV: logprob-only vs workspace-only vs combined.
  - Transfer: train on E4B only, test zero-shot on the other 4 models
    (features z-scored per model; that is the whole transfer trick).
  - Report ROC-AUC + wrong-answer catch rate at 20/30/50% escalation budgets.
  - Confident-only variant for the blind-spot claim.
  - Chart: fog-tercile vs accuracy among confident answers (the layman chart).

Run: python analyze_router.py   (writes assets/figure4_router.png too)
"""
import json
import os
import sys

import numpy as np
import matplotlib.pyplot as plt

sys.stdout.reconfigure(encoding="utf-8", errors="replace")

ORDER = ["gemma-4-e4b-it", "gemma-4-12b-it", "huihui-gemma-4-12b-it-abliterated",
         "gemma-4-26b-a4b-it", "qwen3.6-27b"]
NAME = {"gemma-4-e4b-it": "E4B", "gemma-4-12b-it": "12B",
        "huihui-gemma-4-12b-it-abliterated": "12B-ablit",
        "gemma-4-26b-a4b-it": "26B-MoE", "qwen3.6-27b": "Qwen-27B"}

BASE = ["bl_first_token_logprob", "bl_mean_logprob", "bl_min_logprob", "bl_answer_len"]

def featurize(rows):
    """Workspace features from the layer-entropy trajectory + saved stats."""
    feats = []
    for r in rows:
        e = np.array(r["layer_entropies"])
        n = len(e)
        x = np.arange(n)
        slope = np.polyfit(x, e, 1)[0]
        late = e[2 * n // 3:].mean()
        feats.append({
            "ws_mean_entropy": e.mean(),
            "ws_max_entropy": e.max(),
            "ws_late_entropy": late,
            "ws_entropy_slope": slope,
            "ws_entropy_std": e.std(),
            "ws_ignition_frac": r["ignition_frac"],
            "ws_ignition_depth": r["ignition_depth"],
            "ws_mean_log_rank": r["mean_log_rank_answer"],
            "ws_band_agreement": r["band_agreement"],
            "ws_hedge_rank": r["best_hedge_rank_log"],
        })
    return feats

def load(slug):
    rows = [json.loads(l) for l in open(f"data/uncertainty_trivia_{slug}.jsonl",
                                        encoding="utf-8")]
    ws = featurize(rows)
    WS = list(ws[0].keys())
    X = {**{k: np.array([r[k] for r in rows]) for k in BASE},
         **{k: np.array([w[k] for w in ws]) for k in WS}}
    y = np.array([r["correct"] for r in rows]).astype(float)
    return X, y, WS

def zmat(X, keys):
    M = np.column_stack([X[k] for k in keys])
    return (M - M.mean(0)) / (M.std(0) + 1e-9)

def logit_fit(X, y, iters=3000, lr=0.1, l2=1e-3):
    w = np.zeros(X.shape[1]); b = 0.0
    for _ in range(iters):
        p = 1 / (1 + np.exp(-(X @ w + b)))
        g = p - y
        w -= lr * (X.T @ g / len(y) + l2 * w)
        b -= lr * g.mean()
    return w, b

def auc(score, label):
    o = np.argsort(score); r = np.empty(len(score)); r[o] = np.arange(1, len(score) + 1)
    n1 = label.sum(); n0 = len(label) - n1
    if n1 == 0 or n0 == 0:
        return float("nan")
    return (r[label == 1].sum() - n1 * (n1 + 1) / 2) / (n0 * n1)

def cv_auc(X, y, k=5, seed=0):
    """y here = wrong(1)/right(0). Returns mean test AUC + pooled OOF scores."""
    rng = np.random.default_rng(seed)
    idx = rng.permutation(len(y))
    folds = np.array_split(idx, k)
    oof = np.zeros(len(y))
    aucs = []
    for i in range(k):
        te = folds[i]; tr = np.concatenate([folds[j] for j in range(k) if j != i])
        w, b = logit_fit(X[tr], y[tr])
        s = X[te] @ w + b
        oof[te] = s
        aucs.append(auc(s, y[te].astype(bool)))
    return float(np.mean(aucs)), oof

def catch_at(score, wrong, budget):
    k = int(len(score) * budget)
    idx = np.argsort(-score)[:k]
    return wrong[idx].sum() / max(1, wrong.sum())

SETS = None
print("=" * 84)
print("WITHIN-MODEL 5-FOLD CV (predicting WRONG; logistic regression)")
print("=" * 84)
print(f"{'model':<10} {'lp-only':>8} {'workspace':>10} {'combined':>9} | catch@20/30/50% (combined vs lp-only)")
oof_store = {}
for slug in ORDER:
    X, y, WS = load(slug)
    if SETS is None:
        SETS = {"lp-only": BASE, "workspace": WS, "combined": BASE + WS}
    wrong = (1 - y)
    row = {}
    for name, keys in SETS.items():
        a, oof = cv_auc(zmat(X, keys), wrong)
        row[name] = (a, oof)
    oof_store[slug] = (row, X, y)
    c_comb = [catch_at(row["combined"][1], wrong.astype(bool), b) for b in (.2, .3, .5)]
    c_lp = [catch_at(row["lp-only"][1], wrong.astype(bool), b) for b in (.2, .3, .5)]
    print(f"{NAME[slug]:<10} {row['lp-only'][0]:>8.3f} {row['workspace'][0]:>10.3f} "
          f"{row['combined'][0]:>9.3f} | "
          + " ".join(f"{c:.0%}" for c in c_comb) + "  vs  "
          + " ".join(f"{c:.0%}" for c in c_lp))

print()
print("=" * 84)
print("ZERO-SHOT TRANSFER: train on E4B only, test on the others (per-model z-scored)")
print("=" * 84)
Xe, ye, WS = load("gemma-4-e4b-it")
we = {}
for name, keys in SETS.items():
    we[name] = logit_fit(zmat(Xe, keys), 1 - ye)
print(f"{'model':<10} {'lp-only':>8} {'workspace':>10} {'combined':>9} | combined catch@30%")
for slug in ORDER[1:]:
    X, y, _ = load(slug)
    wrong = (1 - y).astype(bool)
    out = {}
    for name, keys in SETS.items():
        w, b = we[name]
        s = zmat(X, keys) @ w + b
        out[name] = (auc(s, wrong), s)
    print(f"{NAME[slug]:<10} {out['lp-only'][0]:>8.3f} {out['workspace'][0]:>10.3f} "
          f"{out['combined'][0]:>9.3f} | {catch_at(out['combined'][1], wrong, .3):.0%}")

print()
print("=" * 84)
print("CONFIDENT-ONLY SUBSET (top-half output logprob): the blind-spot test, CV")
print("=" * 84)
print(f"{'model':<10} {'lp-only':>8} {'workspace':>10} {'combined':>9}   n_wrong")
for slug in ORDER:
    X, y, _ = load(slug)
    hc = X["bl_first_token_logprob"] > np.median(X["bl_first_token_logprob"])
    Xs = {k: v[hc] for k, v in X.items()}
    ys = y[hc]
    res = {}
    for name, keys in SETS.items():
        a, _ = cv_auc(zmat(Xs, keys), 1 - ys)
        res[name] = a
    print(f"{NAME[slug]:<10} {res['lp-only']:>8.3f} {res['workspace']:>10.3f} "
          f"{res['combined']:>9.3f}   {int((1-ys).sum())}")

# ---------------- figure 4 ----------------
plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10})
fig, axes = plt.subplots(1, 3, figsize=(15, 4.6))
GREEN, RED, GOLD, BLUE = "#2e9e5b", "#cf4a42", "#b8860b", "#3f7fbf"

# (a) fog terciles vs accuracy among confident answers, all models as grouped bars
ax = axes[0]
width = 0.15
for mi, slug in enumerate(ORDER):
    X, y, _ = load(slug)
    hc = X["bl_first_token_logprob"] > np.median(X["bl_first_token_logprob"])
    ent = X["ws_mean_entropy"][hc]; yy = y[hc]
    t1, t2 = np.percentile(ent, [33, 67])
    accs = [yy[ent <= t1].mean(), yy[(ent > t1) & (ent <= t2)].mean(), yy[ent > t2].mean()]
    ax.bar(np.arange(3) + (mi - 2) * width, accs, width * 0.92,
           label=NAME[slug],
           color=["#8891a6", "#c25a5a", "#b07ac2", "#7bc27b", "#d4a24e"][mi],
           edgecolor="black", lw=0.4)
ax.set_xticks(range(3)); ax.set_xticklabels(["low fog", "medium fog", "high fog"])
ax.set_ylabel("accuracy among CONFIDENT answers")
ax.set_title("As internal fog rises, confident answers\nget less trustworthy", weight="bold", fontsize=11)
ax.legend(fontsize=7.5, frameon=False, ncol=2)
ax.spines[["top", "right"]].set_visible(False)

# (b) ROC curves, E4B, confident subset
ax = axes[1]
X, y, _ = load("gemma-4-e4b-it")
hc = X["bl_first_token_logprob"] > np.median(X["bl_first_token_logprob"])
Xs = {k: v[hc] for k, v in X.items()}; wrong = (1 - y[hc])
for name, keys, c in [("logprob only", BASE, "#8891a6"),
                      ("workspace only", SETS["workspace"], BLUE),
                      ("combined", SETS["combined"], GOLD)]:
    a, oof = cv_auc(zmat(Xs, keys), wrong)
    order = np.argsort(-oof)
    tpr = np.cumsum(wrong[order]) / wrong.sum()
    fpr = np.cumsum(1 - wrong[order]) / (1 - wrong).sum()
    ax.plot(fpr, tpr, color=c, lw=2.2, label=f"{name} (AUC {a:.2f})")
ax.plot([0, 1], [0, 1], "--", color="#555", lw=1)
ax.set_xlabel("false positive rate"); ax.set_ylabel("wrong answers caught")
ax.set_title("Catching overconfident wrong answers\n(E4B, confident subset, 5-fold OOF)", weight="bold", fontsize=11)
ax.legend(fontsize=8.5, frameon=False, loc="lower right")
ax.spines[["top", "right"]].set_visible(False)

# (c) escalation curves, E4B all answers
ax = axes[2]
X, y, _ = load("gemma-4-e4b-it")
wrong = (1 - y).astype(bool)
budgets = np.linspace(0.02, 0.7, 35)
for name, keys, c in [("logprob only", BASE, "#8891a6"),
                      ("workspace only", SETS["workspace"], BLUE),
                      ("combined", SETS["combined"], GOLD)]:
    _, oof = cv_auc(zmat(X, keys), 1 - y)
    ax.plot(budgets, [catch_at(oof, wrong, b) for b in budgets], color=c, lw=2.2, label=name)
ax.plot(budgets, budgets, "--", color="#555", lw=1, label="random routing")
ax.set_xlabel("fraction of queries escalated to the big model")
ax.set_ylabel("fraction of wrong answers caught")
ax.set_title("The router curve (E4B, out-of-fold)", weight="bold", fontsize=11)
ax.legend(fontsize=8.5, frameon=False)
ax.spines[["top", "right"]].set_visible(False)

plt.tight_layout()
os.makedirs("assets", exist_ok=True)
plt.savefig("assets/figure4_router.png", dpi=130, bbox_inches="tight")
print("\nwrote assets/figure4_router.png")
