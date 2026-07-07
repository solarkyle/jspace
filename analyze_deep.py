"""Deeper cross-model analysis of the emotion matrix + hallucination data.

Sections:
  1. Emotion geometry: does cross-emotion bleed match the human affect circumplex?
  2. Specificity: diagonal vs off-diagonal delta (right emotion vs just loud)
  3. Matched-active-params: E4B dense vs 26B-A4B MoE (both ~4B active)
  4. Abliteration per-emotion unlock
  5. Hallucination: confounds, per-feature AUC, escalation-router simulation
"""
import json
import sys
import numpy as np

sys.stdout.reconfigure(encoding="utf-8", errors="replace")

EMOS = ["fury", "terror", "grief", "euphoria", "amusement"]
SHORT = {
    "google/gemma-4-E4B-it": "E4B-dense",
    "google/gemma-4-12B-it": "12B-dense",
    "huihui-ai/Huihui-gemma-4-12B-it-abliterated": "12B-ablit",
    "google/gemma-4-26B-A4B-it": "26B-MoE",
    "Qwen/Qwen3.6-27B": "Qwen-27B",
}
data = {SHORT[r["model"]]: r for r in json.load(open("out/emotion_all5.json", encoding="utf-8"))}
order = list(SHORT.values())

# Human affect theory prediction (Russell circumplex):
#   valence: fury -, terror -, grief -, euphoria +, amusement +
#   arousal: fury +, terror +, grief -, euphoria +, amusement +/mid
VAL = {"fury": -1, "terror": -1, "grief": -1, "euphoria": 1, "amusement": 1}
ARO = {"fury": 1, "terror": 1, "grief": -1, "euphoria": 1, "amusement": 0.5}
def circumplex_sim(a, b):
    return VAL[a] * VAL[b] + ARO[a] * ARO[b]  # crude similarity

print("=" * 72)
print("1. EMOTION GEOMETRY: does off-diagonal bleed match the human circumplex?")
print("=" * 72)
# delta_matrix[cond][lex] = how much condition `cond` boosts lexicon `lex` vs neutral
# (positive = boosted). Off-diagonal similarity between conditions in delta-space.
for m in order:
    dm = data[m]["delta_matrix"]
    # matrix of deltas: rows=condition, cols=lexicon
    M = np.array([[dm[c][l] for l in EMOS] for c in EMOS])
    # correlate model's cross-emotion structure with circumplex prediction
    pred, obs = [], []
    for i, a in enumerate(EMOS):
        for j, b in enumerate(EMOS):
            if i == j:
                continue
            pred.append(circumplex_sim(a, b))
            obs.append((M[i, j] + M[j, i]) / 2)  # symmetrized bleed
    r = np.corrcoef(pred, obs)[0, 1]
    print(f"  {m:<10} circumplex correlation r = {r:+.3f}")
print()
print("  Full symmetrized bleed matrices (cond boosts lexicon, + = louder):")
for m in order:
    dm = data[m]["delta_matrix"]
    M = np.array([[dm[c][l] for l in EMOS] for c in EMOS])
    S = (M + M.T) / 2
    print(f"\n  {m} (rows/cols: {', '.join(e[:4] for e in EMOS)})")
    for i, e in enumerate(EMOS):
        print("   ", e[:4], " ".join(f"{S[i,j]:+.2f}" for j in range(5)))

print()
print("=" * 72)
print("2. SPECIFICITY: right-emotion boost minus wrong-emotion boost")
print("=" * 72)
for m in order:
    dm = data[m]["delta_matrix"]
    diag = np.mean([dm[e][e] for e in EMOS])
    off = np.mean([dm[c][l] for c in EMOS for l in EMOS if c != l])
    print(f"  {m:<10} diag {diag:+.2f}  off-diag {off:+.2f}  specificity {diag - off:+.2f}")

print()
print("=" * 72)
print("3. MATCHED ACTIVE PARAMS: E4B dense vs 26B-A4B MoE (~4B active each)")
print("=" * 72)
for m in ["E4B-dense", "26B-MoE"]:
    ranks = {e: min(rk for _, rk in data[m]["evidence"][e]) for e in EMOS}
    vivid = np.mean([np.log10(r + 1) for r in ranks.values()])
    print(f"  {m:<10} best ranks: " + "  ".join(f"{e[:4]}#{ranks[e]}" for e in EMOS)
          + f"   vividness(log10) {vivid:.2f}")

print()
print("=" * 72)
print("4. ABLITERATION: per-emotion unlock (12B base -> abliterated)")
print("=" * 72)
for e in EMOS:
    b = min(rk for _, rk in data["12B-dense"]["evidence"][e])
    a = min(rk for _, rk in data["12B-ablit"]["evidence"][e])
    tok_b = min(data["12B-dense"]["evidence"][e], key=lambda x: x[1])[0]
    tok_a = min(data["12B-ablit"]["evidence"][e], key=lambda x: x[1])[0]
    print(f"  {e:<10} base #{b:<6} ({tok_b})  ->  ablit #{a:<6} ({tok_a})   "
          f"unlock {np.log10((b+1)/(a+1)):+.2f} orders")

# ---------------- hallucination ----------------
rows = [json.loads(l) for l in open("out/uncertainty_v2.jsonl", encoding="utf-8")]
y = np.array([r["correct"] for r in rows]).astype(float)
feats = {k: np.array([r[k] for r in rows]) for k in rows[0] if k not in ("q", "answer", "correct")}

def auc(score, label):
    """AUC of score predicting label=1, rank-based."""
    order_ = np.argsort(score)
    r = np.empty(len(score)); r[order_] = np.arange(1, len(score) + 1)
    n1 = label.sum(); n0 = len(label) - n1
    return (r[label == 1].sum() - n1 * (n1 + 1) / 2) / (n0 * n1)

print()
print("=" * 72)
print("5a. HALLUCINATION: per-feature AUC (oriented to predict CORRECT)")
print("=" * 72)
for k, v in feats.items():
    a = auc(v, y)
    a = max(a, 1 - a)
    print(f"  {k:<28} AUC {a:.3f}")

print()
print("5b. CONFOUND CHECK: is mean_entropy just answer length / question type?")
for k in ["bl_answer_len", "bl_first_token_logprob", "bl_mean_logprob"]:
    r = np.corrcoef(feats["mean_entropy"], feats[k])[0, 1]
    print(f"  corr(mean_entropy, {k}) = {r:+.3f}")
# partial: does entropy predict correctness within answer-length terciles?
alen = feats["bl_answer_len"]
for lo, hi, name in [(0, np.percentile(alen, 33), "short"),
                     (np.percentile(alen, 33), np.percentile(alen, 67), "mid"),
                     (np.percentile(alen, 67), 1e9, "long")]:
    m = (alen >= lo) & (alen < hi)
    if m.sum() > 20 and 0 < y[m].sum() < m.sum():
        a = auc(feats["mean_entropy"][m], y[m]); a = max(a, 1 - a)
        print(f"  entropy AUC within {name:<5} answers (n={m.sum()}): {a:.3f}")

print()
print("5c. OVERCONFIDENT-HALLUCINATION subset, with real stats")
lp = feats["bl_first_token_logprob"]
hc = lp > np.median(lp)
ent = feats["mean_entropy"]
g_right, g_wrong = ent[hc & (y == 1)], ent[hc & (y == 0)]
print(f"  high-conf & correct: n={len(g_right)}, entropy {g_right.mean():.3f}")
print(f"  high-conf & WRONG:   n={len(g_wrong)}, entropy {g_wrong.mean():.3f}")
# Mann-Whitney U via rank AUC + normal approx p-value
sub_y = y[hc]; sub_s = ent[hc]
a = auc(sub_s, 1 - sub_y)  # entropy predicting WRONG
n1, n0 = int((1 - sub_y).sum()), int(sub_y.sum())
u = a * n0 * n1
mu, sd = n0 * n1 / 2, np.sqrt(n0 * n1 * (n0 + n1 + 1) / 12)
z = (u - mu) / sd
print(f"  entropy->wrong AUC among high-conf: {a:.3f}  (z={z:.2f})")
# effect size
pooled = np.sqrt((g_right.var() + g_wrong.var()) / 2)
print(f"  Cohen's d = {(g_wrong.mean() - g_right.mean()) / pooled:.2f}")

print()
print("5d. ESCALATION ROUTER SIMULATION")
print("  Policy: answer locally; escalate the X% of queries the router distrusts.")
print("  Assume big model answers escalated queries correctly (upper bound) or")
print("  at 90% (realistic). Base local accuracy:", f"{y.mean():.3f}")

def router_curve(score_distrust, budgets=(0.1, 0.2, 0.3, 0.5)):
    out = []
    for b in budgets:
        k = int(len(y) * b)
        idx = np.argsort(-score_distrust)[:k]  # most distrusted
        esc = np.zeros(len(y), bool); esc[idx] = True
        caught = (y[esc] == 0).sum()          # wrong answers caught
        total_wrong = (y == 0).sum()
        acc_ub = (y[~esc].sum() + esc.sum()) / len(y)
        acc_90 = (y[~esc].sum() + 0.9 * esc.sum()) / len(y)
        out.append((b, caught / total_wrong, acc_ub, acc_90))
    return out

# distrust scores
z_ent = (ent - ent.mean()) / ent.std()
z_lp = (lp - lp.mean()) / lp.std()
scores = {
    "output logprob only": -z_lp,
    "workspace entropy only": z_ent,
    "combined (entropy - logprob)": z_ent - z_lp,
}
print(f"  {'router':<30} {'budget':>6} {'%wrong caught':>14} {'acc(UB)':>8} {'acc(90%)':>9}")
for name, s in scores.items():
    for b, frac, ub, a90 in router_curve(s):
        print(f"  {name:<30} {b:>5.0%} {frac:>13.1%} {ub:>8.3f} {a90:>9.3f}")
    print()

print("5e. FAIR HEAD-TO-HEAD inside each signal's blind spot")
def sub_auc(score, label):
    a = auc(score, label)
    return max(a, 1 - a)
hcm = lp > np.median(lp)
print(f"  among HIGH-output-confidence answers (n={hcm.sum()}, {int((1-y[hcm]).sum())} wrong):")
print(f"    workspace entropy -> wrong        AUC {sub_auc(ent[hcm], 1-y[hcm]):.3f}")
print(f"    residual logprob  -> wrong        AUC {sub_auc(-lp[hcm], 1-y[hcm]):.3f}")
lcm = ent < np.median(ent)
print(f"  among LOW-workspace-entropy answers (n={lcm.sum()}, {int((1-y[lcm]).sum())} wrong):")
print(f"    output logprob    -> wrong        AUC {sub_auc(-lp[lcm], 1-y[lcm]):.3f}")
print(f"    residual entropy  -> wrong        AUC {sub_auc(ent[lcm], 1-y[lcm]):.3f}")

print()
print("5f. QUADRANT TABLE (median splits on both signals)")
qe = ent < np.median(ent)   # workspace clean
ql = lp > np.median(lp)     # output confident
for m, name in [(ql & qe, "both confident"), (ql & ~qe, "output conf, workspace noisy"),
                (~ql & qe, "output unsure, workspace clean"), (~ql & ~qe, "both unsure")]:
    print(f"  {name:<32} n={m.sum():>3}  accuracy {y[m].mean():.3f}")

print()
print("=" * 72)
print("6. CIRCUMPLEX PERMUTATION TEST (10k label shuffles)")
print("=" * 72)
rng = np.random.default_rng(0)
for m in order:
    dm = data[m]["delta_matrix"]
    M = np.array([[dm[c][l] for l in EMOS] for c in EMOS])
    def corr_for(perm):
        pred, obs = [], []
        for i in range(5):
            for j in range(5):
                if i == j:
                    continue
                a_, b_ = EMOS[perm[i]], EMOS[perm[j]]
                pred.append(VAL[a_] * VAL[b_] + ARO[a_] * ARO[b_])
                obs.append((M[i, j] + M[j, i]) / 2)
        return np.corrcoef(pred, obs)[0, 1]
    real = corr_for(list(range(5)))
    null = [corr_for(rng.permutation(5)) for _ in range(10000)]
    p = np.mean([n >= real for n in null])
    print(f"  {m:<10} r={real:+.3f}  perm-p={p:.3f}")
print("  (not significant anywhere at n=1 prompt/condition; suggestive ordering only)")
