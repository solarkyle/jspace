"""Figures for the writeup: emotional vividness by model + hallucination separation."""
import json
import numpy as np
import matplotlib.pyplot as plt

data = {r["model"]: r for r in json.load(open("out/emotion_all5.json", encoding="utf-8"))}
SHORT = {
    "google/gemma-4-E4B-it": "Gemma4-E4B\n(4B dense)",
    "google/gemma-4-12B-it": "Gemma4-12B\n(dense)",
    "huihui-ai/Huihui-gemma-4-12B-it-abliterated": "Gemma4-12B\nabliterated",
    "google/gemma-4-26B-A4B-it": "Gemma4-26B\n(MoE)",
    "Qwen/Qwen3.6-27B": "Qwen3.6-27B\n(dense)",
}
order = list(SHORT)
EMOS = ["fury", "terror", "grief", "euphoria", "amusement"]

plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 11})
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5.2))

# --- Panel 1: emotional vividness (inverted so taller = more vivid) ---
vivid = []
for m in order:
    ranks = [min(rk for _, rk in data[m]["evidence"][e]) for e in EMOS]
    vivid.append(np.mean([np.log10(r + 1) for r in ranks]))
inv = [3 - v for v in vivid]  # taller = more vivid
colors = ["#8891a6", "#c25a5a", "#5aa0c2", "#7bc27b", "#d4a24e"]
bars = ax1.bar(range(len(order)), inv, color=colors, edgecolor="black", linewidth=0.6)
ax1.set_xticks(range(len(order)))
ax1.set_xticklabels([SHORT[m] for m in order], fontsize=9)
ax1.set_ylabel("emotional vividness\n(higher = emotions nearer top of vocabulary)")
ax1.set_title("How vividly does each model hold a covert emotion?", fontsize=12, weight="bold")
for i, m in enumerate(order):
    ranks = sorted(min(rk for _, rk in data[m]["evidence"][e]) for e in EMOS)
    ax1.text(i, inv[i] + 0.03, f"best rank\n#{ranks[0]}", ha="center", fontsize=8)
ax1.set_ylim(0, 3.3)
ax1.spines[["top", "right"]].set_visible(False)

# --- Panel 2: hallucination separation among high-confidence answers ---
rows = [json.loads(l) for l in open("out/uncertainty_v2.jsonl", encoding="utf-8")]
y = np.array([r["correct"] for r in rows])
ent = np.array([r["mean_entropy"] for r in rows])
lp = np.array([r["bl_first_token_logprob"] for r in rows])
hc = lp > np.median(lp)  # high output confidence
groups = [ent[hc & y], ent[hc & ~y], ent[~hc & y], ent[~hc & ~y]]
labels = ["high-conf\n+ correct", "high-conf\n+ WRONG", "low-conf\n+ correct", "low-conf\n+ wrong"]
gc = ["#7bc27b", "#c25a5a", "#a9d3a9", "#e0a0a0"]
parts = ax2.violinplot(groups, showmeans=True)
for pc, c in zip(parts["bodies"], gc):
    pc.set_facecolor(c); pc.set_alpha(0.8)
ax2.set_xticks(range(1, 5)); ax2.set_xticklabels(labels, fontsize=9)
ax2.set_ylabel("workspace entropy (higher = model less sure inside)")
ax2.set_title("Workspace catches overconfident hallucinations", fontsize=12, weight="bold")
ax2.annotate("even when output looks\nconfident, wrong answers\nhave a noisier workspace",
             xy=(2, groups[1].mean()), xytext=(2.6, groups[1].mean() + 1.2),
             fontsize=8.5, arrowprops=dict(arrowstyle="->", color="#c25a5a"))
ax2.spines[["top", "right"]].set_visible(False)

plt.tight_layout()
plt.savefig("assets/figures.png", dpi=130, bbox_inches="tight")
print("wrote assets/figures.png")
