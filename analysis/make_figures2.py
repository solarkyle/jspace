"""Figure 2: emotion structure emerging with capability + the quadrant result.

Top row:    5 bleed-matrix heatmaps (covert condition x emotion lexicon,
            delta log-rank vs neutral), shared color scale. The diagonal
            "lighting up" left to right IS the finding.
Bottom row: specificity bars + hallucination quadrant accuracy.
"""
import json
import numpy as np
import matplotlib.pyplot as plt

EMOS = ["fury", "terror", "grief", "euphoria", "amusement"]
SHORT = {
    "google/gemma-4-E4B-it": "Gemma E4B\n(4B dense)",
    "google/gemma-4-12B-it": "Gemma 12B\n(dense)",
    "huihui-ai/Huihui-gemma-4-12B-it-abliterated": "Gemma 12B\n(abliterated)",
    "google/gemma-4-26B-A4B-it": "Gemma 26B\n(MoE, 4B active)",
    "Qwen/Qwen3.6-27B": "Qwen 3.6 27B\n(dense)",
}
data = {r["model"]: r for r in json.load(open("out/emotion_all5.json", encoding="utf-8"))}
order = list(SHORT)

plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10})
fig = plt.figure(figsize=(14, 8.6))
gs = fig.add_gridspec(2, 10, height_ratios=[1.05, 1], hspace=0.42, wspace=0.9)

# ---------- top row: heatmaps ----------
VMAX = 8
axes = [fig.add_subplot(gs[0, i * 2:i * 2 + 2]) for i in range(5)]
for ax, m in zip(axes, order):
    dm = data[m]["delta_matrix"]
    M = np.array([[dm[c][l] for l in EMOS] for c in EMOS])
    im = ax.imshow(M, cmap="RdBu_r", vmin=-VMAX, vmax=VMAX)
    ax.set_title(SHORT[m], fontsize=9.5)
    ax.set_xticks(range(5)); ax.set_yticks(range(5))
    ax.set_xticklabels([e[:4] for e in EMOS], fontsize=7.5, rotation=45)
    ax.set_yticklabels([e[:4] for e in EMOS] if ax is axes[0] else [], fontsize=7.5)
    for i in range(5):  # outline the diagonal
        ax.add_patch(plt.Rectangle((i - .5, i - .5), 1, 1, fill=False,
                                   edgecolor="black", lw=1.4))
axes[0].set_ylabel("covert emotion (instruction)", fontsize=9)
axes[2].set_xlabel("emotion word list boosted in the workspace (delta log-rank vs neutral)",
                   fontsize=9, labelpad=8)
cb = fig.colorbar(im, ax=axes, fraction=0.012, pad=0.01)
cb.set_label("boost vs neutral\n(+ = louder)", fontsize=8)
fig.text(0.5, 0.965, "A covert emotion instruction lights up the RIGHT emotion words, "
         "and the diagonal sharpens with capability",
         ha="center", fontsize=12.5, weight="bold")

# ---------- bottom left: specificity ----------
ax1 = fig.add_subplot(gs[1, 0:4])
spec, names = [], []
for m in order:
    dm = data[m]["delta_matrix"]
    diag = np.mean([dm[e][e] for e in EMOS])
    off = np.mean([dm[c][l] for c in EMOS for l in EMOS if c != l])
    spec.append(diag - off)
    names.append(SHORT[m].replace("\n", " "))
colors = ["#8891a6", "#c25a5a", "#b07ac2", "#7bc27b", "#d4a24e"]
ax1.barh(range(5), spec, color=colors, edgecolor="black", lw=0.6)
ax1.set_yticks(range(5)); ax1.set_yticklabels(names, fontsize=8.5)
ax1.invert_yaxis()
ax1.axvline(0, color="black", lw=0.8)
ax1.set_xlim(-0.85, 4.5)
ax1.set_xlabel("emotional specificity\n(right-emotion boost minus wrong-emotion boost, log-rank units)")
ax1.set_title("Capable models boost the RIGHT emotion,\nnot just any emotion", fontsize=11, weight="bold")
for i, s in enumerate(spec):
    ax1.text(s + (0.08 if s >= 0 else -0.08), i, f"{s:+.2f}",
             va="center", ha="left" if s >= 0 else "right", fontsize=8.5)
ax1.spines[["top", "right"]].set_visible(False)

# ---------- bottom right: quadrant accuracy ----------
ax2 = fig.add_subplot(gs[1, 5:10])
rows = [json.loads(l) for l in open("out/uncertainty_v2.jsonl", encoding="utf-8")]
y = np.array([r["correct"] for r in rows]).astype(float)
ent = np.array([r["mean_entropy"] for r in rows])
lp = np.array([r["bl_first_token_logprob"] for r in rows])
qe = ent < np.median(ent)
ql = lp > np.median(lp)
cells = [
    ("output confident\nworkspace clean", y[ql & qe]),
    ("output confident\nworkspace NOISY", y[ql & ~qe]),
    ("output unsure\nworkspace clean", y[~ql & qe]),
    ("output unsure\nworkspace noisy", y[~ql & ~qe]),
]
acc = [c.mean() for _, c in cells]
ns = [len(c) for _, c in cells]
gc = ["#7bc27b", "#c25a5a", "#a9c8d3", "#8891a6"]
bars = ax2.bar(range(4), acc, color=gc, edgecolor="black", lw=0.6)
ax2.set_xticks(range(4))
ax2.set_xticklabels([c[0] for c in cells], fontsize=8.2)
ax2.set_ylabel("accuracy (TriviaQA, Gemma E4B)")
ax2.set_ylim(0, 0.9)
for i, (a, n) in enumerate(zip(acc, ns)):
    ax2.text(i, a + 0.02, f"{a:.0%}\n(n={n})", ha="center", fontsize=8.5)
ax2.annotate("sounds sure, isn't:\n75% drops to 42% when\nthe workspace flickers",
             xy=(1, acc[1]), xytext=(1.9, 0.66), fontsize=9,
             arrowprops=dict(arrowstyle="->", color="#c25a5a"))
ax2.set_title("The workspace flags overconfident wrong answers", fontsize=11, weight="bold")
ax2.spines[["top", "right"]].set_visible(False)

plt.savefig("assets/figures2.png", dpi=130, bbox_inches="tight")
print("wrote assets/figures2.png")
