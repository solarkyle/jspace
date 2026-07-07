"""Figure 3: what making-it-up looks like from the inside.

Top left:   every high-output-confidence answer from Gemma E4B as one line -
            per-layer workspace entropy across the band. Green = correct,
            red = wrong. The two populations visibly separate.
Top right:  mean trajectories for the other four models (small multiples).
Bottom:     two real examples at maximum output confidence - the actual top
            workspace tokens per layer, right vs wrong, receipts included.

Inputs: out/uncertainty_trivia_<slug>.jsonl (cloud runs, layer_entropies)
        out/qa_dump.json (per-layer top tokens for picked examples)
"""
import glob
import json
import os

import numpy as np
import matplotlib.pyplot as plt

NAME = {"gemma-4-e4b-it": "Gemma E4B", "gemma-4-12b-it": "Gemma 12B",
        "huihui-gemma-4-12b-it-abliterated": "12B abliterated",
        "gemma-4-26b-a4b-it": "Gemma 26B MoE", "qwen3.6-27b": "Qwen 3.6 27B"}
ORDER = list(NAME)

def load(slug):
    p = f"out/uncertainty_trivia_{slug}.jsonl"
    if not os.path.exists(p):
        return None
    return [json.loads(l) for l in open(p, encoding="utf-8")]

plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10})
fig = plt.figure(figsize=(14, 11), facecolor="white")
gs = fig.add_gridspec(2, 2, height_ratios=[1, 1.15], hspace=0.3, wspace=0.18)

GREEN, RED = "#2e9e5b", "#cf4a42"

# ---------- top left: the stack (E4B, high-confidence only) ----------
ax = fig.add_subplot(gs[0, 0])
rows = load("gemma-4-e4b-it")
lp = np.array([r["bl_first_token_logprob"] for r in rows])
hc = lp > np.median(lp)
E = np.array([r["layer_entropies"] for r in rows])
y = np.array([r["correct"] for r in rows])
x = np.linspace(0.25, 0.75, E.shape[1])
for i in np.where(hc)[0]:
    ax.plot(x, E[i], color=GREEN if y[i] else RED, alpha=0.10, lw=0.7,
            zorder=2 if y[i] else 3)
for val, c, lab in [(1, GREEN, "correct"), (0, RED, "WRONG")]:
    m = E[hc & (y == val)].mean(0)
    ax.plot(x, m, color=c, lw=3.2, zorder=5,
            label=f"{lab} (n={int((hc & (y==val)).sum())})")
ax.set_xlabel("depth through the network (fraction of layers)")
ax.set_ylabel("workspace entropy at the answer position")
ax.set_title("Gemma E4B - every answer the model was CONFIDENT about\n"
             "(one line per question; the fog is visible before it speaks)",
             fontsize=11, weight="bold")
ax.legend(frameon=False, loc="upper right")
ax.spines[["top", "right"]].set_visible(False)

# ---------- top right: small multiples, mean +- IQR ----------
sub = fig.add_subplot(gs[0, 1]); sub.axis("off")
sub.set_title("same picture on every model (mean ± IQR, confident answers only)",
              fontsize=11, weight="bold")
others = [s for s in ORDER if s != "gemma-4-e4b-it" and load(s)]
for k, slug in enumerate(others[:4]):
    r2 = load(slug)
    lp2 = np.array([r["bl_first_token_logprob"] for r in r2])
    hc2 = lp2 > np.median(lp2)
    E2 = np.array([r["layer_entropies"] for r in r2])
    y2 = np.array([r["correct"] for r in r2])
    x2 = np.linspace(0.25, 0.75, E2.shape[1])
    axk = sub.inset_axes([0.06 + 0.5 * (k % 2), 0.02 + 0.48 * (1 - k // 2), 0.40, 0.38])
    for val, c in [(1, GREEN), (0, RED)]:
        m = hc2 & (y2 == val)
        if m.sum() < 5:
            continue
        med = np.median(E2[m], 0)
        q1, q3 = np.percentile(E2[m], [25, 75], 0)
        axk.fill_between(x2, q1, q3, color=c, alpha=0.18, lw=0)
        axk.plot(x2, med, color=c, lw=2.2)
    axk.set_title(NAME[slug], fontsize=9)
    axk.tick_params(labelsize=7)
    axk.spines[["top", "right"]].set_visible(False)

# ---------- bottom: token ladders for two real max-confidence examples ----------
if os.path.exists("out/qa_dump.json"):
    qa = json.load(open("out/qa_dump.json", encoding="utf-8"))
    # pick the crispest right + wrong
    ex_r = next(e for e in qa if e["correct"])
    ex_w = next(e for e in qa if not e["correct"]
                and "Downtown" in e["q"]) if any("Downtown" in e["q"] for e in qa if not e["correct"]) \
        else next(e for e in qa if not e["correct"])
    for col, (ex, c, verdict) in enumerate([(ex_r, GREEN, "CORRECT"),
                                            (ex_w, RED, "WRONG")]):
        axb = fig.add_subplot(gs[1, col])
        layers = ex["layers"]
        ents = [l["entropy"] for l in layers]
        lo, hi = min(ents), max(ents)
        axb.set_xlim(0, 1); axb.set_ylim(-1.5, len(layers))
        axb.axis("off")
        q = ex["q"] if len(ex["q"]) < 70 else ex["q"][:67] + "..."
        axb.set_title(f"“{q}”\nmodel says: “{ex['model_answer']}” - {verdict} "
                      f"(output confidence: maximum)",
                      fontsize=10.5, weight="bold", color=c)
        for i, l in enumerate(reversed(layers)):     # deep layers on top
            yy = len(layers) - 1 - i
            heat = (l["entropy"] - lo) / (hi - lo + 1e-9)
            axb.add_patch(plt.Rectangle((0.13, yy - 0.42), 0.85, 0.84,
                          color=plt.cm.RdYlGn_r(0.15 + 0.7 * heat), alpha=0.55, lw=0))
            toks = [t.strip() or "·" for t in l["top"][:5]]
            # bold the token if it's the model's eventual answer token
            ans = ex["first_token"].strip()
            txt = "  ".join(f"[{t}]" if t == ans else t for t in toks)
            axb.text(0.14, yy, txt[:64], fontsize=8.3, va="center",
                     family="monospace")
            axb.text(0.115, yy, f"L{l['layer']}", fontsize=7, va="center",
                     ha="right", color="#666")
            axb.text(0.995, yy, f"{l['entropy']:.1f}", fontsize=7, va="center",
                     ha="right", color="#666")
        axb.text(0.14, -1.0, "top workspace tokens per layer (deep layers at top); "
                 "[answer] = the token it ends up saying;\nrow color = entropy "
                 "(green calm → red fog); right column = entropy value",
                 fontsize=8, color="#555")

plt.savefig("assets/figure3_confidently_wrong.png", dpi=130, bbox_inches="tight")
print("wrote assets/figure3_confidently_wrong.png")
