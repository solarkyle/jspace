"""Figure 6: the anatomy of a hallucination (and the detector's honest limits).

A: what wrong answers actually are - substitution / fabrication / mislabeled -
   per model, ordered by capability. Fabrication shrinks as models improve.
B: the detector scorecard - fraction of each error kind flagged by workspace
   noise. It catches fabrication, not substitution; on Qwen it catches neither.
C: stability gradient - wrong answers resample to the same wrong answer at a
   rate set by onset noise (junk-robust containment clustering).
D: deception vs delusion - during instructed lies the truth stays elevated in
   the workspace (E4B, Qwen); believed myths show no such trace (ablit).

Inputs: data/wrong_classified_{e4b,12b,qwen}.json, data/stability_e4b.jsonl,
        data/categories_*.jsonl, out/categories_huihui-*.jsonl
"""
import json

import numpy as np
import matplotlib.pyplot as plt
from scipy import stats as sstats

plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10})
GREEN, RED, GRey, BLUE = "#2e9e5b", "#cf4a42", "#9a9a9a", "#3d6fb5"

fig = plt.figure(figsize=(14, 10.5), facecolor="white")
gs = fig.add_gridspec(2, 2, hspace=0.42, wspace=0.24)

# ---------------- A: error composition ----------------
ax = fig.add_subplot(gs[0, 0])
comp_models = [("gemma-4-e4b-it", "E4B", "data/wrong_classified_e4b.json"),
               ("gemma-4-12b-it", "12B", "data/wrong_classified_12b.json"),
               ("qwen3.6-27b", "Qwen 27B", "data/wrong_classified_qwen.json")]
labels, subs, fabs, rights = [], [], [], []
for slug, name, path in comp_models:
    items = [it for it in json.load(open(path, encoding="utf-8")) if it.get("cls")]
    n = len(items)
    labels.append(f"{name}\n(n={n})")
    subs.append(sum(it["cls"] == "SUBSTITUTION" for it in items) / n * 100)
    fabs.append(sum(it["cls"] == "FABRICATION" for it in items) / n * 100)
    rights.append(sum(it["cls"] == "ACTUALLY_RIGHT" for it in items) / n * 100)
x = np.arange(len(labels))
ax.bar(x, subs, 0.62, color=BLUE, label="substitution (real entity, wrong one)")
ax.bar(x, fabs, 0.62, bottom=subs, color=RED, label="fabrication (invented)")
ax.bar(x, rights, 0.62, bottom=np.array(subs) + np.array(fabs), color=GRey,
       label="actually right (label noise)")
for i in range(len(labels)):
    ax.text(i, subs[i] / 2, f"{subs[i]:.0f}%", ha="center", color="white", fontsize=9)
    ax.text(i, subs[i] + fabs[i] / 2, f"{fabs[i]:.0f}%", ha="center", color="white", fontsize=9)
    ax.text(i, subs[i] + fabs[i] + rights[i] / 2, f"{rights[i]:.0f}%", ha="center",
            color="white", fontsize=8)
ax.set_xticks(x, labels)
ax.set_ylabel("% of answers graded wrong")
ax.set_title("A. What wrong answers actually are\n(two graders, kappa 0.88; 9-15% aren't even wrong)",
             fontsize=11)
ax.legend(fontsize=8, loc="lower right", framealpha=0.95)
ax.spines[["top", "right"]].set_visible(False)

# ---------------- B: detector scorecard ----------------
ax = fig.add_subplot(gs[0, 1])
score = {"E4B": (67, 51), "12B": (72, 49), "Qwen 27B": (53, 57)}
x = np.arange(len(score))
fab_v = [v[0] for v in score.values()]
sub_v = [v[1] for v in score.values()]
ax.bar(x - 0.18, fab_v, 0.34, color=RED, label="fabrications flagged")
ax.bar(x + 0.18, sub_v, 0.34, color=BLUE, label="substitutions flagged")
ax.axhline(50, color=GRey, lw=1, ls="--")
ax.text(2.42, 51.5, "coin flip", color=GRey, fontsize=8)
for i, (f, s) in enumerate(zip(fab_v, sub_v)):
    ax.text(i - 0.18, f + 1.5, f"{f}%", ha="center", fontsize=9)
    ax.text(i + 0.18, s + 1.5, f"{s}%", ha="center", fontsize=9)
ax.set_xticks(x, list(score))
ax.set_ylabel("% flagged by workspace noise")
ax.set_ylim(0, 90)
ax.set_title("B. The detector catches improvisation, not wrong beliefs\n(and on Qwen, neither: the errors that remain are beliefs)",
             fontsize=11)
ax.legend(fontsize=8, framealpha=0.95)
ax.spines[["top", "right"]].set_visible(False)

# ---------------- C: stability gradient ----------------
ax = fig.add_subplot(gs[1, 0])
rows = [json.loads(l) for l in open("data/stability_e4b.jsonl", encoding="utf-8") if l.strip()]

def norm(s):
    s = s.split("thought")[0]
    return "".join(c for c in s.lower() if c.isalnum() or c == " ").strip()

def modal_matches(r):
    reps, counts = [], []
    for s in r["samples"]:
        ns = norm(s)
        if not ns:
            continue
        placed = False
        for i, rep in enumerate(reps):
            a, b = (ns, rep) if len(ns) <= len(rep) else (rep, ns)
            if len(a) >= 3 and a in b:
                counts[i] += 1
                if len(ns) < len(rep):
                    reps[i] = ns
                placed = True
                break
        if not placed:
            reps.append(ns); counts.append(1)
    if not reps:
        return None
    mi = int(np.argmax(counts))
    no = norm(r["orig_answer"])
    a, b = (no, reps[mi]) if len(no) <= len(reps[mi]) else (reps[mi], no)
    return float(len(a) >= 3 and a in b)

wrong = [r for r in rows if r["group"] != "correct"]
ents = np.array([r["mean_entropy"] for r in wrong])
qs = np.quantile(ents, [0.25, 0.5, 0.75])
bins = [(-np.inf, qs[0]), (qs[0], qs[1]), (qs[1], qs[2]), (qs[2], np.inf)]
vals, ns = [], []
for lo, hi in bins:
    mm = [modal_matches(r) for r, e in zip(wrong, ents) if lo < e <= hi]
    mm = [v for v in mm if v is not None]
    vals.append(np.mean(mm) * 100); ns.append(len(mm))
corr_val = np.mean([v for v in (modal_matches(r) for r in rows if r["group"] == "correct")
                    if v is not None]) * 100
x = np.arange(4)
colors = [plt.cm.RdYlGn_r(0.15 + 0.7 * i / 3) for i in range(4)]
ax.bar(x, vals, 0.62, color=colors)
ax.axhline(corr_val, color=GREEN, lw=1.6, ls="--")
ax.text(3.45, corr_val - 4.5, f"correct answers: {corr_val:.0f}%", color=GREEN,
        fontsize=8.5, ha="right")
for i, (v, n) in enumerate(zip(vals, ns)):
    ax.text(i, v + 1.5, f"{v:.0f}%", ha="center", fontsize=9)
ax.set_xticks(x, ["cleanest\nquartile", "Q2", "Q3", "noisiest\nquartile"])
ax.set_ylabel("% resampling to the SAME wrong answer (6x, T=1)")
ax.set_title("C. Clean-wrong answers are stable wrong beliefs\n(wrong answers by onset workspace noise, E4B)",
             fontsize=11)
ax.set_ylim(0, 105)
ax.spines[["top", "right"]].set_visible(False)

# ---------------- D: deception vs delusion ----------------
ax = fig.add_subplot(gs[1, 1])
# lies v2: type-matched controls, honest baseline verified, contamination
# (control == emitted lie) excluded per analysis/analyze_lies_v2.py
lie_sources = [("E4B", "data/lies_v2_gemma-4-e4b-it.jsonl", "instructed_lie"),
               ("12B", "data/lies_v2_gemma-4-12b-it.jsonl", "instructed_lie"),
               ("Qwen 27B", "data/lies_v2_qwen3.6-27b.jsonl", "instructed_lie"),
               ("26B MoE", "data/lies_v2_gemma-4-26b-a4b-it.jsonl", "instructed_lie"),
               ("12B ablit", "data/lies_v2_huihui-gemma-4-12b-it-abliterated.jsonl", "instructed_lie")]
CTRL_OF = {it["id"]: it["control"] for it in
           json.load(open("probes/categories.json", encoding="utf-8"))["items"]
           if it["category"] == "instructed_lie"}
NORM = lambda s: "".join(c for c in s.lower() if c.isalnum() or c == " ").strip()
names, t_meds, c_meds, ps = [], [], [], []
for name, path, cat in lie_sources:
    rws = [json.loads(l) for l in open(path, encoding="utf-8") if l.strip()]
    if cat:
        rws = [r for r in rws if r["category"] == cat]
    t, c = [], []
    for r in rws:
        if not r.get("truth"):
            continue
        # exclude contaminated items: the control token IS the emitted lie
        if NORM(CTRL_OF.get(r["id"], "")) and NORM(CTRL_OF[r["id"]]) in NORM(r["answer"]):
            continue
        snap = r["snapshots"][0]
        tr = [s["rank_truth"] for s in snap["layers"].values() if s["rank_truth"] is not None]
        cr = [s.get("rank_control") for s in snap["layers"].values()
              if s.get("rank_control") is not None]
        if tr and cr:
            t.append(min(tr)); c.append(min(cr))
    t, c = np.array(t, float), np.array(c, float)
    _, p = sstats.wilcoxon(np.log1p(t), np.log1p(c))
    names.append(name); t_meds.append(np.median(t)); c_meds.append(np.median(c)); ps.append(p)

y = np.arange(len(names))[::-1]
for yi, tm, cm, p, nm in zip(y, t_meds, c_meds, ps, names):
    ax.plot([tm, cm], [yi, yi], color=GRey, lw=1.2, zorder=1)
    ax.scatter([tm], [yi], s=70, color=GREEN, zorder=3,
               label="truth (median best rank)" if yi == y[0] else None)
    ax.scatter([cm], [yi], s=70, color=GRey, zorder=3,
               label="mismatched control" if yi == y[0] else None)
    star = "***" if p < 0.001 else "*" if p < 0.05 else "ns"
    ax.text(max(tm, cm) * 1.7, yi, f"p={p:.3f} {star}", fontsize=8.5, va="center")
ax.set_yticks(y, names)
ax.set_xscale("log")
ax.set_xlabel("workspace rank of the TRUE answer while lying (log, lower = more present)")
ax.set_xlim(20, 3e5)
ax.set_title("D. Deception leaves a trace; believed myths do not\n(lies v2: type-matched controls, honest baseline verified, contamination excluded)",
             fontsize=11)
ax.legend(fontsize=8, loc="upper right", framealpha=0.95)
ax.spines[["top", "right"]].set_visible(False)

fig.suptitle("Anatomy of a hallucination: what the workspace sees, and what it honestly cannot",
             fontsize=13.5, y=0.99)
fig.savefig("assets/figure6_anatomy.png", dpi=160, bbox_inches="tight")
print("wrote assets/figure6_anatomy.png")
