"""Analysis for the graded-clues and tool-call probes (iteration 2).

Clues: does fog track information supply? Can it tell a reasonable miss
(in the valid set) from an unreasonable one? Tools: does the workspace get
foggy right before the model invents a tool that doesn't exist?
"""
import json
import re
import sys

import numpy as np

sys.stdout.reconfigure(encoding="utf-8", errors="replace")

def auc(score, label):
    o = np.argsort(score); r = np.empty(len(score)); r[o] = np.arange(1, len(score)+1)
    n1 = label.sum(); n0 = len(label)-n1
    if n1 == 0 or n0 == 0:
        return float("nan")
    return (r[label == 1].sum() - n1*(n1+1)/2) / (n0*n1)

def norm(s):
    s = s.split("thought")[0]  # strip Gemma thinking artifacts
    return "".join(c for c in s.lower() if c.isalnum() or c == " ").strip()

MODELS = [("gemma-4-e4b-it", "E4B"), ("gemma-4-12b-it", "12B")]

print("=" * 76)
print("GRADED CLUES: information withheld on purpose")
print("=" * 76)
for slug, name in MODELS:
    rows = [json.loads(l) for l in open(f"out/uncertainty_clues_{slug}.jsonl", encoding="utf-8")]
    print(f"\n{name}:")
    print(f"  {'depth':>5} {'n':>4} {'hit target':>10} {'reasonable':>10} {'mean fog':>9}")
    for d in (1, 2, 3):
        sub = [r for r in rows if r["clue_depth"] == d]
        hit = np.array([r["correct"] for r in sub])
        reas = np.array([any(norm(v) and norm(v) in norm(r["answer"]) for v in r["valid_set"])
                         for r in sub])
        fog = np.array([r["mean_entropy"] for r in sub])
        print(f"  {d:>5} {len(sub):>4} {hit.mean():>10.0%} {reas.mean():>10.0%} {fog.mean():>9.2f}")
    # the commenter's question: among MISSES at depths 1-2, does fog separate
    # reasonable (good judgement, in valid set) from unreasonable?
    sub = [r for r in rows if r["clue_depth"] in (1, 2) and not r["correct"]]
    reas = np.array([any(norm(v) and norm(v) in norm(r["answer"]) for v in r["valid_set"])
                     for r in sub])
    fog = np.array([r["mean_entropy"] for r in sub])
    if 0 < reas.sum() < len(reas):
        a = auc(fog, ~reas)  # fog predicting UNREASONABLE
        print(f"  misses at depth 1-2: n={len(sub)}, reasonable {reas.mean():.0%}; "
              f"fog(reasonable) {fog[reas].mean():.2f} vs fog(unreasonable) {fog[~reas].mean():.2f}; "
              f"AUC fog->unreasonable {a:.3f}")
    # sanity within depth: fog -> wrong
    for d in (1, 2, 3):
        sub = [r for r in rows if r["clue_depth"] == d]
        y = np.array([r["correct"] for r in sub]).astype(float)
        fog = np.array([r["mean_entropy"] for r in sub])
        if 0 < y.sum() < len(y):
            print(f"  AUC fog->wrong at fixed depth {d}: {max(auc(fog,1-y),1-auc(fog,1-y)):.3f}")

print()
print("=" * 76)
print("TOOL CALLS: does fog precede invented tools?")
print("=" * 76)
TOOLS = {"web_search", "calculator", "get_weather", "read_file", "send_email", "get_calendar"}
def classify(r):
    ans = r["answer"]
    m = re.search(r'"tool"\s*:\s*"([^"]+)"', ans)
    called = m.group(1) if m else None
    fl = r["flavor"]
    if fl == "solvable":
        if called == r["expected_tool"]:
            return "correct_tool"
        if called in TOOLS:
            return "wrong_tool"
        if called:
            return "INVENTED"
        return "no_call"
    if fl == "missing_tool":
        if called is None:
            return "refused_ok"
        if called in TOOLS:
            return "misused_real"
        return "INVENTED"
    if fl == "no_tool":
        return "answered_ok" if called is None else "unnecessary_call"

for slug, name in MODELS:
    rows = [json.loads(l) for l in open(f"out/uncertainty_tools_{slug}.jsonl", encoding="utf-8")]
    print(f"\n{name}:")
    from collections import Counter, defaultdict
    outc = [classify(r) for r in rows]
    fogby = defaultdict(list)
    for r, o in zip(rows, outc):
        fogby[o].append(r["mean_entropy"])
    for o, c in Counter(outc).most_common():
        print(f"  {o:<16} n={c:<3} mean fog {np.mean(fogby[o]):.2f}")
    # the money comparison: bad tool behavior vs good, fog AUC
    bad = np.array([o in ("INVENTED", "misused_real", "wrong_tool", "unnecessary_call") for o in outc])
    fog = np.array([r["mean_entropy"] for r in rows])
    if 0 < bad.sum() < len(bad):
        print(f"  AUC fog->bad tool behavior (all flavors): {auc(fog, bad):.3f}")
    # within missing_tool only: refused vs (invented|misused)
    idx = [i for i, r in enumerate(rows) if r["flavor"] == "missing_tool"]
    if idx:
        b = np.array([outc[i] != "refused_ok" for i in idx])
        f = np.array([fog[i] for i in idx])
        if 0 < b.sum() < len(b):
            print(f"  within missing_tool: refused {(~b).sum()}, failed {b.sum()}; "
                  f"AUC fog->failure {auc(f, b):.3f}")
    ex = [(rows[i], outc[i]) for i in range(len(rows)) if outc[i] in ("INVENTED", "misused_real")][:3]
    for r, o in ex:
        print(f"    e.g. [{o}] \"{r['q'][:55]}\" -> {r['answer'][:60]!r} (fog {r['mean_entropy']:.2f})")
