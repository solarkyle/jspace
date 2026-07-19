"""Phase 1 + 2 analysis: cross-model hallucination replication + fake entities.

Reads out/uncertainty_trivia_<slug>.jsonl and out/uncertainty_fake_<slug>.jsonl
(from analysis/modal_fit.py::uncertainty). Reports, per model:
  - overall accuracy, entropy/baseline AUCs
  - the blind-spot head-to-head (entropy vs residual logprob among high-conf)
  - the quadrant table
  - threshold transfer: E4B's z-scored entropy threshold applied verbatim
Fake-entity section: does workspace entropy detect fabricated entities better
than output confidence does, especially among fluent (non-hedging) answers?
"""
import glob
import json
import os
import sys

import numpy as np

sys.stdout.reconfigure(encoding="utf-8", errors="replace")

ORDER = ["gemma-4-e4b-it", "gemma-4-12b-it", "huihui-gemma-4-12b-it-abliterated",
         "gemma-4-26b-a4b-it", "qwen3.6-27b"]
NAME = {"gemma-4-e4b-it": "E4B-dense", "gemma-4-12b-it": "12B-dense",
        "huihui-gemma-4-12b-it-abliterated": "12B-ablit",
        "gemma-4-26b-a4b-it": "26B-MoE", "qwen3.6-27b": "Qwen-27B"}

def auc(score, label):
    o = np.argsort(score); r = np.empty(len(score)); r[o] = np.arange(1, len(score) + 1)
    n1 = label.sum(); n0 = len(label) - n1
    if n1 == 0 or n0 == 0:
        return float("nan")
    return (r[label == 1].sum() - n1 * (n1 + 1) / 2) / (n0 * n1)

def o(a):  # orient
    return max(a, 1 - a)

def load(tag, slug):
    paths = glob.glob(f"out/uncertainty_{tag}_{slug}.jsonl")
    if not paths:
        return None
    return [json.loads(l) for l in open(paths[0], encoding="utf-8")]

HEDGES = ["unknown", "i don't know", "i do not know", "not sure", "no such",
          "fictional", "does not exist", "doesn't exist", "no record", "none",
          "there is no", "unanswerable", "cannot", "n/a", "no widely",
          # refusal phrasings observed in the actual outputs
          "not won", "never won", "did not win", "didn't win", "no nobel",
          "hasn't won", "has not", "never wrote", "did not write", "no battle",
          "not a real", "not real", "no chemical", "not an element",
          "no element", "not a recognized", "no country", "not exist"]

print("=" * 78)
print("PHASE 1 - TriviaQA 500, all models (gate: quadrant must hold on 3+ of 4 new)")
print("=" * 78)
gate_pass = 0
gate_total = 0
e4b_thr = None
for slug in ORDER:
    rows = load("trivia", slug)
    if rows is None:
        print(f"  {NAME[slug]:<10} (not landed yet)")
        continue
    y = np.array([r["correct"] for r in rows]).astype(float)
    ent = np.array([r["mean_entropy"] for r in rows])
    lp = np.array([r["bl_first_token_logprob"] for r in rows])
    hc = lp > np.median(lp)
    qe = ent < np.median(ent)
    acc_cc = y[hc & qe].mean()      # output conf + workspace clean
    acc_cn = y[hc & ~qe].mean()     # output conf + workspace NOISY
    gap = acc_cc - acc_cn
    a_ent = o(auc(ent[hc], 1 - y[hc]))
    a_lp = o(auc(-lp[hc], 1 - y[hc]))
    zthr = (np.median(ent) - ent.mean()) / ent.std()
    if slug == "gemma-4-e4b-it":
        e4b_thr = 0.0  # E4B's median in z-space is ~0 by construction; transfer below uses z>0
    else:
        gate_total += 1
        if gap > 0.10 and a_ent > a_lp:
            gate_pass += 1
    print(f"  {NAME[slug]:<10} acc {y.mean():.3f} | conf+clean {acc_cc:.3f} vs "
          f"conf+noisy {acc_cn:.3f} (gap {gap*100:+.0f}pt) | blind-spot AUC "
          f"ent {a_ent:.3f} vs lp {a_lp:.3f} | ent-alone {o(auc(ent, y)):.3f} "
          f"lp-alone {o(auc(lp, y)):.3f}")
print(f"\n  GATE (gap>10pt AND entropy beats residual logprob in blind spot): "
      f"{gate_pass}/{gate_total} new models")

print()
print("  Threshold transfer: escalate when z(entropy) > 0 (E4B rule, no per-model calibration)")
for slug in ORDER:
    rows = load("trivia", slug)
    if rows is None:
        continue
    y = np.array([r["correct"] for r in rows]).astype(float)
    ent = np.array([r["mean_entropy"] for r in rows])
    z = (ent - ent.mean()) / ent.std()
    esc = z > 0
    caught = ((y == 0) & esc).sum() / max(1, (y == 0).sum())
    print(f"    {NAME[slug]:<10} escalates {esc.mean():.0%}, catches {caught:.0%} of wrong answers")

print()
print("=" * 78)
print("PHASE 2 - fake entities (50 real / 50 fabricated, matched templates)")
print("=" * 78)
for slug in ORDER:
    rows = load("fake", slug)
    if rows is None:
        print(f"  {NAME[slug]:<10} (not landed yet)")
        continue
    real = np.array([r["entity_real"] for r in rows]).astype(float)
    ent = np.array([r["mean_entropy"] for r in rows])
    lp = np.array([r["bl_first_token_logprob"] for r in rows])
    ans = [r["answer"].lower() for r in rows]
    hedged = np.array([any(h in a for h in HEDGES) for a in ans])
    # detection: does the signal know the entity is fake?
    a_ent = o(auc(ent, 1 - real))
    a_lp = o(auc(-lp, 1 - real))
    # behavior: how often does it fabricate fluently on fakes?
    fab = (~hedged) & (real == 0)
    n_fake = int((real == 0).sum())
    # among fluent answers only (the dangerous ones), can entropy find the fakes?
    fl = ~hedged
    a_ent_fl = o(auc(ent[fl], 1 - real[fl])) if 0 < real[fl].sum() < fl.sum() else float("nan")
    a_lp_fl = o(auc(-lp[fl], 1 - real[fl])) if 0 < real[fl].sum() < fl.sum() else float("nan")
    print(f"  {NAME[slug]:<10} fabricates fluently on {fab.sum()}/{n_fake} fakes | "
          f"fake-detect AUC: ent {a_ent:.3f} lp {a_lp:.3f} | "
          f"among fluent answers only: ent {a_ent_fl:.3f} lp {a_lp_fl:.3f}")
    ex = [r for r in rows if not r["entity_real"]
          and not any(h in r["answer"].lower() for h in HEDGES)][:3]
    for r in ex:
        print(f"      e.g. \"{r['q']}\" -> \"{r['answer']}\" (ent {r['mean_entropy']:.2f})")
