"""Score Gate D (prospective zero-shot transfer), per campaign/PREREG_STAGE2.md.

The FROZEN Stage 1 classifiers (combined / logprob / workspace; hashes verified
on load) score Stage 2 feature rows zero-shot. Per NEW source:

  - primary: AUROC(frozen combined) - AUROC(frozen logprob)   [the increment]
  - secondary: AUROC(frozen workspace) vs logprob; absolute combined AUROC;
    catch-rate at 20% routing budget (combined vs logprob)

Gate D HIT if mean increment over the five new sources >= +0.02 AND the
increment is positive on >= 4 of 5. squad_v2 (regen) is scored separately -
it is a repaired Stage 1 source, not a new one. Degenerate sources
(error rate < 2% or > 98%) are excluded from breadth with the exclusion stated.

    python -m campaign.score_gate_d --input out/campaign/stage2_features.jsonl
"""

from __future__ import annotations

import argparse
import json

import numpy as np

from campaign.freeze_classifier import score_rows
from campaign.score_gate_b import catch_rate

NEW_SOURCES = ("truthfulqa", "nq_open", "facts_grounding",
               "legal_hallucinations", "bfcl")
SEPARATE = ("squad_v2",)
NEED_MEAN = 0.02
NEED_BREADTH = 4  # of 5


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True)
    ap.add_argument("--out", default="")
    args = ap.parse_args()
    from sklearn.metrics import roc_auc_score

    rows = [json.loads(l) for l in open(args.input, encoding="utf-8") if l.strip()]
    labeled = [r for r in rows if "label" in r]
    print(f"{len(labeled)} labeled of {len(rows)} rows")

    scores = {fam: score_rows(labeled, fam)
              for fam in ("combined", "logprob", "workspace")}
    y = np.array([r["label"] for r in labeled])
    src = np.array([r["source_dataset"] for r in labeled])

    results, increments = {}, []
    print(f"\n{'source':>21} {'n':>5} {'err%':>5} {'AUC_lp':>7} {'AUC_ws':>7} "
          f"{'AUC_comb':>8} {'increment':>9} {'catch20_lp':>10} {'catch20_cb':>10}")
    for s in NEW_SOURCES + SEPARATE:
        m = src == s
        ys = y[m]
        if m.sum() == 0:
            print(f"{s:>21}  (no rows)")
            continue
        err = float(ys.mean())
        if err < 0.02 or err > 0.98:
            print(f"{s:>21} {m.sum():>5} {err*100:>4.0f}%  DEGENERATE - excluded")
            results[s] = {"n": int(m.sum()), "error_rate": err,
                          "degenerate": True}
            continue
        auc = {f: float(roc_auc_score(ys, scores[f][m]))
               for f in ("combined", "logprob", "workspace")}
        inc = auc["combined"] - auc["logprob"]
        c_lp = catch_rate(ys, scores["logprob"][m], 0.20)
        c_cb = catch_rate(ys, scores["combined"][m], 0.20)
        results[s] = {"n": int(m.sum()), "error_rate": err, "auroc": auc,
                      "increment": inc, "catch20_logprob": c_lp,
                      "catch20_combined": c_cb, "new_source": s in NEW_SOURCES}
        if s in NEW_SOURCES:
            increments.append(inc)
        print(f"{s:>21} {m.sum():>5} {err*100:>4.0f}% {auc['logprob']:>7.3f} "
              f"{auc['workspace']:>7.3f} {auc['combined']:>8.3f} {inc:>+9.4f} "
              f"{c_lp*100:>9.1f}% {c_cb*100:>9.1f}%")

    mean_inc = float(np.mean(increments)) if increments else float("nan")
    pos = sum(1 for i in increments if i > 0)
    hit = (mean_inc >= NEED_MEAN and pos >= NEED_BREADTH
           and len(increments) >= NEED_BREADTH)
    print(f"\n== Gate D (prospective zero-shot transfer) ==")
    print(f"mean increment over {len(increments)} evaluable new sources: "
          f"{mean_inc:+.4f} (need >= +{NEED_MEAN})")
    print(f"breadth positive: {pos}/{len(increments)} (need >= {NEED_BREADTH}/5)")
    print(f"verdict: {'HIT' if hit else 'MISS'}")

    if args.out:
        json.dump({"per_source": results, "mean_increment_new": mean_inc,
                   "breadth_positive": pos, "n_evaluable": len(increments),
                   "verdict": "HIT" if hit else "MISS"},
                  open(args.out, "w", encoding="utf-8"), indent=1)
        print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
