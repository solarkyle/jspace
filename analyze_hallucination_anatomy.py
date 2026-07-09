"""Anatomy of a hallucination: how many ways is a model wrong, internally?

Uses the committed TriviaQA traces (500 Q/model, layer_entropies + features +
correctness). No GPU. Questions:

  A. Among WRONG answers, does the workspace show distinct modes?
     k-means on the normalized entropy trajectory; report cluster stats.
  B. The blind-spot census: what fraction of wrong answers are internally
     indistinguishable from correct ones (clean-wrong), per model?
  C. Do clean-wrong and noisy-wrong differ BEHAVIORALLY? (confidence, length,
     ignition, hedging) -> prints exemplars of each mode for eyeball/Gemini
     classification (real-entity substitution vs fabrication).

Usage:
    python analyze_hallucination_anatomy.py [--model gemma-4-e4b-it] [--k 3]
"""

import argparse
import json
from collections import Counter

import numpy as np

TRACE = "data/uncertainty_trivia_{}.jsonl"
MODELS = ["gemma-4-e4b-it", "gemma-4-12b-it", "huihui-gemma-4-12b-it-abliterated",
          "gemma-4-26b-a4b-it", "qwen3.6-27b"]


def load(model: str) -> list[dict]:
    rows = [json.loads(l) for l in open(TRACE.format(model), encoding="utf-8")
            if l.strip()]
    return [r for r in rows if r.get("layer_entropies")]


def traj_matrix(rows: list[dict], n_points: int = 16) -> np.ndarray:
    """Resample each entropy trajectory to n_points so models are comparable."""
    out = []
    for r in rows:
        e = np.asarray(r["layer_entropies"], dtype=float)
        x = np.linspace(0, 1, len(e))
        xi = np.linspace(0, 1, n_points)
        out.append(np.interp(xi, x, e))
    return np.asarray(out)


def kmeans(X: np.ndarray, k: int, seed: int = 0, iters: int = 100):
    rng = np.random.default_rng(seed)
    C = X[rng.choice(len(X), k, replace=False)]
    for _ in range(iters):
        d = ((X[:, None, :] - C[None]) ** 2).sum(-1)
        lab = d.argmin(1)
        newC = np.array([X[lab == j].mean(0) if (lab == j).any() else C[j]
                         for j in range(k)])
        if np.allclose(newC, C):
            break
        C = newC
    return lab, C


def describe(rows: list[dict], idx: np.ndarray, tag: str) -> None:
    sel = [rows[i] for i in np.nonzero(idx)[0]]
    if not sel:
        print(f"  {tag}: empty")
        return
    f = lambda k: np.mean([r[k] for r in sel])
    print(f"  {tag:>14}: n={len(sel):>3}  conf={f('bl_first_token_logprob'):+.2f}  "
          f"ent={f('mean_entropy'):.2f}  ign={f('ignition_frac'):.2f}  "
          f"agree={f('band_agreement'):.2f}  hedge={f('best_hedge_rank_log'):.1f}  "
          f"len={f('bl_answer_len'):.1f}")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--model", default="gemma-4-e4b-it")
    ap.add_argument("--k", type=int, default=3)
    ap.add_argument("--dump_exemplars", default="",
                    help="write wrong-answer exemplars per mode to this JSON")
    args = ap.parse_args()

    print("=" * 72)
    print("B. BLIND-SPOT CENSUS across all models")
    print("   (clean-wrong = wrong answer whose workspace entropy is below the")
    print("    CORRECT-answer median: internally it looks like knowledge)")
    print("=" * 72)
    for m in MODELS:
        try:
            rows = load(m)
        except FileNotFoundError:
            continue
        ent = np.array([r["mean_entropy"] for r in rows])
        correct = np.array([bool(r["correct"]) for r in rows])
        med_correct = np.median(ent[correct])
        wrong = ~correct
        clean_wrong = wrong & (ent <= med_correct)
        conf = np.array([r["bl_first_token_logprob"] for r in rows])
        confident = conf >= np.median(conf)
        print(f"{m:>38}: wrong={wrong.sum():>3}  clean-wrong={clean_wrong.sum():>3} "
              f"({clean_wrong.sum()/max(wrong.sum(),1):.0%} of wrong)  "
              f"clean-wrong AND confident={(clean_wrong & confident).sum():>3} "
              f"({(clean_wrong & confident).sum()/max(wrong.sum(),1):.0%})")

    rows = load(args.model)
    correct = np.array([bool(r["correct"]) for r in rows])
    wrong_rows = [r for r in rows if not r["correct"]]
    print()
    print("=" * 72)
    print(f"A. WRONG-ANSWER MODES on {args.model} "
          f"(k-means k={args.k} on entropy trajectories, wrong answers only)")
    print("=" * 72)
    X = traj_matrix(wrong_rows)
    Xz = (X - X.mean(0)) / (X.std(0) + 1e-9)
    lab, C = kmeans(Xz, args.k)
    order = np.argsort([-X[lab == j].mean() for j in range(args.k)])
    names = {}
    for rank, j in enumerate(order):
        names[j] = f"mode{rank}"
    for rank, j in enumerate(order):
        idx = lab == j
        traj = X[idx].mean(0)
        arrow = " ".join(f"{v:.1f}" for v in traj[::3])
        print(f"\n mode{rank} (n={idx.sum()}): mean trajectory shallow->deep: {arrow}")
        describe(wrong_rows, idx, f"mode{rank}")
        ex = [wrong_rows[i] for i in np.nonzero(idx)[0][:4]]
        for r in ex:
            print(f"     e.g. Q: {r['q'][:60]}")
            print(f"          A: {r['answer'][:50]!r}")

    # Correct-answer reference trajectory for comparison
    Xc = traj_matrix([r for r in rows if r["correct"]])
    print(f"\n correct-answer reference trajectory: "
          + " ".join(f"{v:.1f}" for v in Xc.mean(0)[::3]))

    print()
    print("=" * 72)
    print("C. CLEAN-WRONG vs NOISY-WRONG behavioral profile "
          f"({args.model}, median split on mean_entropy among wrong)")
    print("=" * 72)
    ent_w = np.array([r["mean_entropy"] for r in wrong_rows])
    med = np.median(ent_w)
    describe(wrong_rows, ent_w <= med, "clean-wrong")
    describe(wrong_rows, ent_w > med, "noisy-wrong")

    if args.dump_exemplars:
        dump = []
        for i, r in enumerate(wrong_rows):
            dump.append({
                "q": r["q"], "answer": r["answer"],
                "mode": names[lab[i]] if i < len(lab) else None,
                "clean_wrong": bool(ent_w[i] <= med),
                "conf": r["bl_first_token_logprob"],
            })
        with open(args.dump_exemplars, "w", encoding="utf-8") as fh:
            json.dump(dump, fh, indent=1)
        print(f"\nwrote {len(dump)} wrong-answer exemplars -> {args.dump_exemplars}")


if __name__ == "__main__":
    main()
