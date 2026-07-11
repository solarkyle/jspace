"""Score Gate B (operational routing value), as pre-registered in
campaign/PREREG_STAGE1.md:

    At a fixed 20 percent routing budget (escalate the riskiest 20 percent of
    queries), the combined detector catches at least 5 percentage points more
    wrong answers than logprob-only, averaged over held-out datasets, with the
    bootstrap 95 percent CI excluding zero. HIT if both hold; MISS otherwise.
    Report per-dataset catch rates regardless.

Catch rate at budget b on a held-out dataset = (true errors among the top-b
fraction of rows ranked by detector score) / (all true errors in the dataset).
Detectors are LightGBM (the registered production model) trained under LODO on
the other datasets. The bootstrap resamples split_groups within each held-out
dataset (cluster bootstrap), recomputes per-dataset deltas, and averages.

    python -m campaign.score_gate_b --input out/campaign/stage1_features.jsonl
"""

from __future__ import annotations

import argparse
import json

import numpy as np

from campaign.split_groups import leave_one_dataset_out
from campaign.train_baselines import LOGPROB, _matrix

BUDGET = 0.20
NEED_PP = 5.0
N_BOOT = 2000
SEED = 0


def _lgbm():
    import lightgbm as lgb
    return lgb.LGBMClassifier(n_estimators=300, learning_rate=0.05,
                              num_leaves=31, min_child_samples=20,
                              subsample=0.9, colsample_bytree=0.9,
                              class_weight="balanced", verbosity=-1)


def catch_rate(y, score, budget):
    """Fraction of true errors inside the top-`budget` scored rows."""
    n_err = int(y.sum())
    if n_err == 0:
        return float("nan")
    k = max(1, int(round(budget * len(y))))
    routed = np.argsort(-score)[:k]
    return float(y[routed].sum() / n_err)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True)
    ap.add_argument("--budget", type=float, default=BUDGET)
    ap.add_argument("--out", default="")
    args = ap.parse_args()

    rows = [json.loads(l) for l in open(args.input, encoding="utf-8") if l.strip()]
    labeled = [r for r in rows if "label" in r]
    names = sorted(labeled[0]["features"].keys())
    lp_names = [n for n in names if n in LOGPROB]
    if not lp_names:
        raise SystemExit(f"no logprob-family features found; got {names[:8]}...")

    from sklearn.preprocessing import StandardScaler

    y_all = np.array([r["label"] for r in labeled])
    X_comb = _matrix(labeled, names)
    X_lp = _matrix(labeled, lp_names)
    groups = np.array([r["split_group"] for r in labeled])

    per_ds = []       # (dataset, catch_lp, catch_comb, delta_pp)
    boot_deltas = {}  # dataset -> per-replicate delta array
    rng = np.random.default_rng(SEED)

    for held, tr, te in leave_one_dataset_out(labeled):
        yte = y_all[te]
        if len(set(yte)) < 2 or len(set(y_all[tr])) < 2:
            print(f"{held:>11}: no error variation - skipped")
            continue

        scores = {}
        for fam, X in (("logprob", X_lp), ("combined", X_comb)):
            sc = StandardScaler().fit(X[tr])
            clf = _lgbm().fit(sc.transform(X[tr]), y_all[tr])
            scores[fam] = clf.predict_proba(sc.transform(X[te]))[:, 1]

        c_lp = catch_rate(yte, scores["logprob"], args.budget)
        c_cb = catch_rate(yte, scores["combined"], args.budget)
        per_ds.append((held, c_lp, c_cb, (c_cb - c_lp) * 100))

        # cluster bootstrap over split_groups within the held-out dataset
        gte = groups[te]
        uniq = np.unique(gte)
        idx_by_group = {g: np.where(gte == g)[0] for g in uniq}
        deltas = np.empty(N_BOOT)
        for b in range(N_BOOT):
            take = rng.choice(uniq, size=len(uniq), replace=True)
            idx = np.concatenate([idx_by_group[g] for g in take])
            yb = yte[idx]
            if yb.sum() == 0 or yb.sum() == len(yb):
                deltas[b] = np.nan
                continue
            deltas[b] = (catch_rate(yb, scores["combined"][idx], args.budget)
                         - catch_rate(yb, scores["logprob"][idx], args.budget)) * 100
        boot_deltas[held] = deltas

    print(f"\n== Gate B: catch-rate at {args.budget:.0%} routing budget "
          f"(LightGBM, LODO) ==")
    print(f"{'held-out':>11} {'catch_lp':>9} {'catch_comb':>10} {'delta_pp':>9}")
    for ds, c_lp, c_cb, d in per_ds:
        print(f"{ds:>11} {c_lp*100:>8.1f}% {c_cb*100:>9.1f}% {d:>+8.2f}")

    mean_delta = float(np.mean([d for _, _, _, d in per_ds]))
    # mean-over-datasets per bootstrap replicate (paired across datasets)
    mat = np.vstack([boot_deltas[ds] for ds, *_ in per_ds])
    rep_means = np.nanmean(mat, axis=0)
    lo, hi = np.nanpercentile(rep_means, [2.5, 97.5])

    hit = mean_delta >= NEED_PP and lo > 0
    print(f"\nmean delta: {mean_delta:+.2f}pp   (need >= +{NEED_PP:.0f}pp)")
    print(f"bootstrap 95% CI (cluster, {N_BOOT} reps): [{lo:+.2f}, {hi:+.2f}]"
          f"   (need CI > 0)")
    print(f"verdict: {'HIT' if hit else 'MISS'}")
    if not hit and mean_delta > 0 and lo > 0:
        print("  (positive and CI excludes zero, but below the registered "
              "+5pp magnitude)")

    if args.out:
        json.dump({"budget": args.budget, "per_dataset": [
                       {"dataset": ds, "catch_logprob": c_lp,
                        "catch_combined": c_cb, "delta_pp": d}
                       for ds, c_lp, c_cb, d in per_ds],
                   "mean_delta_pp": mean_delta, "ci95": [lo, hi],
                   "verdict": "HIT" if hit else "MISS"},
                  open(args.out, "w", encoding="utf-8"), indent=1)
        print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
