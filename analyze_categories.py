"""Analyze the response-type taxonomy traces from probe_categories.py.

Questions, in order:
  H1  Do response types have distinguishable workspace signatures?
      -> per-category feature table, pairwise separability matrix,
         multiclass CV confusion matrix.
  H2  Is INVITED fabrication (creative_invited) distinguishable from
      UNINVITED fabrication (fake_entity answered fluently)?
  H3  Do confident misconceptions (myth repeated) differ from clean correct
      retrieval in trajectory shape, even though entropy reads clean?
  H4  Does grounded copy-from-context differ from parametric retrieval?
  H5  During an instructed lie, does the workspace hold the TRUE answer?
      -> rank-of-truth by layer while the output token is the false one.

Usage:
    python analyze_categories.py [--trace out/categories_gemma-4-e4b-it.jsonl]
"""

import argparse
import json
from collections import defaultdict
from itertools import combinations

import numpy as np

FEATURES = [
    "ent_mean", "ent_late", "ent_slope", "rival_mean", "tail_mean",
    "tail_late", "ign_frac", "ign_depth", "agree", "tail_max3", "ent_max3",
]


def row_features(row: dict) -> dict:
    """Trajectory features from the onset snapshot + max over all snapshots."""
    snap = row["snapshots"][0]
    layers = sorted(snap["layers"].keys(), key=int)
    ent = np.array([snap["layers"][l]["entropy"] for l in layers])
    rival = np.array([snap["layers"][l]["rival_mass"] for l in layers])
    tail = np.array([snap["layers"][l]["tail_mass"] for l in layers])
    rank_gen = np.array([snap["layers"][l]["rank_gen"] for l in layers])
    top1 = np.array([snap["layers"][l]["top_ids"][0] for l in layers])
    gen_id = None  # top_ids are ids; rank_gen==0 marks agreement instead
    n = len(layers)
    late = slice(int(n * 0.75), n)
    x = np.arange(n)
    ign = np.nonzero(rank_gen <= 10)[0]

    def snap_stat(key):
        vals = []
        for s in row["snapshots"]:
            ls = sorted(s["layers"].keys(), key=int)
            vals.append(float(np.mean([s["layers"][l][key] for l in ls])))
        return max(vals)

    return {
        "ent_mean": float(ent.mean()),
        "ent_late": float(ent[late].mean()),
        "ent_slope": float(np.polyfit(x, ent, 1)[0]),
        "rival_mean": float(rival.mean()),
        "tail_mean": float(tail.mean()),
        "tail_late": float(tail[late].mean()),
        "ign_frac": float((rank_gen <= 10).mean()),
        "ign_depth": float(ign[0] / n) if len(ign) else 1.0,
        "agree": float((rank_gen == 0).mean()),
        "tail_max3": snap_stat("tail_mass"),
        "ent_max3": snap_stat("entropy"),
    }


def auc(x: np.ndarray, y: np.ndarray) -> float:
    """AUC of x predicting binary y (1 = positive class)."""
    order = np.argsort(x)
    ranks = np.empty(len(x))
    ranks[order] = np.arange(len(x))
    pos = y == 1
    if pos.sum() == 0 or (~pos).sum() == 0:
        return float("nan")
    return float((ranks[pos].mean() - (pos.sum() - 1) / 2) / (~pos).sum())


def pair_auc(fa: np.ndarray, fb: np.ndarray) -> float:
    """Best single-feature separation between two feature matrices (columns=FEATURES)."""
    best = 0.5
    y = np.concatenate([np.ones(len(fa)), np.zeros(len(fb))])
    for j in range(fa.shape[1]):
        x = np.concatenate([fa[:, j], fb[:, j]])
        a = auc(x, y)
        best = max(best, a, 1 - a)
    return best


def logistic_cv(X: np.ndarray, y: np.ndarray, k: int = 5, seed: int = 0):
    """Multiclass logistic regression, k-fold CV, returns (acc, confusion)."""
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import StratifiedKFold
    from sklearn.preprocessing import StandardScaler

    classes = sorted(set(y))
    conf = np.zeros((len(classes), len(classes)), dtype=int)
    cidx = {c: i for i, c in enumerate(classes)}
    skf = StratifiedKFold(n_splits=k, shuffle=True, random_state=seed)
    correct = 0
    for tr, te in skf.split(X, y):
        sc = StandardScaler().fit(X[tr])
        clf = LogisticRegression(max_iter=2000).fit(sc.transform(X[tr]), y[tr])
        pred = clf.predict(sc.transform(X[te]))
        for yt, yp in zip(y[te], pred):
            conf[cidx[yt], cidx[yp]] += 1
            correct += yt == yp
    return correct / len(y), classes, conf


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--trace", default="out/categories_gemma-4-e4b-it.jsonl")
    ap.add_argument("--lies_trace", default="",
                    help="separate instructed_lie trace that carries rank_control")
    args = ap.parse_args()

    rows = [json.loads(l) for l in open(args.trace, encoding="utf-8") if l.strip()]
    feats = {r["id"]: row_features(r) for r in rows}
    by_cat = defaultdict(list)
    for r in rows:
        by_cat[r["category"]].append(r)
    cats = sorted(by_cat)
    print(f"{len(rows)} rows, {len(cats)} categories: "
          + ", ".join(f"{c}({len(by_cat[c])})" for c in cats))

    # ---- H1a: per-category signature table ----
    print("\n== per-category workspace signature (onset snapshot means) ==")
    hdr = ["category", "n", "ent", "rival", "tail", "ign_fr", "ign_dep", "agree", "tail_mx3"]
    print(" | ".join(f"{h:>9}" for h in hdr))
    for c in cats:
        F = np.array([[feats[r["id"]][f] for f in FEATURES] for r in by_cat[c]])
        m = dict(zip(FEATURES, F.mean(0)))
        print(" | ".join([f"{c:>9.9}", f"{len(by_cat[c]):>9}"] + [
            f"{m[k]:>9.3f}" for k in
            ["ent_mean", "rival_mean", "tail_mean", "ign_frac", "ign_depth", "agree", "tail_max3"]]))

    # ---- H1b: pairwise best-single-feature separability ----
    print("\n== pairwise separability (best single feature AUC) ==")
    mats = {c: np.array([[feats[r["id"]][f] for f in FEATURES] for r in by_cat[c]])
            for c in cats}
    short = {c: c[:8] for c in cats}
    print(" " * 10 + " ".join(f"{short[c]:>8}" for c in cats))
    grid = {}
    for a in cats:
        line = f"{short[a]:>9} "
        for b in cats:
            if a == b:
                line += f"{'-':>8} "
            else:
                v = grid.get((b, a)) or pair_auc(mats[a], mats[b])
                grid[(a, b)] = v
                line += f"{v:>8.2f} "
        print(line)

    # ---- H1c: multiclass ----
    X = np.array([[feats[r["id"]][f] for f in FEATURES] for r in rows])
    y = np.array([r["category"] for r in rows])
    try:
        acc, classes, conf = logistic_cv(X, y)
        print(f"\n== multiclass 5-fold CV accuracy: {acc:.3f} (chance {1/len(cats):.3f}) ==")
        print(" " * 10 + " ".join(f"{short[c]:>8}" for c in classes))
        for i, c in enumerate(classes):
            print(f"{short[c]:>9} " + " ".join(f"{v:>8}" for v in conf[i]))
    except ImportError:
        print("\n(sklearn not installed; skipping multiclass)")

    # ---- H2: invited vs uninvited fabrication ----
    fake_fluent = [r for r in by_cat.get("fake_entity", [])
                   if not any(w in r["answer"].lower()
                              for w in ["not", "no ", "unknown", "fictional", "does not",
                                        "i cannot", "sorry", "there is no"])]
    if fake_fluent and by_cat.get("creative_invited"):
        fa = np.array([[feats[r["id"]][f] for f in FEATURES] for r in by_cat["creative_invited"]])
        fb = np.array([[feats[r["id"]][f] for f in FEATURES] for r in fake_fluent])
        print(f"\n== H2 invited fabrication vs fluent-fake-entity "
              f"(n={len(fa)} vs {len(fb)}) ==")
        y2 = np.concatenate([np.ones(len(fa)), np.zeros(len(fb))])
        for j, f in enumerate(FEATURES):
            x = np.concatenate([fa[:, j], fb[:, j]])
            a = auc(x, y2)
            print(f"  {f:>16}: AUC {max(a, 1 - a):.3f} "
                  f"({'creative higher' if a > 0.5 else 'fake higher'})")

    # ---- H3: misconception myth-repeat vs correct retrieval ----
    myths_wrong = [r for r in by_cat.get("misconception", []) if r["correct"] is False]
    retr_right = [r for r in by_cat.get("retrieval_easy", []) if r["correct"]]
    if myths_wrong:
        fa = np.array([[feats[r["id"]][f] for f in FEATURES] for r in myths_wrong])
        fb = np.array([[feats[r["id"]][f] for f in FEATURES] for r in retr_right])
        print(f"\n== H3 myth-repeated ({len(fa)}) vs correct easy retrieval ({len(fb)}) ==")
        y3 = np.concatenate([np.ones(len(fa)), np.zeros(len(fb))])
        for j, f in enumerate(FEATURES):
            x = np.concatenate([fa[:, j], fb[:, j]])
            a = auc(x, y3)
            if abs(a - 0.5) > 0.15:
                print(f"  {f:>16}: AUC {max(a, 1 - a):.3f} "
                      f"({'myth higher' if a > 0.5 else 'retrieval higher'})")
        print("  answers that repeated the myth:",
              [r["id"] for r in myths_wrong])
    else:
        print("\n== H3: model debunked every myth; no myth-repeat cases on this model ==")

    # ---- H4: grounded vs parametric retrieval ----
    if by_cat.get("grounded") and retr_right:
        fa = np.array([[feats[r["id"]][f] for f in FEATURES] for r in by_cat["grounded"]])
        fb = np.array([[feats[r["id"]][f] for f in FEATURES] for r in retr_right])
        print(f"\n== H4 grounded copy ({len(fa)}) vs parametric retrieval ({len(fb)}) ==")
        y4 = np.concatenate([np.ones(len(fa)), np.zeros(len(fb))])
        for j, f in enumerate(FEATURES):
            x = np.concatenate([fa[:, j], fb[:, j]])
            a = auc(x, y4)
            if abs(a - 0.5) > 0.15:
                print(f"  {f:>16}: AUC {max(a, 1 - a):.3f} "
                      f"({'grounded higher' if a > 0.5 else 'parametric higher'})")

    # ---- confound check: are the signatures just prompt length? ----
    plen = np.array([len(r["q"]) for r in rows])
    print("\n== confound: |corr(feature, prompt chars)| ==")
    for f in FEATURES:
        v = np.array([feats[r["id"]][f] for r in rows])
        c = np.corrcoef(plen, v)[0, 1]
        flag = "  <-- check" if abs(c) > 0.4 else ""
        print(f"  {f:>16}: {c:+.2f}{flag}")

    # ---- H5: does the workspace hold the truth during an instructed lie? ----
    lies = [r for r in by_cat.get("instructed_lie", []) if r.get("truth")]
    if args.lies_trace:
        lies = [json.loads(l) for l in open(args.lies_trace, encoding="utf-8") if l.strip()]
        lies = [r for r in lies if r.get("truth")]
    if lies:
        print(f"\n== H5 instructed lies (n={len(lies)}): rank of TRUE answer in workspace ==")
        best_truth, onset_truth_by_layer = [], defaultdict(list)
        lied = 0
        for r in lies:
            norm = lambda s: "".join(c for c in s.lower() if c.isalnum() or c == " ").strip()
            actually_lied = norm(r["truth"]) not in norm(r["answer"])
            lied += actually_lied
            snap = r["snapshots"][0]
            ranks = {int(l): s["rank_truth"] for l, s in snap["layers"].items()
                     if s["rank_truth"] is not None}
            if ranks:
                best_truth.append(min(ranks.values()))
                for l, v in ranks.items():
                    onset_truth_by_layer[l].append(v)
        bt = np.array(best_truth)
        print(f"  actually lied: {lied}/{len(lies)}")
        print(f"  best truth rank across band, median {np.median(bt):.0f}, "
              f"frac<=10: {(bt <= 10).mean():.2f}, frac<=100: {(bt <= 100).mean():.2f}")
        # paired null: rank of a MISMATCHED truth in the same snapshots
        ctrl = []
        for r in lies:
            snap = r["snapshots"][0]
            cr = [s.get("rank_control") for s in snap["layers"].values()
                  if s.get("rank_control") is not None]
            if cr:
                ctrl.append(min(cr))
        if ctrl:
            ct = np.array(ctrl)
            wins = sum(a < b for a, b in zip(best_truth, ctrl))
            print(f"  CONTROL (mismatched truth): median best rank {np.median(ct):.0f}, "
                  f"frac<=10: {(ct <= 10).mean():.2f}, frac<=100: {(ct <= 100).mean():.2f}")
            print(f"  paired: truth outranks control on {wins}/{len(ctrl)} items")
        print("  median truth rank by layer (shallow->deep):")
        for l in sorted(onset_truth_by_layer):
            v = np.median(onset_truth_by_layer[l])
            print(f"    L{l:>2}: {v:>8.0f}")


if __name__ == "__main__":
    main()
