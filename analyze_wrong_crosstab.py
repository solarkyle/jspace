"""Cross-tab: wrong-answer KIND (substitution/fabrication/label-noise) vs
workspace state (clean/noisy, trajectory mode).

The hypothesis on the table: clean-wrong = substitution, noisy-wrong =
fabrication. If that holds, the noise detector's blind spot is not a bug in
the features, it is the nature of substitution errors: they ARE clean
retrieval, of the wrong fact.

Usage:
    python analyze_wrong_crosstab.py [--file scratch/wrong_classified_e4b.json]
"""

import argparse
import json
from collections import Counter, defaultdict

import numpy as np


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--file", default="scratch/wrong_classified_e4b.json")
    args = ap.parse_args()

    items = json.load(open(args.file, encoding="utf-8"))
    items = [it for it in items if it.get("cls")]
    print(f"{len(items)} classified wrong answers ({args.file})")
    print("class counts:", dict(Counter(it["cls"] for it in items)))

    # after removing label noise, the honest error set:
    real_wrong = [it for it in items if it["cls"] != "ACTUALLY_RIGHT"]
    print(f"\nlabel noise (ACTUALLY_RIGHT): "
          f"{len(items) - len(real_wrong)}/{len(items)} "
          f"({(len(items) - len(real_wrong)) / len(items):.0%}) of 'wrong' answers")

    print("\n== workspace state x error kind (row %) ==")
    table = defaultdict(Counter)
    for it in real_wrong:
        state = "clean-wrong" if it["clean_wrong"] else "noisy-wrong"
        table[state][it["cls"]] += 1
    classes = ["SUBSTITUTION", "FABRICATION"]
    print(f"{'':>12} " + " ".join(f"{c:>13}" for c in classes) + f" {'n':>5}")
    for state in ["clean-wrong", "noisy-wrong"]:
        n = sum(table[state].values())
        row = " ".join(f"{table[state][c]:>6} ({table[state][c]/max(n,1):>4.0%})"
                       for c in classes)
        print(f"{state:>12} {row} {n:>5}")

    print("\n== trajectory mode x error kind ==")
    tmode = defaultdict(Counter)
    for it in real_wrong:
        tmode[it.get("mode") or "?"][it["cls"]] += 1
    for mode in sorted(tmode):
        n = sum(tmode[mode].values())
        row = " ".join(f"{tmode[mode][c]:>6} ({tmode[mode][c]/max(n,1):>4.0%})"
                       for c in classes)
        print(f"{mode:>12} {row} {n:>5}")

    print("\n== mean output confidence by kind ==")
    for c in classes + ["ACTUALLY_RIGHT"]:
        confs = [it["conf"] for it in items if it["cls"] == c]
        if confs:
            print(f"  {c:>14}: {np.mean(confs):+.3f} (n={len(confs)})")

    # the detector's honest scorecard on REAL errors
    subs = [it for it in real_wrong if it["cls"] == "SUBSTITUTION"]
    fabs = [it for it in real_wrong if it["cls"] == "FABRICATION"]
    if subs and fabs:
        catch = lambda xs: sum(1 for it in xs if not it["clean_wrong"]) / len(xs)
        print(f"\n== detector scorecard (noisy = flagged) ==")
        print(f"  fabrications flagged: {catch(fabs):.0%} (n={len(fabs)})")
        print(f"  substitutions flagged: {catch(subs):.0%} (n={len(subs)})")


if __name__ == "__main__":
    main()
