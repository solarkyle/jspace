"""Claim-A fix per peer review: is the 10-way category accuracy workspace
signal, or surface confounds?

The reviewer showed lens-free surface features (answer length, token type,
template flags) score ~65% on this task. So the honest questions are:

  1. INCREMENT: does surface+workspace beat surface alone, and by how much?
  2. REAL TRANSFER: train the workspace classifier on model A, test on
     model B (never done before; the old "transfer" was two separate CVs).
     Surface features are model-independent by construction, so the
     interesting comparison is whether the workspace increment survives
     the model swap.

Usage:
    python analysis/analyze_categories_surface.py
"""

import json
import re

import numpy as np

from analyze_categories import FEATURES, row_features

TRACES = {
    "e4b": "data/categories_gemma-4-e4b-it.jsonl",
    "12b": "data/categories_gemma-4-12b-it.jsonl",
    "moe": "data/categories_gemma-4-26b-a4b-it.jsonl",
    "qwen": "data/categories_qwen3.6-27b.jsonl",
}

YESNO = {"yes", "no"}


def surface_features(r: dict) -> list[float]:
    """Everything a category classifier could use WITHOUT the lens."""
    ans = r["answer"]
    first_word = re.split(r"\W+", ans.strip())[0].lower() if ans.strip() else ""
    q = r["q"]
    return [
        len(ans),
        float(r["bl_answer_len"]),
        float(r["bl_first_token_logprob"]),
        float(r["bl_mean_logprob"]),
        float(first_word.isdigit()),
        float(first_word in YESNO),
        len(q),
        float("Context:" in q),
        float("WRONG" in q),
        float(q.startswith("Make up") or q.startswith("Invent")),
        float("just the answer" in q),
    ]


def load(path: str):
    rows = [json.loads(l) for l in open(path, encoding="utf-8") if l.strip()]
    Xs = np.array([surface_features(r) for r in rows])
    Xw = np.array([[row_features(r)[f] for f in FEATURES] for r in rows])
    y = np.array([r["category"] for r in rows])
    return Xs, Xw, y


def cv_acc(X, y, k=5, seed=0):
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import StratifiedKFold
    from sklearn.preprocessing import StandardScaler

    correct = 0
    for tr, te in StratifiedKFold(k, shuffle=True, random_state=seed).split(X, y):
        sc = StandardScaler().fit(X[tr])
        clf = LogisticRegression(max_iter=2000).fit(sc.transform(X[tr]), y[tr])
        correct += (clf.predict(sc.transform(X[te])) == y[te]).sum()
    return correct / len(y)


def fit_full(X, y):
    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import StandardScaler

    sc = StandardScaler().fit(X)
    clf = LogisticRegression(max_iter=2000).fit(sc.transform(X), y)
    return lambda Xt: clf.predict(sc.transform(Xt))


def main() -> None:
    data = {}
    for tag, path in TRACES.items():
        try:
            data[tag] = load(path)
        except FileNotFoundError:
            print(f"({tag} trace missing, skipped)")

    print("== 1. increment over the surface baseline (5-fold CV, 10-way acc) ==")
    print(f"{'model':>6} {'surface':>8} {'workspace':>10} {'surf+ws':>8} {'increment':>10}")
    for tag, (Xs, Xw, y) in data.items():
        a_s = cv_acc(Xs, y)
        a_w = cv_acc(Xw, y)
        a_c = cv_acc(np.hstack([Xs, Xw]), y)
        print(f"{tag:>6} {a_s:>8.3f} {a_w:>10.3f} {a_c:>8.3f} {a_c - a_s:>+10.3f}")

    print("\n== 2. real cross-model transfer (train A -> test B, workspace features, per-model z) ==")
    z = lambda X: (X - X.mean(0)) / (X.std(0) + 1e-9)
    tags = list(data)
    print(f"{'':>6} " + " ".join(f"{t:>7}" for t in tags))
    for a in tags:
        Xw_a, y_a = z(data[a][1]), data[a][2]
        pred = fit_full(Xw_a, y_a)
        line = f"{a:>6} "
        for b in tags:
            if a == b:
                line += f"{'-':>7} "
                continue
            Xw_b, y_b = z(data[b][1]), data[b][2]
            acc = float((pred(Xw_b) == y_b).mean())
            line += f"{acc:>7.3f} "
        print(line)
    print("(rows = trained on, cols = tested on; chance = 0.100)")

    print("\n== 3. transfer of the INCREMENT (surface+ws vs surface, train A -> test B) ==")
    for a in tags:
        Xs_a, Xw_a, y_a = data[a]
        pred_s = fit_full(Xs_a, y_a)
        pred_c = fit_full(np.hstack([Xs_a, z(Xw_a)]), y_a)
        line = f"{a:>6} "
        for b in tags:
            if a == b:
                line += f"{'-':>15} "
                continue
            Xs_b, Xw_b, y_b = data[b]
            acc_s = float((pred_s(Xs_b) == y_b).mean())
            acc_c = float((pred_c(np.hstack([Xs_b, z(Xw_b)])) == y_b).mean())
            line += f"{acc_s:.3f}->{acc_c:.3f} "
        print(line)
    print("(surface-only -> surface+workspace, cross-model)")


if __name__ == "__main__":
    main()
