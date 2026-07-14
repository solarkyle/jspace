"""Reproduce the Gate D table in 90 seconds, CPU only, no GPU, no repo installs.

Requirements (installed by you, this script installs nothing):

    pip install lightgbm numpy huggingface_hub

What it does:

1. Downloads the Stage 2 feature table (stage2_features.jsonl) from the
   public HF dataset solarkyle/jspace-hallucination-campaign.
2. Loads the frozen Stage 1 LightGBM classifiers committed in this repo
   (campaign/frozen/), verifying their SHA-256 hashes against the values
   recorded in campaign/PREREG_STAGE2.md. A hash mismatch aborts: the whole
   point is that these artifacts were frozen before Stage 2 data existed.
3. Scores every labeled Stage 2 row zero-shot with all three frozen models
   (combined / logprob / workspace) and recomputes the per-source AUROC
   table and the Gate D verdict.
4. Prints the recomputed numbers next to the published ones
   (campaign/reports/STAGE2_REPORT.md) with a PASS/FAIL match column
   (tolerance 0.002).

Run from the repo root:

    python campaign/reproduce_mini.py
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
FROZEN_DIR = os.path.join(HERE, "frozen")
PREREG = os.path.join(HERE, "PREREG_STAGE2.md")

HF_DATASET = "solarkyle/jspace-hallucination-campaign"
FEATURES_FILE = "stage2_features.jsonl"

NEW_SOURCES = ("truthfulqa", "nq_open", "facts_grounding",
               "legal_hallucinations", "bfcl")
SEPARATE = ("squad_v2",)
FAMILIES = ("combined", "logprob", "workspace")
TOL = 0.002

# Published numbers: campaign/reports/STAGE2_REPORT.md (final, 2026-07-13).
PUBLISHED = {
    "truthfulqa":           {"lp": 0.620, "ws": 0.592, "comb": 0.581, "inc": -0.038},
    "nq_open":              {"lp": 0.746, "ws": 0.739, "comb": 0.768, "inc": +0.023},
    "facts_grounding":      {"lp": 0.605, "ws": 0.633, "comb": 0.668, "inc": +0.063},
    "legal_hallucinations": {"lp": 0.555, "ws": 0.419, "comb": 0.448, "inc": -0.107},
    "bfcl":                 {"lp": 0.572, "ws": 0.499, "comb": 0.552, "inc": -0.020},
    "squad_v2":             {"lp": 0.363, "ws": 0.425, "comb": 0.419, "inc": +0.056},
}
PUBLISHED_MEAN = -0.016
PUBLISHED_BREADTH = 2
PUBLISHED_VERDICT = "MISS"


def prereg_hashes():
    """The three frozen-model SHA-256 hashes recorded in PREREG_STAGE2.md."""
    text = open(PREREG, encoding="utf-8").read()
    out = {}
    for fam in FAMILIES:
        m = re.search(fam + r"\s*\(\d+ features\):\s*([0-9a-f]{64})", text)
        if not m:
            sys.exit(f"FATAL: no {fam} hash found in {PREREG}")
        out[fam] = m.group(1)
    return out


def load_frozen(family, expected_sha):
    import lightgbm as lgb
    path = os.path.join(FROZEN_DIR, f"lgbm_stage1_{family}.txt")
    sha = hashlib.sha256(open(path, "rb").read()).hexdigest()
    if sha != expected_sha:
        sys.exit(f"FATAL: {path} sha256 {sha} does not match the hash "
                 f"pre-registered in PREREG_STAGE2.md ({expected_sha}). "
                 f"The frozen artifact was modified; refusing to score.")
    meta = json.load(open(os.path.join(FROZEN_DIR, "frozen_meta.json"),
                          encoding="utf-8"))["families"][family]
    if meta["model_sha256"] != expected_sha:
        sys.exit(f"FATAL: frozen_meta.json hash for {family} disagrees with "
                 f"PREREG_STAGE2.md")
    print(f"  frozen [{family:>9}] sha256 OK ({sha[:16]}...)")
    return lgb.Booster(model_file=path), meta


def auroc(y, s):
    """Rank-based AUROC with tie handling (Mann-Whitney), no sklearn needed."""
    y = np.asarray(y, dtype=float)
    s = np.asarray(s, dtype=float)
    order = np.argsort(s)
    ranks = np.empty(len(s))
    ranks[order] = np.arange(1, len(s) + 1)
    # average ranks over ties
    sorted_s = s[order]
    i = 0
    while i < len(s):
        j = i
        while j + 1 < len(s) and sorted_s[j + 1] == sorted_s[i]:
            j += 1
        if j > i:
            ranks[order[i:j + 1]] = (i + j) / 2 + 1
        i = j + 1
    n_pos = y.sum()
    n_neg = len(y) - n_pos
    if n_pos == 0 or n_neg == 0:
        return float("nan")
    return float((ranks[y == 1].sum() - n_pos * (n_pos + 1) / 2)
                 / (n_pos * n_neg))


def main():
    print("== jspace Gate D mini-reproduction (CPU only) ==\n")

    print("[1/3] verifying frozen classifiers against PREREG_STAGE2.md hashes")
    hashes = prereg_hashes()
    models = {fam: load_frozen(fam, hashes[fam]) for fam in FAMILIES}

    print(f"\n[2/3] downloading {FEATURES_FILE} from hf.co/datasets/{HF_DATASET}")
    from huggingface_hub import hf_hub_download
    path = hf_hub_download(HF_DATASET, FEATURES_FILE, repo_type="dataset")
    rows = [json.loads(l) for l in open(path, encoding="utf-8") if l.strip()]
    labeled = [r for r in rows if "label" in r]
    print(f"  {len(labeled)} labeled of {len(rows)} rows"
          + ("" if len(labeled) == len(rows) else
             " (unlabeled rows are ambiguous-judge exclusions; scoring "
             "labeled rows only, same as the published table)"))

    print("\n[3/3] scoring zero-shot with the frozen models")
    scores = {}
    for fam, (booster, meta) in models.items():
        names = meta["feature_names"]
        missing = [n for n in names if n not in labeled[0]["features"]]
        if missing:
            sys.exit(f"FATAL: rows lack frozen features: {missing}")
        X = np.array([[r["features"][n] for n in names] for r in labeled])
        Xs = (X - np.array(meta["scaler_mean"])) / np.array(meta["scaler_scale"])
        scores[fam] = booster.predict(Xs)
    y = np.array([r["label"] for r in labeled])
    src = np.array([r["source_dataset"] for r in labeled])

    print(f"\n{'source':>21} {'n':>5} {'AUC_lp':>7} {'AUC_ws':>7} {'AUC_comb':>8} "
          f"{'increment':>9} {'published':>9} {'match':>6}")
    increments, all_pass = [], True
    for s in NEW_SOURCES + SEPARATE:
        m = src == s
        if m.sum() == 0:
            print(f"{s:>21}  (no rows on HF; cannot check this source)")
            all_pass = False
            continue
        a = {f: auroc(y[m], scores[f][m]) for f in FAMILIES}
        inc = a["combined"] - a["logprob"]
        if s in NEW_SOURCES:
            increments.append(inc)
        pub = PUBLISHED[s]
        ok = (abs(a["logprob"] - pub["lp"]) <= TOL
              and abs(a["workspace"] - pub["ws"]) <= TOL
              and abs(a["combined"] - pub["comb"]) <= TOL
              and abs(inc - pub["inc"]) <= TOL)
        all_pass &= ok
        print(f"{s:>21} {m.sum():>5} {a['logprob']:>7.3f} {a['workspace']:>7.3f} "
              f"{a['combined']:>8.3f} {inc:>+9.4f} {pub['inc']:>+9.3f} "
              f"{'PASS' if ok else 'FAIL':>6}")

    mean_inc = float(np.mean(increments)) if increments else float("nan")
    breadth = sum(1 for i in increments if i > 0)
    verdict = ("HIT" if mean_inc >= 0.02 and breadth >= 4
               and len(increments) >= 4 else "MISS")
    mean_ok = abs(mean_inc - PUBLISHED_MEAN) <= TOL
    verdict_ok = (verdict == PUBLISHED_VERDICT and breadth == PUBLISHED_BREADTH)
    all_pass &= mean_ok and verdict_ok

    print(f"\nGate D mean increment over {len(increments)} new sources: "
          f"{mean_inc:+.4f} (published {PUBLISHED_MEAN:+.3f}) "
          f"{'PASS' if mean_ok else 'FAIL'}")
    print(f"breadth positive: {breadth}/5 (published {PUBLISHED_BREADTH}/5), "
          f"verdict: {verdict} (published {PUBLISHED_VERDICT}) "
          f"{'PASS' if verdict_ok else 'FAIL'}")
    print(f"\n== OVERALL: {'PASS' if all_pass else 'FAIL'} "
          f"(tolerance {TOL} per cell) ==")
    sys.exit(0 if all_pass else 1)


if __name__ == "__main__":
    main()
