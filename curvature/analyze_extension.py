"""Extension analysis: does contextual curvature predict hallucination
(answer incorrectness) on Gemma-4-E4B, and does it ADD signal over the logprob
baselines and the JSpace workspace features? Uses the user's existing JSpace
trivia labels joined with our Gemma curvature run.

Reports per-feature AUROC for predicting INCORRECT answers, and nested logistic
regression (5-fold CV) AUROC: logprob-only vs +curvature vs +JSpace vs all.
"""

from pathlib import Path

import numpy as np
import pandas as pd
from huggingface_hub import hf_hub_download
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.metrics import roc_auc_score

ROOT = Path(__file__).resolve().parents[1]
OUT_REPO = "solarkyle/curvature-repro-results"


def main():
    df = pd.read_parquet(hf_hub_download(
        OUT_REPO, "extension/gemma_curvature_trivia.parquet", repo_type="dataset"))
    df = df.dropna(subset=["curv_band_mean_deg", "bl_mean_logprob"]).reset_index(drop=True)
    y = (~df.correct).astype(int).to_numpy()  # predict INCORRECT (hallucination)
    print(f"n={len(df)}, incorrect={y.sum()} ({100*y.mean():.0f}%)")

    curv = ["curv_band_mean_deg", "curv_band_min_deg"]
    logp = ["bl_first_token_logprob", "bl_mean_logprob", "bl_min_logprob"]
    jspace = ["ws_mean_entropy", "ws_ignition_frac", "ws_band_agreement"]

    print("\n=== single-feature AUROC (predicting INCORRECT) ===")
    for f in curv + logp + jspace:
        v = df[f].to_numpy()
        m = ~np.isnan(v)
        if m.sum() > 20 and len(set(y[m])) == 2:
            auc = roc_auc_score(y[m], v[m])
            # report directional AUROC (max of auc, 1-auc) with sign
            print(f"  {f:>24}: AUROC {auc:.3f}  (discrimination {max(auc,1-auc):.3f})")

    def cv_auc(cols):
        X = df[cols].to_numpy()
        m = ~np.isnan(X).any(1)
        if m.sum() < 30 or len(set(y[m])) < 2:
            return float("nan")
        clf = make_pipeline(StandardScaler(), LogisticRegression(max_iter=1000))
        return cross_val_score(clf, X[m], y[m], cv=5, scoring="roc_auc").mean()

    print("\n=== nested 5-fold CV AUROC (does curvature ADD signal?) ===")
    combos = {
        "logprob only": logp,
        "logprob + curvature": logp + curv,
        "JSpace only": jspace,
        "JSpace + curvature": jspace + curv,
        "logprob + JSpace": logp + jspace,
        "logprob + JSpace + curvature (all)": logp + jspace + curv,
        "curvature only": curv,
    }
    res = {name: cv_auc(cols) for name, cols in combos.items()}
    for name, auc in res.items():
        print(f"  {name:>36}: {auc:.3f}")
    print(f"\n  curvature's marginal add over logprob: "
          f"{res['logprob + curvature'] - res['logprob only']:+.3f}")
    print(f"  curvature's marginal add over logprob+JSpace: "
          f"{res['logprob + JSpace + curvature (all)'] - res['logprob + JSpace']:+.3f}")

    outdir = ROOT / "outputs" / "extension"
    outdir.mkdir(parents=True, exist_ok=True)
    pd.DataFrame([{"model": name, "cv_auroc": auc} for name, auc in res.items()]).to_csv(
        outdir / "gemma_hallucination_auroc.csv", index=False)
    print(f"\nwrote {outdir / 'gemma_hallucination_auroc.csv'}")


if __name__ == "__main__":
    main()
