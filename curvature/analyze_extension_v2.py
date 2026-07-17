"""Extension v2: (1) pool TriviaQA + PopQA to tighten the confident-slice
estimate of whether curvature adds signal; (2) per-layer AUROC (PopQA, which
has per-layer curvature) to locate which layer, if any, carries the
confident-wrong signal.
"""

import numpy as np
import pandas as pd
from huggingface_hub import hf_hub_download
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.metrics import roc_auc_score

OUT_REPO = "solarkyle/curvature-repro-results"
BAND = ["curv_band_mean_deg", "curv_band_min_deg"]
LOGP = ["bl_first_token_logprob", "bl_mean_logprob", "bl_min_logprob"]
ENT = ["ws_mean_entropy"]


def load(ds):
    return pd.read_parquet(hf_hub_download(
        OUT_REPO, f"extension/gemma_curvature_{ds}.parquet", repo_type="dataset"))


def cv(df, cols, mask):
    X = df.loc[mask, cols].to_numpy()
    y = (~df.correct).astype(int).to_numpy()[mask.to_numpy()]
    m2 = ~np.isnan(X).any(1)
    if len(set(y[m2])) < 2 or m2.sum() < 30:
        return np.nan
    clf = make_pipeline(StandardScaler(), LogisticRegression(max_iter=1000))
    return cross_val_score(clf, X[m2], y[m2], cv=5, scoring="roc_auc").mean()


def confident_mask(df):
    return (df.bl_mean_logprob >= df.bl_mean_logprob.median()) & \
           (df.ws_mean_entropy <= df.ws_mean_entropy.median())


def main():
    dfs = {}
    for ds in ("trivia", "popqa"):
        try:
            dfs[ds] = load(ds).dropna(subset=BAND + LOGP + ENT).reset_index(drop=True)
            print(f"{ds}: n={len(dfs[ds])}, wrong={int((~dfs[ds].correct).sum())}")
        except Exception as e:
            print(f"{ds}: not available ({e})")

    # pooled band-level confident-slice test
    common = BAND + LOGP + ENT + ["correct"]
    pooled = pd.concat([d[common] for d in dfs.values()], ignore_index=True)
    print(f"\n=== POOLED ({len(pooled)} questions) ===")
    for label, m in [("all", pd.Series([True] * len(pooled))),
                     ("confident slice", confident_mask(pooled))]:
        base = cv(pooled, LOGP + ENT, m)
        plus = cv(pooled, LOGP + ENT + BAND, m)
        conly = cv(pooled, BAND, m)
        n = int(m.sum())
        wrong = 100 * (~pooled.correct[m.to_numpy()]).mean()
        print(f"  {label} (n={n}, wrong={wrong:.0f}%): logprob+entropy {base:.3f} "
              f"-> +curvature {plus:.3f} (delta {plus-base:+.3f}); curvature-only {conly:.3f}")

    # per-layer AUROC on popqa (locate the signal)
    if "popqa" in dfs and any(c.startswith("curv_L") for c in dfs["popqa"].columns):
        d = dfs["popqa"]
        y = (~d.correct).astype(int).to_numpy()
        conf = confident_mask(d).to_numpy()
        laycols = sorted([c for c in d.columns if c.startswith("curv_L")],
                         key=lambda c: int(c.split("L")[1].split("_")[0]))
        print(f"\n=== PER-LAYER AUROC for flagging WRONG (PopQA) ===")
        print(f"{'layer':>8} {'all':>7} {'confident':>10}")
        for c in laycols:
            v = d[c].to_numpy()
            m_all = ~np.isnan(v)
            a_all = roc_auc_score(y[m_all], v[m_all]) if len(set(y[m_all])) == 2 else np.nan
            mc = conf & ~np.isnan(v)
            a_c = roc_auc_score(y[mc], v[mc]) if mc.sum() > 20 and len(set(y[mc])) == 2 else np.nan
            print(f"{c.replace('curv_','').replace('_deg',''):>8} {a_all:>7.3f} {a_c:>10.3f}")


if __name__ == "__main__":
    main()
