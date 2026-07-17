# /// script
# requires-python = ">=3.11"
# dependencies = ["torch", "transformers", "numpy", "pandas", "pyarrow", "huggingface_hub", "accelerate", "scikit-learn"]
# ///
"""Extension: contextual curvature on Gemma-4-E4B for hallucination prediction,
joined to the user's existing JSpace trivia labels + features.

For each JSpace trivia question, build the same prompt, run Gemma, and read
contextual curvature at the answer-onset position (final prompt token) across
a middle-layer band. Output per-question curvature features joined with the
JSpace row (correct label, logprob baselines, JSpace workspace features), so we
can compare curvature vs logprob vs JSpace for predicting correctness.

Env: N (default 500), MODEL_ID (google/gemma-4-E4B-it), OUT_REPO, BAND (0.25,0.75).
Uploads extension/gemma_curvature_trivia.parquet.
"""

import json
import math
import os
import urllib.request

import numpy as np
import torch
from huggingface_hub import HfApi
from transformers import AutoModelForCausalLM, AutoTokenizer

MODEL_ID = os.environ.get("MODEL_ID", "google/gemma-4-E4B-it")
N = int(os.environ.get("N_Q", "500"))
OUT_REPO = os.environ.get("OUT_REPO", "solarkyle/curvature-repro-results")
BAND_LO, BAND_HI = 0.25, 0.75
JSPACE_URL = "https://raw.githubusercontent.com/solarkyle/jspace/master/data/uncertainty_trivia_gemma-4-e4b-it.jsonl"
NORM_RTOL, COS_EPS = 1e-9, 1e-7


def turn_angles(x):
    x = np.asarray(x, np.float64); T = x.shape[0]
    c = np.full(T, np.nan)
    if T < 3: return c
    v = x[1:] - x[:-1]; a, b = v[:-1], v[1:]
    na, nb = np.linalg.norm(a, axis=1), np.linalg.norm(b, axis=1)
    floor = NORM_RTOL * max(np.nanmean(np.linalg.norm(v, axis=1)), 1e-300)
    ok = (na > floor) & (nb > floor)
    cos = np.full(T - 2, np.nan)
    cos[ok] = np.einsum("ij,ij->i", a[ok], b[ok]) / (na[ok] * nb[ok])
    c[:T - 2] = np.arccos(np.clip(cos, -1 + COS_EPS, 1 - COS_EPS))
    return c


def contextual_curvature_last(x):
    """C_k at the final valid position k = last prompt token (window k-4..k-2)."""
    c = turn_angles(x)
    valid = np.flatnonzero(~np.isnan(c))
    if len(valid) < 3:
        return np.nan
    k = valid[-1] + 2  # last angle is c[k-2]; align to token index
    idx = [k - 4, k - 3, k - 2]
    if idx[0] < 0 or np.any(np.isnan(c[idx])):
        # fall back to mean of last 3 valid angles
        return float(np.nanmean(c[valid[-3:]]))
    return float(np.nanmean(c[idx]))


def get_decoder_layers(model):
    # Gemma 4 may nest the text model under language_model / model
    for path in ("model.layers", "language_model.model.layers",
                 "model.language_model.layers", "language_model.layers"):
        obj = model
        try:
            for p in path.split("."):
                obj = getattr(obj, p)
            if hasattr(obj, "__len__") and len(obj) > 0:
                print(f"decoder layers at: {path} ({len(obj)})")
                return obj
        except AttributeError:
            continue
    raise ValueError("could not locate decoder layers")


def main():
    device = "cuda"
    data = urllib.request.urlopen(JSPACE_URL).read().decode()
    rows_js = [json.loads(l) for l in data.strip().split("\n") if l][:N]
    print(f"{len(rows_js)} trivia questions")

    tok = AutoTokenizer.from_pretrained(MODEL_ID)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID, dtype=torch.bfloat16, device_map="auto").eval()
    layers = get_decoder_layers(model)
    n_layers = len(layers)
    band = list(range(int(n_layers * BAND_LO), int(n_layers * BAND_HI)))
    print(f"{n_layers} layers, band L{band[0]}-L{band[-1]}")

    cap = {}
    for li in band:
        layers[li].register_forward_hook(
            lambda m, i, o, li=li: cap.__setitem__(li, (o[0] if isinstance(o, tuple) else o).detach()))

    out_rows = []
    for qi, r in enumerate(rows_js):
        prompt = tok.apply_chat_template(
            [{"role": "user", "content": f"Answer with just the answer, nothing else: {r['q']}"}],
            tokenize=False, add_generation_prompt=True)
        ids = tok(prompt, return_tensors="pt", add_special_tokens=False).input_ids.to(device)
        if ids.shape[1] < 8:
            continue
        cap.clear()
        with torch.no_grad():
            model(ids)
        curvs = [contextual_curvature_last(cap[li][0].float().cpu().numpy()) for li in band]
        rec = {"q": r["q"], "correct": str(r["correct"]).lower() == "true",
               "curv_band_mean_deg": float(np.degrees(np.nanmean(curvs))),
               "curv_band_min_deg": float(np.degrees(np.nanmin(curvs))),
               "bl_first_token_logprob": float(r["bl_first_token_logprob"]),
               "bl_mean_logprob": float(r["bl_mean_logprob"]),
               "bl_min_logprob": float(r["bl_min_logprob"]),
               "ws_mean_entropy": float(r.get("mean_entropy", "nan")),
               "ws_ignition_frac": float(r.get("ignition_frac", "nan")),
               "ws_band_agreement": float(r.get("band_agreement", "nan"))}
        out_rows.append(rec)
        if (qi + 1) % 50 == 0:
            print(f"  {qi+1}/{len(rows_js)}")

    import pandas as pd
    df = pd.DataFrame(out_rows)
    df.to_parquet("/tmp/gemma_curv.parquet")
    HfApi().upload_file(path_or_fileobj="/tmp/gemma_curv.parquet",
                        path_in_repo="extension/gemma_curvature_trivia.parquet",
                        repo_id=OUT_REPO, repo_type="dataset")
    # quick in-job AUROC sanity
    try:
        from sklearn.metrics import roc_auc_score
        y = df.correct.to_numpy()  # True = correct answer
        for feat in ("curv_band_mean_deg", "bl_mean_logprob", "ws_mean_entropy"):
            v = df[feat].to_numpy()
            m = ~np.isnan(v)
            if m.sum() > 10 and len(set(y[m])) == 2:
                auc = roc_auc_score(y[m], v[m])
                print(f"AUROC(correct ~ {feat}) = {auc:.3f} (|0.5-auc|={abs(0.5-auc):.3f})")
    except Exception as e:
        print("auroc skip:", e)
    print(f"n={len(df)} wrong={int((~df.correct).sum())} DONE")


if __name__ == "__main__":
    main()
