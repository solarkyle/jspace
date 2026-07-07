"""Fit Jacobian lenses on Modal GPUs — sharded over prompts, merged at the end.

The jlens fitter averages per-prompt Jacobians, so shards fit independently on
disjoint prompt slices and JacobianLens.merge() combines them exactly.

Usage:
    modal run modal_fit.py --model google/gemma-4-E4B-it --n-prompts 8 --shards 2   # ~$1 validation
    modal run modal_fit.py --model google/gemma-4-12B-it --n-prompts 100 --shards 4
    modal run modal_fit.py --model Qwen/Qwen3.6-27B --n-prompts 100 --shards 4

Download the merged lens:
    modal volume get jlens-out <model-slug>/lens.pt out/<model-slug>/lens.pt
"""

import os

import modal

app = modal.App("jlens-fit")

# A100-80GB/H100 need a payment method on the account even with credits;
# L40S (48GB) and below run on credits alone. Override per-run:
#   JLENS_GPU=A100-80GB modal run modal_fit.py ...
GPU = os.environ.get("JLENS_GPU", "L40S")

image = (
    modal.Image.debian_slim(python_version="3.12")
    .apt_install("git")
    .pip_install(
        "torch",
        "transformers>=5.5",
        "datasets",
        "accelerate",
        "huggingface_hub",
    )
    .pip_install("git+https://github.com/anthropics/jacobian-lens")
)

hf_cache = modal.Volume.from_name("jlens-hf-cache", create_if_missing=True)
out_vol = modal.Volume.from_name("jlens-out", create_if_missing=True)


def _slug(model_id: str) -> str:
    return model_id.split("/")[-1].lower()


@app.function(
    image=image,
    gpu=GPU,
    timeout=8 * 3600,
    volumes={"/hf": hf_cache, "/out": out_vol},
    secrets=[modal.Secret.from_name("huggingface")],
)
def fit_shard(
    model_id: str, start: int, n: int, n_total: int, dim_batch: int = 16
) -> str:
    import logging
    import os

    os.environ["HF_HOME"] = "/hf"
    os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")

    import torch
    import transformers

    import jlens
    from jlens.examples import load_wikitext_prompts

    tokenizer = transformers.AutoTokenizer.from_pretrained(model_id)
    kwargs = dict(dtype=torch.bfloat16, device_map="cuda")
    try:
        hf_model = transformers.AutoModelForCausalLM.from_pretrained(
            model_id, **kwargs
        )
    except ValueError:
        hf_model = transformers.AutoModelForImageTextToText.from_pretrained(
            model_id, **kwargs
        )
    hf_cache.commit()  # persist model download for the sibling shards / next runs
    model = jlens.from_hf(hf_model, tokenizer)

    # Streaming order is deterministic, so every shard sees the same list and
    # slices its own disjoint window.
    prompts = load_wikitext_prompts(n_total)[start : start + n]

    out_dir = f"/out/{_slug(model_id)}"
    os.makedirs(out_dir, exist_ok=True)
    lens = jlens.fit(
        model,
        prompts,
        dim_batch=dim_batch,
        checkpoint_path=f"{out_dir}/ckpt-{start:04d}.pt",
        checkpoint_every=5,
    )
    path = f"{out_dir}/lens-shard-{start:04d}.pt"
    lens.save(path)
    out_vol.commit()
    return path


@app.function(image=image, timeout=1800, volumes={"/out": out_vol})
def merge_available(model_id: str) -> str:
    """Merge whatever lens-shard-*.pt exist for a model into lens.pt."""
    import glob
    import jlens

    d = f"/out/{_slug(model_id)}"
    paths = sorted(glob.glob(f"{d}/lens-shard-*.pt"))
    if not paths:
        return f"NO SHARDS for {model_id}"
    merged = jlens.JacobianLens.merge([jlens.JacobianLens.load(p) for p in paths])
    merged.save(f"{d}/lens.pt")
    out_vol.commit()
    return f"merged {len(paths)} shards -> {d}/lens.pt (n_prompts={merged.n_prompts})"


@app.local_entrypoint()
def merge(models: str):
    for m in models.split(","):
        print(merge_available.remote(m.strip()))


@app.function(image=image, timeout=1800, volumes={"/out": out_vol})
def merge_shards(model_id: str, shard_paths: list[str]) -> str:
    import jlens

    merged = jlens.JacobianLens.merge(
        [jlens.JacobianLens.load(p) for p in shard_paths]
    )
    path = f"/out/{_slug(model_id)}/lens.pt"
    merged.save(path)
    out_vol.commit()
    return path


@app.function(
    image=image, gpu="A100-80GB", timeout=3600,
    volumes={"/hf": hf_cache, "/out": out_vol},
    secrets=[modal.Secret.from_name("huggingface")],
)
def emotion_matrix(model_id: str, covert_probes: dict) -> dict:
    """Emotion x lexicon confusion matrix, run where the big model + lens live."""
    import os
    os.environ["HF_HOME"] = "/hf"
    import numpy as np, torch, transformers, jlens

    LEXICON = {
        "fury": [" angry", " anger", " rage", " furious", " fury", " irate", " seething", " hatred", " livid"],
        "terror": [" fear", " afraid", " terror", " scared", " dread", " panic", " horror", " terrified"],
        "grief": [" sad", " grief", " sorrow", " mourning", " weep", " grieving", " heartbreak", " tears"],
        "euphoria": [" happy", " joy", " joyful", " elated", " ecstatic", " thrilled", " delighted", " euphoric"],
        "amusement": [" funny", " laugh", " hilarious", " amusing", " giggle", " humor", " comedy"],
    }
    tok = transformers.AutoTokenizer.from_pretrained(model_id)
    kwargs = dict(dtype=torch.bfloat16, device_map="cuda")
    try:
        hf = transformers.AutoModelForCausalLM.from_pretrained(model_id, **kwargs)
    except ValueError:
        hf = transformers.AutoModelForImageTextToText.from_pretrained(model_id, **kwargs)
    model = jlens.from_hf(hf, tok)
    lens = jlens.JacobianLens.load(f"/out/{_slug(model_id)}/lens.pt")
    band = [l for l in range(int(model.n_layers*0.25), int(model.n_layers*0.75)) if l in lens.jacobians]
    lex = {e: sorted({tok(w, add_special_tokens=False).input_ids[0]
                      for w in ws if tok(w, add_special_tokens=False).input_ids}) for e, ws in LEXICON.items()}
    n_sent = len(tok("The meeting has been moved to noon on Thursday.", add_special_tokens=False).input_ids)

    # word -> its first-token id, for per-word evidence
    word_ids = {emo: {w: tok(w, add_special_tokens=False).input_ids[0]
                      for w in ws if tok(w, add_special_tokens=False).input_ids}
                for emo, ws in LEXICON.items()}

    def score(probe):
        msgs = [{"role": "user", "content": probe["user"]},
                {"role": "assistant", "content": probe["assistant_prefill"]}]
        prompt = tok.apply_chat_template(msgs, tokenize=False, continue_final_message=True)
        ll, _, _ = lens.apply(model, prompt, positions=list(range(-n_sent, 0)))
        # best rank per lexicon (log) + best rank per individual word (for receipts)
        rankmats = {}
        for layer in band:
            order = ll[layer].float().argsort(dim=-1, descending=True)
            rm = torch.empty_like(order)
            rm.scatter_(-1, order, torch.arange(order.shape[-1]).expand_as(order))
            rankmats[layer] = rm
        out, word_ranks = {}, {}
        for emo, ids in lex.items():
            t = torch.tensor(ids)
            out[emo] = float(np.log1p(min(int(rankmats[l][:, t].min()) for l in band)))
        for emo, wd in word_ids.items():
            for w, wid in wd.items():
                word_ranks[w.strip()] = min(int(rankmats[l][:, wid].min()) for l in band)
        return out, word_ranks

    neutral, _ = score(covert_probes["neutral"])
    matrix, hits, evidence = {}, 0, {}
    for cond in ["fury", "terror", "grief", "euphoria", "amusement"]:
        s, wr = score(covert_probes[cond])
        matrix[cond] = {e: neutral[e] - s[e] for e in LEXICON}
        if max(matrix[cond], key=matrix[cond].get) == cond:
            hits += 1
        # receipts: this emotion's own words, best-ranked first
        own_words = [w.strip() for w in LEXICON[cond]]
        own = sorted(((w, wr[w]) for w in own_words if w in wr), key=lambda x: x[1])
        evidence[cond] = own[:6]
    return {"model": model_id, "band": [band[0], band[-1]], "diagonal_hits": hits,
            "n_layers": model.n_layers, "delta_matrix": matrix, "evidence": evidence}


@app.local_entrypoint()
def emotions(models: str = "google/gemma-4-26B-A4B-it", out: str = "out/emotion_all.json"):
    import json
    probes = {p["slug"].replace("covert-", ""): p
              for p in json.load(open("probes/emotions.json", encoding="utf-8"))
              if p["slug"].startswith("covert-")}
    model_list = [m.strip() for m in models.split(",")]
    results = list(emotion_matrix.map(model_list, kwargs={"covert_probes": probes}))
    for r in results:
        print(f"\n{r['model']}  band=L{r['band'][0]}..L{r['band'][1]} of {r['n_layers']}  hits={r['diagonal_hits']}/5")
        print(f"{'cond':<10}" + "".join(f"{e[:6]:>8}" for e in r['delta_matrix']['fury']))
        for c, row in r['delta_matrix'].items():
            mark = " <" if max(row, key=row.get) == c else ""
            print(f"{c:<10}" + "".join(f"{row[e]:>+8.2f}" for e in row) + mark)
        print("  evidence:", {c: [f"{w}#{rk}" for w, rk in ev[:3]] for c, ev in r["evidence"].items()})
    with open(out, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)
    print(f"\nwrote {out}")


@app.local_entrypoint()
def main(
    model: str = "google/gemma-4-E4B-it",
    n_prompts: int = 100,
    shards: int = 4,
    dim_batch: int = 16,
):
    per = n_prompts // shards
    shard_args = []
    for i in range(shards):
        n = per + (n_prompts - per * shards if i == shards - 1 else 0)
        shard_args.append((model, i * per, n, n_prompts, dim_batch))
    print(f"fitting {model}: {shards} shards x ~{per} prompts on A100-80GB")
    paths = list(fit_shard.starmap(shard_args))
    final = merge_shards.remote(model, paths)
    print(f"merged lens: {final}")
    print(
        f"download: modal volume get jlens-out "
        f"{_slug(model)}/lens.pt out/{_slug(model)}/lens.pt"
    )
