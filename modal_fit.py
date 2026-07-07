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
def merge_shards(model_id: str, shard_paths: list[str]) -> str:
    import jlens

    merged = jlens.JacobianLens.merge(
        [jlens.JacobianLens.load(p) for p in shard_paths]
    )
    path = f"/out/{_slug(model_id)}/lens.pt"
    merged.save(path)
    out_vol.commit()
    return path


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
