"""Fit a Jacobian lens on a local HF model.

Resumable: fitting checkpoints to <out>/ckpt.pt and picks up where it left
off if interrupted, so ctrl-c / crashes / overnight chunking are all safe.

Usage:
    python fit.py                          # Qwen3-4B-Instruct, 100 prompts
    python fit.py --n-prompts 25           # quick low-quality lens (~30 min)
    python fit.py --model Qwen/Qwen3-8B --dim-batch 4
"""

import argparse
import logging
import os

# New downloads go to E: (C: is nearly full). Existing C: cache still resolves
# for models already downloaded there. Must be set before importing transformers.
os.environ.setdefault("HF_HOME", "E:/hf-cache")

import json  # noqa: E402

import torch  # noqa: E402
import transformers  # noqa: E402

import jlens  # noqa: E402
from jlens.examples import load_wikitext_prompts  # noqa: E402


# Gemma 4 multimodal: the Jacobian backward only runs through the decoder
# layers, so everything gradient-free goes to CPU. device_map="auto" instead
# fills VRAM with the vision tower + PLE tables and pushes 5 decoder layers to
# CPU, which made backward passes ~20x slower (23+ min/prompt measured).
GEMMA4_DEVICE_MAP = {
    "lm_head": "cpu",  # tied w/ embed_tokens; fit never unembeds
    "model.language_model": 0,
    "model.language_model.embed_tokens": "cpu",  # lookup once per prompt
    "model.language_model.embed_tokens_per_layer": "cpu",  # PLE tables, ~5.6GB
    # PLE projection must sit with the PLE tables: Gemma4TextModel.forward sums
    # their outputs with bare tensor math (no module boundary for hooks to fix).
    "model.language_model.per_layer_model_projection": "cpu",
    "model.language_model.per_layer_projection_norm": "cpu",
    "model.vision_tower": "cpu",
    "model.audio_tower": "cpu",
    "model.embed_vision": "cpu",
    "model.embed_audio": "cpu",
}


def _from_pretrained(model_id: str, **kwargs):
    """AutoModelForCausalLM, falling back to the multimodal wrapper class
    (Gemma-style checkpoints reject AutoModelForCausalLM)."""
    try:
        return transformers.AutoModelForCausalLM.from_pretrained(model_id, **kwargs)
    except ValueError:
        return transformers.AutoModelForImageTextToText.from_pretrained(
            model_id, **kwargs
        )


def load_model(model_id: str, device_map=None):
    if device_map is None:
        device_map = GEMMA4_DEVICE_MAP if "gemma-4" in model_id else "auto"
    if isinstance(device_map, dict):
        # A dict with "cpu" entries makes from_pretrained OFFLOAD those modules
        # (meta tensors, crashes at forward). Load real weights into RAM, then
        # dispatch with main_device="cpu": accelerate only meta-offloads "cpu"
        # modules when main_device is a GPU (big_modeling.py:405); with "cpu"
        # they keep real weights while explicit cuda modules still run on GPU.
        from accelerate import dispatch_model

        hf_model = _from_pretrained(model_id, dtype=torch.bfloat16)
        # skip_keys: hooks must NOT deep-copy these kwargs — Gemma4 threads a
        # mutable shared_kv_states dict through the layers (KV sharing).
        return dispatch_model(
            hf_model,
            device_map=device_map,
            main_device="cpu",
            skip_keys=getattr(hf_model, "_skip_keys_device_placement", None),
        )
    return _from_pretrained(model_id, dtype=torch.bfloat16, device_map=device_map)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default="google/gemma-4-E4B-it")
    parser.add_argument("--n-prompts", type=int, default=100)
    parser.add_argument(
        "--dim-batch",
        type=int,
        default=4,
        help="Jacobian rows per backward pass; raise if VRAM allows, lower on OOM. "
        "Keep total VRAM well under 16GB or Windows' sysmem fallback thrashes.",
    )
    parser.add_argument(
        "--layer-stride",
        type=int,
        default=1,
        help="Fit every Nth layer only (2 halves fit time and lens size)",
    )
    parser.add_argument("--out", default=None, help="Output dir; default out/<model-name>")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
    logging.getLogger("jlens").setLevel(logging.DEBUG)  # per-100-pass progress
    logging.getLogger("httpx").setLevel(logging.WARNING)
    out_dir = args.out or os.path.join("out", args.model.split("/")[-1].lower())
    os.makedirs(out_dir, exist_ok=True)

    tokenizer = transformers.AutoTokenizer.from_pretrained(args.model)
    hf_model = load_model(args.model)
    model = jlens.from_hf(hf_model, tokenizer)
    logging.info("loaded %s", model)
    logging.info("device map: %s", getattr(hf_model, "hf_device_map", "n/a"))

    # Cache the corpus so resumed runs see the identical prompt list (the
    # checkpoint resumes by prompt index).
    corpus_path = os.path.join(out_dir, "corpus.json")
    if os.path.exists(corpus_path):
        with open(corpus_path, encoding="utf-8") as f:
            prompts = json.load(f)
    else:
        prompts = load_wikitext_prompts(args.n_prompts)
        with open(corpus_path, "w", encoding="utf-8") as f:
            json.dump(prompts, f)
    if len(prompts) != args.n_prompts:
        raise SystemExit(
            f"corpus.json has {len(prompts)} prompts but --n-prompts={args.n_prompts}; "
            "delete it or match the count"
        )
    logging.info("corpus: %d wikitext prompts (%s)", len(prompts), corpus_path)

    source_layers = list(range(0, model.n_layers - 1, args.layer_stride))
    lens = jlens.fit(
        model,
        prompts,
        source_layers=source_layers,
        dim_batch=args.dim_batch,
        checkpoint_path=os.path.join(out_dir, "ckpt.pt"),
        checkpoint_every=5,
    )
    lens_path = os.path.join(out_dir, "lens.pt")
    lens.save(lens_path)
    logging.info("saved %s (%.0f MB)", lens_path, os.path.getsize(lens_path) / 1e6)


if __name__ == "__main__":
    main()
