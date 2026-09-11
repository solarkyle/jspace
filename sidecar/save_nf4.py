"""Save a bitsandbytes NF4 checkpoint so deployments skip the 16GB BF16 download.

Loads MODEL_ID once the usual way (BF16 weights quantized to NF4 on the fly),
then writes the already-quantized weights plus tokenizer to a directory
(~5GB for E4B). Point MODEL_PATH at that directory and the sidecar loads it
directly -- no BF16 download, no load-time quantization. The same directory
can be uploaded to a private HF model repo for Spaces deployment.

Usage: python -m sidecar.save_nf4 [--model-id ID] [--out DIR]
"""
from __future__ import annotations

import argparse
import os
from pathlib import Path

# Keep new downloads off C: when this repo is run on the author's Windows box.
if os.path.isdir("E:/hf-cache"):
    os.environ.setdefault("HF_HOME", "E:/hf-cache")

import torch
import transformers


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--model-id", default=os.environ.get("MODEL_ID", "google/gemma-4-E4B-it"))
    ap.add_argument("--out", default="", help="output dir (default: <HF_HOME or repo out/>/nf4/<slug>)")
    args = ap.parse_args()

    if not torch.cuda.is_available():
        raise SystemExit("NF4 quantization needs CUDA")

    slug = args.model_id.split("/")[-1].lower()
    if args.out:
        out = Path(args.out)
    elif os.path.isdir("E:/hf-cache"):
        out = Path("E:/hf-cache/nf4") / slug
    else:
        out = Path(__file__).resolve().parents[1] / "out" / "nf4" / slug
    out.mkdir(parents=True, exist_ok=True)

    print(f"[save_nf4] loading {args.model_id} as NF4 ...", flush=True)
    quant = transformers.BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
    )
    try:
        model = transformers.AutoModelForCausalLM.from_pretrained(
            args.model_id, quantization_config=quant, device_map="cuda"
        )
    except ValueError:
        model = transformers.AutoModelForImageTextToText.from_pretrained(
            args.model_id, quantization_config=quant, device_map="cuda"
        )
    tokenizer = transformers.AutoTokenizer.from_pretrained(args.model_id)

    print(f"[save_nf4] writing {out} ...", flush=True)
    model.save_pretrained(out, safe_serialization=True)
    tokenizer.save_pretrained(out)

    total = sum(f.stat().st_size for f in out.rglob("*") if f.is_file())
    print(f"[save_nf4] done: {out} ({total / 1024**3:.2f} GB)")
    print(f'[save_nf4] run with: MODEL_PATH="{out}" python -m uvicorn sidecar.server:app --port 8765')


if __name__ == "__main__":
    main()
