from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np


def probabilities(logits: np.ndarray) -> np.ndarray:
    shifted = logits.astype(np.float64) - float(np.max(logits))
    values = np.exp(shifted)
    return values / values.sum()


def top_row(logits: np.ndarray, top_k: int = 32) -> tuple[list[int], list[float]]:
    probs = probabilities(logits)
    take = min(int(top_k), probs.shape[-1])
    ids = np.argpartition(-probs, take - 1)[:take]
    ids = ids[np.argsort(-probs[ids])]
    return [int(value) for value in ids], [round(float(probs[value]), 8) for value in ids]


def export_capture(
    reference: dict[str, Any],
    *,
    native_url: str,
    model_gguf: str,
    lens_gguf: str,
) -> dict[str, Any]:
    try:
        from jlens_gguf.client import NativeClient
        from jlens_gguf.lens import JacobianLensGGUF
        from jlens_gguf.model_reader import ReadoutWeights
        from jlens_gguf.readout import LensReadout
    except ImportError as exc:
        raise SystemExit(
            "jlens-gguf is required. Install it with `pip install -e /path/to/jlens-gguf`."
        ) from exc

    token_ids = [int(value) for value in reference.get("input_token_ids", [])]
    if not token_ids:
        raise ValueError("reference capture has no input_token_ids; recapture it with J-Space")
    wanted_layers = [int(row["layer"]) for row in reference.get("layers", [])]
    if not wanted_layers:
        raise ValueError("reference capture has no layers")

    lens = JacobianLensGGUF.load(lens_gguf)
    missing = sorted(set(wanted_layers) - set(lens.source_layers))
    if missing:
        raise ValueError(f"GGUF lens is missing reference layers: {missing}")
    weights = ReadoutWeights.from_gguf(model_gguf)
    readout = LensReadout(weights, lens)
    client = NativeClient(native_url)
    forward = client.forward(
        token_ids,
        capture_layers=wanted_layers,
        dtype="f32",
        logits_positions=[-1],
    )

    rows: list[dict[str, Any]] = []
    for layer in wanted_layers:
        residual = forward.activations[layer][-1].astype(np.float32)
        logits = readout.lens_logits(residual[None, :], layer, use_lens=True)[0]
        ids, probs = top_row(logits)
        rows.append(
            {
                "layer": layer,
                "residual": [round(float(value), 7) for value in residual.tolist()],
                "residual_norm": round(float(np.linalg.norm(residual)), 7),
                "top_ids": ids,
                "top_probs": probs,
            }
        )

    if not forward.logits:
        raise RuntimeError("jlens-server returned no final logits")
    final_logits = forward.logits[max(forward.logits)]
    final_ids, final_probs = top_row(final_logits)
    props = client.props()
    vocab = client.vocab()
    return {
        "schema_version": 1,
        "capture_type": "jspace_conformance",
        "backend": "llama.cpp",
        "model": props.get("model_name") or Path(model_gguf).name,
        "quant": "gguf",
        "prompt_tokens": len(token_ids),
        "input_token_ids": token_ids,
        "residual_point": "post-block, final prompt position",
        "lens": {
            "path": str(lens_gguf),
            "fit_method": lens.fit_method,
            "n_prompts": lens.n_prompts,
        },
        "layers": rows,
        "final": {
            "next_token_id": final_ids[0],
            "next_token": vocab[final_ids[0]],
            "top_ids": final_ids,
            "top_probs": final_probs,
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Export a jlens-gguf capture in J-Space conformance format")
    parser.add_argument("--reference", required=True, help="HF reference JSON exported by /lab")
    parser.add_argument("--native-url", default="http://127.0.0.1:8091")
    parser.add_argument("--model-gguf", required=True)
    parser.add_argument("--lens-gguf", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    reference = json.loads(Path(args.reference).read_text(encoding="utf-8"))
    capture = export_capture(
        reference,
        native_url=args.native_url,
        model_gguf=args.model_gguf,
        lens_gguf=args.lens_gguf,
    )
    Path(args.output).write_text(json.dumps(capture, indent=2), encoding="utf-8")
    print(f"wrote {args.output}: {len(capture['layers'])} aligned layers")


if __name__ == "__main__":
    main()
