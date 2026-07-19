"""Causal bridge test for noisy workspace hallucinations.

This is the smallest useful intervention between the current observational
router and Anthropic-style coordinate swaps:

1. Pick confident/noisy wrong questions from a saved trace.
2. Build a paired "clean" prompt by adding the correct answer as a hint.
3. At each workspace-band layer, compute the residual delta
   `hinted_residual - original_residual` at the answer-onset position.
4. Add that delta back into the original run and measure whether the logits
   move from the model's wrong first token toward the correct answer token.

This does not prove the tail-smear directions themselves are causal; it tests
whether replacing the noisy answer-onset state with a corrected workspace state
is downstream-load-bearing. Use it as the first causal bridge, then graduate to
sparse J-space coordinate swaps/ablations.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import unicodedata
from pathlib import Path
from typing import Any

if os.path.isdir("E:/hf-cache"):
    os.environ.setdefault("HF_HOME", "E:/hf-cache")

import torch  # noqa: E402
import transformers  # noqa: E402

import jlens  # noqa: E402
from fit import load_model  # noqa: E402
from jlens.hooks import ActivationRecorder  # noqa: E402


BAND_LO, BAND_HI = 0.25, 0.75


def norm(text: str) -> str:
    text = unicodedata.normalize("NFC", text or "")
    return "".join(c for c in text.lower() if c.isalnum() or c == " ").strip()


def load_trace(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]


def load_question_aliases(path: Path | None, n: int) -> dict[str, list[str]]:
    if path is not None:
        data = json.loads(path.read_text(encoding="utf-8"))
        out: dict[str, list[str]] = {}
        for rec in data:
            aliases = rec.get("aliases") or rec.get("valid_set") or []
            if aliases:
                out[str(rec["q"])] = [str(a) for a in aliases]
        return out

    from datasets import load_dataset

    ds = load_dataset("mandarjoshi/trivia_qa", "rc.nocontext", split="validation", streaming=True)
    out = {}
    for rec in ds:
        aliases = rec["answer"]["aliases"] + [rec["answer"]["value"]]
        out[rec["question"]] = [str(a) for a in aliases if str(a).strip()]
        if len(out) >= n:
            break
    return out


def choose_targets(
    trace: list[dict[str, Any]],
    aliases_by_q: dict[str, list[str]],
    n: int,
) -> list[dict[str, Any]]:
    rows = [
        row
        for row in trace
        if not row.get("correct") and row.get("q") in aliases_by_q and aliases_by_q[row["q"]]
    ]
    rows.sort(
        key=lambda row: (
            float(row.get("bl_first_token_logprob", -999.0)),
            float(row.get("mean_entropy", row.get("ws_mean_entropy", 0.0))),
        ),
        reverse=True,
    )
    return rows[:n]


def build_prompt(tokenizer: Any, question: str, hint: str | None = None) -> str:
    content = f"Answer with just the answer, nothing else: {question}"
    if hint:
        content += f"\nHint: the answer is {hint}."
    return tokenizer.apply_chat_template(
        [{"role": "user", "content": content}],
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=False,
    )


def first_token_candidates(tokenizer: Any, aliases: list[str]) -> list[int]:
    ids: set[int] = set()
    for alias in aliases:
        for text in (alias, " " + alias):
            encoded = tokenizer(text, add_special_tokens=False).input_ids
            if encoded:
                ids.add(int(encoded[0]))
    return sorted(ids)


@torch.no_grad()
def next_logits(model: Any, input_ids: torch.Tensor) -> torch.Tensor:
    hidden = model.forward(input_ids).last_hidden_state[:, -1]
    head = model._lm_head
    logits = head(hidden.to(head.weight.dtype).to(head.weight.device))
    softcap = getattr(model, "_logit_softcap", None)
    if softcap is not None:
        logits = softcap * torch.tanh(logits / softcap)
    return logits.float()


@torch.no_grad()
def capture_residual(model: Any, input_ids: torch.Tensor, layer: int) -> torch.Tensor:
    with ActivationRecorder(model.layers, at=[layer]) as recorder:
        model.forward(input_ids)
        return recorder.activations[layer][0, -1].detach().float()


@torch.no_grad()
def patched_logits(
    model: Any,
    input_ids: torch.Tensor,
    layer: int,
    delta: torch.Tensor,
    alpha: float,
) -> torch.Tensor:
    block = model.layers[layer]

    def hook(_module, _inputs, output):
        hidden = output if torch.is_tensor(output) else output[0]
        patched = hidden.clone()
        patch = (alpha * delta).to(device=patched.device, dtype=patched.dtype)
        patched[:, -1, :] = patched[:, -1, :] + patch
        if torch.is_tensor(output):
            return patched
        return (patched, *output[1:])

    handle = block.register_forward_hook(hook)
    try:
        return next_logits(model, input_ids)
    finally:
        handle.remove()


def logit_margin(logits: torch.Tensor, target_ids: list[int], wrong_id: int) -> float:
    target = torch.tensor(target_ids, device=logits.device)
    return float(logits[0, target].max().item() - logits[0, wrong_id].item())


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default="google/gemma-4-E4B-it")
    parser.add_argument("--trace", type=Path, default=Path("data/uncertainty_trivia_gemma-4-e4b-it.jsonl"))
    parser.add_argument("--questions", type=Path, default=None, help="Optional JSON list with q + aliases/valid_set. If omitted, TriviaQA validation aliases are streamed.")
    parser.add_argument("--n", type=int, default=24)
    parser.add_argument("--alias-pool", type=int, default=500)
    parser.add_argument("--alpha", type=float, default=1.0)
    parser.add_argument("--layer-stride", type=int, default=2)
    parser.add_argument("--max-prompt-tokens", type=int, default=1024)
    parser.add_argument("--out", type=Path, default=Path("data/causal_hint_patch.jsonl"))
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
    trace = load_trace(args.trace)
    aliases_by_q = load_question_aliases(args.questions, args.alias_pool)
    targets = choose_targets(trace, aliases_by_q, args.n)
    if not targets:
        raise SystemExit("No wrong trace rows had aliases. Pass --questions with q+aliases or allow TriviaQA alias loading.")

    tokenizer = transformers.AutoTokenizer.from_pretrained(args.model)
    hf_model = load_model(args.model)
    model = jlens.from_hf(hf_model, tokenizer)
    band = list(range(int(model.n_layers * BAND_LO), int(model.n_layers * BAND_HI), args.layer_stride))
    logging.info("loaded %s; %d targets; sweeping %d layers", model, len(targets), len(band))

    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("w", encoding="utf-8") as f:
        for i, row in enumerate(targets, start=1):
            q = str(row["q"])
            aliases = aliases_by_q[q]
            target_ids = first_token_candidates(tokenizer, aliases)
            if not target_ids:
                continue
            hint = aliases[0]
            base_prompt = build_prompt(tokenizer, q)
            hint_prompt = build_prompt(tokenizer, q, hint)
            base_ids = model.encode(base_prompt, max_length=args.max_prompt_tokens)
            hint_ids = model.encode(hint_prompt, max_length=args.max_prompt_tokens)

            base_logits = next_logits(model, base_ids)
            wrong_id = int(base_logits.argmax(dim=-1).item())
            wrong_text = tokenizer.decode([wrong_id])
            before_margin = logit_margin(base_logits, target_ids, wrong_id)
            layer_results = []

            for layer in band:
                base_resid = capture_residual(model, base_ids, layer)
                hint_resid = capture_residual(model, hint_ids, layer)
                logits = patched_logits(
                    model,
                    base_ids,
                    layer,
                    hint_resid - base_resid,
                    args.alpha,
                )
                after_margin = logit_margin(logits, target_ids, wrong_id)
                top_id = int(logits.argmax(dim=-1).item())
                layer_results.append(
                    {
                        "layer": layer,
                        "margin_before": before_margin,
                        "margin_after": after_margin,
                        "margin_delta": after_margin - before_margin,
                        "top_token": tokenizer.decode([top_id]),
                        "top_is_target": top_id in target_ids,
                    }
                )

            best = max(layer_results, key=lambda rec: rec["margin_delta"])
            record = {
                "q": q,
                "target_alias": hint,
                "all_aliases": aliases,
                "original_trace_answer": row.get("answer"),
                "base_first_token": wrong_text,
                "base_first_token_id": wrong_id,
                "target_first_token_ids": target_ids,
                "alpha": args.alpha,
                "best": best,
                "layers": layer_results,
            }
            f.write(json.dumps(record) + "\n")
            logging.info(
                "%d/%d best L%d delta=%+.3f top=%r target=%s",
                i,
                len(targets),
                best["layer"],
                best["margin_delta"],
                best["top_token"],
                best["top_is_target"],
            )

    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
