"""Generate optional expensive baseline scores for `analysis/benchmark_baselines.py`.

The default baselines are:

- `p_true`: ask the same model whether its greedy answer is True/False and use
  the false-vs-true probability as a risk score. Cost is roughly one extra
  forward pass per question.
- `sample_answer_entropy`: sample several answers and compute normalized
  lexical answer entropy. This is not full semantic entropy; it is the cheap
  sampling baseline you can run before adding an NLI/LLM clustering judge.

Output JSONL rows can be passed to:

    python analysis/benchmark_baselines.py --extra-scores data/expensive_baselines.jsonl
"""

from __future__ import annotations

import argparse
import json
import math
import os
import unicodedata
from collections import Counter
from pathlib import Path
from typing import Any

if os.path.isdir("E:/hf-cache"):
    os.environ.setdefault("HF_HOME", "E:/hf-cache")

import torch  # noqa: E402
import transformers  # noqa: E402

import jlens  # noqa: E402
from fit import load_model  # noqa: E402


def model_slug(model_id: str) -> str:
    return model_id.split("/")[-1].lower()


def norm(text: str) -> str:
    text = unicodedata.normalize("NFKD", text or "")
    text = "".join(ch for ch in text if unicodedata.category(ch) != "Mn")
    return "".join(ch for ch in text.lower() if ch.isalnum() or ch == " ").strip()


def load_rows(path: Path, n: int) -> list[dict[str, Any]]:
    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]
    return rows[:n] if n else rows


def chat_prompt(tokenizer: Any, content: str) -> str:
    return tokenizer.apply_chat_template(
        [{"role": "user", "content": content}],
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=False,
    )


@torch.no_grad()
def next_logits(model: Any, ids: torch.Tensor) -> torch.Tensor:
    hidden = model.forward(ids).last_hidden_state[:, -1]
    head = model._lm_head
    logits = head(hidden.to(head.weight.dtype).to(head.weight.device))
    softcap = getattr(model, "_logit_softcap", None)
    if softcap is not None:
        logits = softcap * torch.tanh(logits / softcap)
    return logits.float()


def stop_ids(tokenizer: Any) -> set[int]:
    out = {tid for tid in [getattr(tokenizer, "eos_token_id", None)] if isinstance(tid, int)}
    for token in ("<end_of_turn>", "<|im_end|>"):
        tid = tokenizer.convert_tokens_to_ids(token)
        if isinstance(tid, int) and tid >= 0:
            out.add(tid)
    return out


@torch.no_grad()
def generate(
    model: Any,
    tokenizer: Any,
    prompt: str,
    *,
    max_new: int,
    temperature: float,
) -> str:
    ids = model.encode(prompt, max_length=1536)
    stops = stop_ids(tokenizer)
    out: list[int] = []
    for _ in range(max_new):
        logits = next_logits(model, ids)
        if temperature <= 0:
            nxt = int(logits.argmax(dim=-1).item())
        else:
            probs = (logits / temperature).softmax(-1)
            nxt = int(torch.multinomial(probs[0], num_samples=1).item())
        if nxt in stops:
            break
        out.append(nxt)
        ids = torch.cat([ids, torch.tensor([[nxt]], device=ids.device, dtype=ids.dtype)], dim=1)
    return tokenizer.decode(out, skip_special_tokens=True).strip()


def normalized_entropy(values: list[str]) -> float:
    if not values:
        return 0.0
    counts = Counter(norm(v) for v in values)
    total = sum(counts.values())
    entropy = -sum((c / total) * math.log(max(c / total, 1e-12)) for c in counts.values())
    return entropy / math.log(max(2, len(values)))


@torch.no_grad()
def p_true_risk(model: Any, tokenizer: Any, question: str, answer: str) -> float:
    prompt = chat_prompt(
        tokenizer,
        "Question:\n"
        f"{question}\n\n"
        "Proposed answer:\n"
        f"{answer}\n\n"
        "Is the proposed answer correct? Answer exactly True or False.",
    )
    ids = model.encode(prompt, max_length=1536)
    logprobs = next_logits(model, ids).log_softmax(-1)[0]
    true_ids = first_token_ids(tokenizer, ["True", " true", "TRUE", " yes", "Yes"])
    false_ids = first_token_ids(tokenizer, ["False", " false", "FALSE", " no", "No"])
    true_logp = torch.logsumexp(logprobs[true_ids], dim=0)
    false_logp = torch.logsumexp(logprobs[false_ids], dim=0)
    pair = torch.stack([true_logp, false_logp]).softmax(0)
    return float(pair[1].item())


def first_token_ids(tokenizer: Any, texts: list[str]) -> list[int]:
    ids: set[int] = set()
    for text in texts:
        encoded = tokenizer(text, add_special_tokens=False).input_ids
        if encoded:
            ids.add(int(encoded[0]))
    if not ids:
        raise ValueError(f"tokenizer produced no IDs for {texts!r}")
    return sorted(ids)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default="google/gemma-4-12B-it")
    parser.add_argument("--trace", type=Path, default=Path("data/uncertainty_trivia_gemma-4-12b-it.jsonl"))
    parser.add_argument("--out", type=Path, default=Path("data/expensive_baselines.jsonl"))
    parser.add_argument("--n", type=int, default=100)
    parser.add_argument("--samples", type=int, default=8)
    parser.add_argument("--temperature", type=float, default=0.8)
    parser.add_argument("--max-new", type=int, default=24)
    args = parser.parse_args()

    tokenizer = transformers.AutoTokenizer.from_pretrained(args.model)
    hf_model = load_model(args.model)
    model = jlens.from_hf(hf_model, tokenizer)
    slug = model_slug(args.model)
    rows = load_rows(args.trace, args.n)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("w", encoding="utf-8") as f:
        for i, row in enumerate(rows, start=1):
            q = str(row["q"])
            greedy_answer = str(row.get("answer") or "")
            risk = p_true_risk(model, tokenizer, q, greedy_answer)
            f.write(json.dumps({
                "model": slug,
                "q": q,
                "method": "p_true",
                "score": risk,
                "cost_x": 2.0,
            }) + "\n")

            prompt = chat_prompt(tokenizer, f"Answer with just the answer, nothing else: {q}")
            samples = [
                generate(
                    model,
                    tokenizer,
                    prompt,
                    max_new=args.max_new,
                    temperature=args.temperature,
                )
                for _ in range(args.samples)
            ]
            f.write(json.dumps({
                "model": slug,
                "q": q,
                "method": "sample_answer_entropy",
                "score": normalized_entropy(samples),
                "cost_x": float(args.samples),
                "samples": samples,
            }) + "\n")
            print(f"{i}/{len(rows)} {q[:70]!r} p_true_risk={risk:.3f}")

    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
