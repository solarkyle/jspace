"""Response-type taxonomy: what does each KIND of answer look like in j-space?

For every prompt in probes/categories.json (retrieval easy/hard, misconception,
fake entity, derivation, grounded copy, invited creativity, instructed lie,
unanswerable, opinion): generate the model's short answer greedily and capture
a rich workspace snapshot at answer onset plus the next two answer tokens.

Per snapshot, per band layer we record entropy, the top-20 token ids + probs
(enough to recompute rival mass / tail smear offline), the rank of the token
the model actually generated, and, where the item carries a `truth` string
(misconceptions, instructed lies), the rank of the true answer's first token -
so we can ask whether the workspace holds the truth while the mouth lies.

Writes one JSONL row per item, checkpointing every 10. Analysis lives in
analyze_categories.py; this script only collects.

Usage:
    python probe_categories.py                     # E4B, all items
    python probe_categories.py --model google/gemma-4-12B-it --lens out/gemma-4-12b-it/lens.pt
"""

import argparse
import json
import logging
import os

if os.path.isdir("E:/hf-cache"):  # author box keeps HF cache off the full C: drive
    os.environ.setdefault("HF_HOME", "E:/hf-cache")

import numpy as np  # noqa: E402
import torch  # noqa: E402
import transformers  # noqa: E402

import jlens  # noqa: E402
from jlens.hooks import ActivationRecorder  # noqa: E402
from fit import load_model  # noqa: E402

BAND_LO_FRAC, BAND_HI_FRAC = 0.25, 0.75
TOPK = 20          # per-layer top tokens stored raw
N_SNAPSHOTS = 3    # answer onset + next two generated tokens
MAX_GEN = 16


def snapshot_from_ids(model, lens, band, input_ids):
    """Lens logits at the last position of `input_ids` for every band layer."""
    final_layer = model.n_layers - 1
    record_at = sorted({*band, final_layer})
    with ActivationRecorder(model.layers, at=record_at) as recorder:
        model.forward(input_ids)
        activations = {i: recorder.activations[i].detach() for i in record_at}

    lens_logits = {}
    for layer in band:
        residual = lens.transport(activations[layer][0, -1:].float(), layer)
        lens_logits[layer] = model.unembed(residual).float().cpu()[0]
    model_logits = model.unembed(activations[final_layer][0, -1:].float()).float().cpu()[0]
    return lens_logits, model_logits


def layer_stats(logits: torch.Tensor, gen_id: int, truth_ids: list[int],
                control_ids: list[int] | None = None) -> dict:
    probs = logits.softmax(-1)
    order = logits.argsort(descending=True)
    rank_of = torch.empty_like(order)
    rank_of[order] = torch.arange(len(order))
    top = order[:TOPK]
    top_p = probs[top]
    ent = float(-(probs * probs.clamp_min(1e-12).log()).sum())
    return {
        "entropy": round(ent, 4),
        "top_ids": [int(t) for t in top],
        "top_probs": [round(float(p), 5) for p in top_p],
        "rank_gen": int(rank_of[gen_id]),
        "rank_truth": int(min(rank_of[t] for t in truth_ids)) if truth_ids else None,
        # rank of a MISMATCHED truth (another item's answer): the null control
        # for "the workspace holds the truth while lying"
        "rank_control": int(min(rank_of[t] for t in control_ids)) if control_ids else None,
        # rival mass = candidates 2-5 (deliberation); tail = past top-20 (smear)
        "rival_mass": round(float(top_p[1:5].sum()), 5),
        "tail_mass": round(float(1.0 - top_p.sum()), 5),
    }


def first_token_ids(tokenizer, text: str) -> list[int]:
    ids = set()
    for variant in (text, " " + text, text.lower(), " " + text.lower()):
        toks = tokenizer(variant, add_special_tokens=False).input_ids
        if toks:
            ids.add(toks[0])
    return sorted(ids)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default="google/gemma-4-E4B-it")
    parser.add_argument("--lens", default="out/gemma-4-e4b-it/lens.pt")
    parser.add_argument("--probes", default="probes/categories.json")
    parser.add_argument("--out", default="")
    parser.add_argument("--limit", type=int, default=0, help="run only the first N items (smoke test)")
    parser.add_argument("--category", default="", help="run only this category")
    parser.add_argument("--quant", default="", choices=["", "4bit"],
                        help="4bit = bitsandbytes NF4 (needed for 12B on a 16GB card)")
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")

    slug = args.model.split("/")[-1].lower()
    out_path = args.out or f"out/categories_{slug}.jsonl"

    tokenizer = transformers.AutoTokenizer.from_pretrained(args.model)
    if args.quant == "4bit":
        hf_model = transformers.AutoModelForImageTextToText.from_pretrained(
            args.model,
            device_map="cuda",
            quantization_config=transformers.BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_compute_dtype=torch.bfloat16,
            ),
        )
    else:
        hf_model = load_model(args.model)
    model = jlens.from_hf(hf_model, tokenizer)
    lens = jlens.JacobianLens.load(args.lens)
    band = range(int(model.n_layers * BAND_LO_FRAC), int(model.n_layers * BAND_HI_FRAC))
    band = [l for l in band if l in lens.jacobians]

    with open(args.probes, encoding="utf-8") as f:
        items = json.load(f)["items"]
    if args.category:
        items = [it for it in items if it["category"] == args.category]
    if args.limit:
        items = items[: args.limit]

    done_ids = set()
    if os.path.exists(out_path):  # resume support
        with open(out_path, encoding="utf-8") as f:
            done_ids = {json.loads(line)["id"] for line in f if line.strip()}
        logging.info("resuming: %d items already done", len(done_ids))

    logging.info("model=%s band=L%d..L%d items=%d out=%s",
                 args.model, band[0], band[-1], len(items), out_path)

    stop_ids = {tokenizer.eos_token_id}
    eot = tokenizer.convert_tokens_to_ids("<end_of_turn>")
    if isinstance(eot, int) and eot >= 0:
        stop_ids.add(eot)

    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    out_f = open(out_path, "a", encoding="utf-8")
    n_done = 0
    for i, item in enumerate(items):
        if item["id"] in done_ids:
            continue
        prompt = tokenizer.apply_chat_template(
            [{"role": "user", "content": item["prompt"]}],
            tokenize=False, add_generation_prompt=True,
        )
        input_ids = model.encode(prompt, max_length=512)
        truth_ids = first_token_ids(tokenizer, item["truth"]) if item.get("truth") else []
        control_ids = first_token_ids(tokenizer, item["control"]) if item.get("control") else []

        # Greedy generation through the jlens forward path, snapshotting the
        # workspace at the position that PRODUCES each of the first 3 tokens.
        ids, gen_ids, step_logprobs, snapshots = input_ids, [], [], []
        for step in range(MAX_GEN):
            with torch.no_grad():
                hidden = model.forward(ids).last_hidden_state[:, -1]
                head = model._lm_head
                logits = head(hidden.to(head.weight.dtype).to(head.weight.device))
            logprobs = logits.float().log_softmax(-1)
            nxt = int(logits.argmax(-1))
            if nxt in stop_ids:
                break
            if step < N_SNAPSHOTS:
                lens_logits, _ = snapshot_from_ids(model, lens, band, ids)
                snapshots.append({
                    "step": step,
                    "token": tokenizer.decode([nxt]),
                    "layers": {str(l): layer_stats(lens_logits[l], nxt, truth_ids, control_ids)
                               for l in band},
                })
            gen_ids.append(nxt)
            step_logprobs.append(float(logprobs[0, nxt]))
            ids = torch.cat([ids, torch.tensor([[nxt]], device=ids.device)], dim=1)
        if not gen_ids:
            continue

        answer = tokenizer.decode(gen_ids, skip_special_tokens=True).strip()
        norm = lambda s: "".join(c for c in s.lower() if c.isalnum() or c == " ").strip()
        correct = None
        if item.get("aliases"):
            correct = any(norm(a) and norm(a) in norm(answer) for a in item["aliases"])

        row = {
            "id": item["id"],
            "category": item["category"],
            "subtype": item.get("subtype"),
            "q": item["prompt"],
            "answer": answer,
            "correct": correct,
            "truth": item.get("truth"),
            "bl_first_token_logprob": step_logprobs[0],
            "bl_mean_logprob": float(np.mean(step_logprobs)),
            "bl_answer_len": len(gen_ids),
            "snapshots": snapshots,
        }
        out_f.write(json.dumps(row) + "\n")
        n_done += 1
        if n_done % 10 == 0:
            out_f.flush()
            logging.info("%d new done (item %d/%d) cat=%s ans=%r",
                         n_done, i + 1, len(items), item["category"], answer[:40])
    out_f.close()
    logging.info("finished: %d new rows -> %s", n_done, out_path)


if __name__ == "__main__":
    main()
