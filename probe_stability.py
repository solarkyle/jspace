"""Does single-pass workspace noise predict answer INSTABILITY under resampling?

Hypothesis: the onset noise score measures how fuzzy the model's answer
distribution is. Clean-wrong answers are stably-believed wrong facts (same
answer at every temperature-1 resample); noisy-wrong answers flip entities
across resamples. If onset noise correlates with resample cluster entropy,
a 1-pass internal read predicts what semantic entropy needs 5-10 full
generations to measure.

Protocol: take questions from the committed E4B TriviaQA trace (stratified:
clean-wrong / noisy-wrong / correct), re-ask each with the EXACT original
prompt, sample K answers at temperature 1.0, cluster by normalized string,
report cluster entropy + agreement with the original greedy answer.

Usage:
    python probe_stability.py --per_group 40 --k 6
"""

import argparse
import json
import logging
import os
from collections import Counter

if os.path.isdir("E:/hf-cache"):
    os.environ.setdefault("HF_HOME", "E:/hf-cache")

import numpy as np  # noqa: E402
import torch  # noqa: E402
import transformers  # noqa: E402

import jlens  # noqa: E402
from fit import load_model  # noqa: E402


def norm(s: str) -> str:
    s = s.split("thought")[0]
    return "".join(c for c in s.lower() if c.isalnum() or c == " ").strip()


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--model", default="google/gemma-4-E4B-it")
    ap.add_argument("--trace", default="data/uncertainty_trivia_gemma-4-e4b-it.jsonl")
    ap.add_argument("--per_group", type=int, default=40)
    ap.add_argument("--k", type=int, default=6)
    ap.add_argument("--max_new", type=int, default=12)
    ap.add_argument("--out", default="out/stability_e4b.jsonl")
    args = ap.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")

    rows = [json.loads(l) for l in open(args.trace, encoding="utf-8") if l.strip()]
    wrong = [r for r in rows if not r["correct"]]
    right = [r for r in rows if r["correct"]]
    ent_w = np.array([r["mean_entropy"] for r in wrong])
    med = np.median(ent_w)
    clean_wrong = [r for r, e in zip(wrong, ent_w) if e <= med]
    noisy_wrong = [r for r, e in zip(wrong, ent_w) if e > med]
    rng = np.random.default_rng(0)
    pick = lambda xs: [xs[i] for i in rng.choice(len(xs), min(args.per_group, len(xs)), replace=False)]
    selected = ([("clean_wrong", r) for r in pick(clean_wrong)]
                + [("noisy_wrong", r) for r in pick(noisy_wrong)]
                + [("correct", r) for r in pick(right)])
    logging.info("selected %d questions (%d per group)", len(selected), args.per_group)

    tokenizer = transformers.AutoTokenizer.from_pretrained(args.model)
    hf_model = load_model(args.model)
    model = jlens.from_hf(hf_model, tokenizer)

    stop_ids = {tokenizer.eos_token_id}
    eot = tokenizer.convert_tokens_to_ids("<end_of_turn>")
    if isinstance(eot, int) and eot >= 0:
        stop_ids.add(eot)

    done = set()
    if os.path.exists(args.out):
        with open(args.out, encoding="utf-8") as f:
            done = {json.loads(l)["q"] for l in f if l.strip()}
        logging.info("resuming: %d already done", len(done))

    out_f = open(args.out, "a", encoding="utf-8")
    for i, (group, r) in enumerate(selected):
        if r["q"] in done:
            continue
        prompt = tokenizer.apply_chat_template(
            [{"role": "user", "content":
              f"Answer with just the answer, nothing else: {r['q']}"}],
            tokenize=False, add_generation_prompt=True,
        )
        input_ids = model.encode(prompt, max_length=512)
        samples = []
        for _ in range(args.k):
            ids, gen_ids = input_ids, []
            for _ in range(args.max_new):
                with torch.no_grad():
                    hidden = model.forward(ids).last_hidden_state[:, -1]
                    head = model._lm_head
                    logits = head(hidden.to(head.weight.dtype).to(head.weight.device))
                probs = (logits.float() / 1.0).softmax(-1)
                nxt = int(torch.multinomial(probs[0], 1))
                if nxt in stop_ids:
                    break
                gen_ids.append(nxt)
                ids = torch.cat([ids, torch.tensor([[nxt]], device=ids.device)], dim=1)
            samples.append(tokenizer.decode(gen_ids, skip_special_tokens=True).strip())

        clusters = Counter(norm(s) for s in samples if norm(s))
        total = sum(clusters.values())
        probs_c = np.array([v / total for v in clusters.values()]) if total else np.array([1.0])
        cluster_entropy = float(-(probs_c * np.log(probs_c + 1e-12)).sum())
        modal_answer, modal_n = (clusters.most_common(1)[0] if clusters else ("", 0))
        out_f.write(json.dumps({
            "q": r["q"], "group": group, "correct": r["correct"],
            "orig_answer": r["answer"], "mean_entropy": r["mean_entropy"],
            "bl_first_token_logprob": r["bl_first_token_logprob"],
            "ignition_frac": r.get("ignition_frac"),
            "samples": samples,
            "n_distinct": len(clusters),
            "cluster_entropy": cluster_entropy,
            "modal_frac": modal_n / max(total, 1),
            "modal_matches_orig": norm(r["answer"]) == modal_answer,
        }) + "\n")
        if (i + 1) % 10 == 0:
            out_f.flush()
            logging.info("%d/%d done", i + 1, len(selected))
    out_f.close()
    logging.info("done -> %s", args.out)


if __name__ == "__main__":
    main()
