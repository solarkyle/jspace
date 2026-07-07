"""Does the workspace know when it's about to hallucinate?

For N trivia questions: read lens features at the answer-onset position
(ignition depth, band agreement, lens entropy, hedge-token ranks), generate
the model's answer, label it right/wrong against the reference aliases, and
test whether internal state predicts correctness. Writes one JSONL row per
question + prints feature AUCs at the end.

Usage:
    python probe_uncertainty.py --n 150
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
from fit import load_model  # noqa: E402

# Workspace band: middle ~half of the network (paper reads the band, not
# single layers). For 42-layer E4B this is layers 10..31.
BAND_LO_FRAC, BAND_HI_FRAC = 0.25, 0.75

HEDGE_WORDS = [
    " guess", " maybe", " unsure", " unknown", " perhaps", " possibly",
    " unclear", " uncertain", "?", " hmm", " Hmm", " probably",
]


def load_trivia(n: int) -> list[dict]:
    from datasets import load_dataset

    ds = load_dataset("mandarjoshi/trivia_qa", "rc.nocontext", split="validation",
                      streaming=True)
    items = []
    for rec in ds:
        aliases = rec["answer"]["aliases"] + [rec["answer"]["value"]]
        items.append({"q": rec["question"], "aliases": aliases})
        if len(items) == n:
            break
    return items


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default="google/gemma-4-E4B-it")
    parser.add_argument("--lens", default="out/gemma-4-e4b-it/lens.pt")
    parser.add_argument("--n", type=int, default=150)
    parser.add_argument("--out", default="out/uncertainty.jsonl")
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")

    tokenizer = transformers.AutoTokenizer.from_pretrained(args.model)
    hf_model = load_model(args.model)
    model = jlens.from_hf(hf_model, tokenizer)
    lens = jlens.JacobianLens.load(args.lens)
    band = range(int(model.n_layers * BAND_LO_FRAC), int(model.n_layers * BAND_HI_FRAC))
    band = [l for l in band if l in lens.jacobians]

    hedge_ids = sorted({
        tid for w in HEDGE_WORDS
        for tid in tokenizer(w, add_special_tokens=False).input_ids[:1]
    })

    items = load_trivia(args.n)
    logging.info("loaded %d trivia questions; band=L%d..L%d", len(items), band[0], band[-1])

    rows = []
    for i, item in enumerate(items):
        prompt = tokenizer.apply_chat_template(
            [{"role": "user", "content":
              f"Answer with just the answer, nothing else: {item['q']}"}],
            tokenize=False, add_generation_prompt=True,
        )
        input_ids = model.encode(prompt, max_length=512)

        # Greedy generation through the jlens forward path (hf.generate routes
        # through the multimodal wrapper, which breaks on our split device map).
        # text_module output is post-final-norm, so lm_head applies directly.
        stop_ids = {tokenizer.eos_token_id}
        eot = tokenizer.convert_tokens_to_ids("<end_of_turn>")
        if isinstance(eot, int) and eot >= 0:
            stop_ids.add(eot)
        ids, gen_ids, step_logprobs = input_ids, [], []
        for _ in range(16):
            with torch.no_grad():
                hidden = model.forward(ids).last_hidden_state[:, -1]
                head = model._lm_head
                logits = head(hidden.to(head.weight.dtype).to(head.weight.device))
            logprobs = logits.float().log_softmax(-1)
            nxt = int(logits.argmax(-1))
            if nxt in stop_ids:
                break
            gen_ids.append(nxt)
            step_logprobs.append(float(logprobs[0, nxt]))
            ids = torch.cat(
                [ids, torch.tensor([[nxt]], device=ids.device)], dim=1
            )
        if not gen_ids:
            continue
        answer = tokenizer.decode(gen_ids, skip_special_tokens=True).strip()
        first_answer_id = gen_ids[0]
        # Output-confidence baselines: what the final logits alone reveal.
        # The workspace claim only stands if it beats / adds to these.
        baseline = {
            "bl_first_token_logprob": step_logprobs[0],
            "bl_mean_logprob": float(np.mean(step_logprobs)),
            "bl_min_logprob": float(np.min(step_logprobs)),
            "bl_answer_len": len(gen_ids),
        }
        norm = lambda s: "".join(c for c in s.lower() if c.isalnum() or c == " ").strip()
        correct = any(norm(a) and norm(a) in norm(answer) for a in item["aliases"])

        # Lens readouts at the final prompt position (pre-answer state).
        lens_logits, _, _ = lens.apply(model, prompt, positions=[-1])

        ranks_ans, ranks_hedge, entropies, top1s = [], [], [], []
        for layer in band:
            logits = lens_logits[layer][0].float()
            order = logits.argsort(descending=True)
            rank_of = torch.empty_like(order)
            rank_of[order] = torch.arange(len(order))
            ranks_ans.append(int(rank_of[first_answer_id]))
            ranks_hedge.append(int(min(rank_of[t] for t in hedge_ids)))
            probs = logits.softmax(-1)
            entropies.append(float(-(probs * probs.clamp_min(1e-12).log()).sum()))
            top1s.append(int(order[0]))

        ranks_arr = np.array(ranks_ans)
        ignited = np.nonzero(ranks_arr <= 10)[0]
        features = {
            # Fraction of the band where the eventual answer already leads.
            "ignition_frac": float((ranks_arr <= 10).mean()),
            # Depth (band-relative) where the answer first ignites; 1.0 = never.
            "ignition_depth": float(ignited[0] / len(band)) if len(ignited) else 1.0,
            "mean_log_rank_answer": float(np.log1p(ranks_arr).mean()),
            "band_agreement": float(np.mean(np.array(top1s) == first_answer_id)),
            "mean_entropy": float(np.mean(entropies)),
            "best_hedge_rank_log": float(np.log1p(min(ranks_hedge))),
        }
        rows.append({"q": item["q"], "answer": answer, "correct": correct,
                     **baseline, **features})
        if (i + 1) % 10 == 0:
            acc = np.mean([r["correct"] for r in rows])
            logging.info("%d/%d done, running accuracy %.2f", i + 1, len(items), acc)

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row) + "\n")

    WORKSPACE = ["ignition_frac", "ignition_depth", "mean_log_rank_answer",
                 "band_agreement", "mean_entropy", "best_hedge_rank_log"]
    BASELINE = ["bl_first_token_logprob", "bl_mean_logprob", "bl_min_logprob",
                "bl_answer_len"]

    def auc_score(x: np.ndarray, y: np.ndarray) -> float:
        order = np.argsort(x)
        ranks = np.empty(len(x)); ranks[order] = np.arange(len(x))
        return float((ranks[y].mean() - (y.sum() - 1) / 2) / (len(y) - y.sum()))

    def cv_auc(feats: list[str], rows: list[dict], y: np.ndarray, k: int = 5) -> float:
        """5-fold CV logistic regression (plain numpy GD), mean test AUC."""
        X = np.array([[r[f] for f in feats] for r in rows])
        X = (X - X.mean(0)) / (X.std(0) + 1e-9)
        rng = np.random.default_rng(0)
        idx = rng.permutation(len(y))
        folds = np.array_split(idx, k)
        aucs = []
        for i in range(k):
            test = folds[i]
            train = np.concatenate([folds[j] for j in range(k) if j != i])
            if y[test].sum() in (0, len(test)) or y[train].sum() in (0, len(train)):
                continue
            w = np.zeros(X.shape[1]); b = 0.0
            for _ in range(2000):
                p = 1 / (1 + np.exp(-(X[train] @ w + b)))
                g = p - y[train]
                w -= 0.1 * (X[train].T @ g / len(train) + 1e-3 * w)
                b -= 0.1 * g.mean()
            aucs.append(auc_score(X[test] @ w + b, y[test].astype(bool)))
        return float(np.mean(aucs))

    y = np.array([r["correct"] for r in rows])
    logging.info("accuracy: %.3f (%d/%d)", y.mean(), y.sum(), len(y))
    if 0 < y.sum() < len(y):
        yb = y.astype(bool)
        for feat in WORKSPACE + BASELINE:
            x = np.array([r[feat] for r in rows])
            auc = auc_score(x, yb)
            logging.info("AUC(%s) = %.3f  (oriented: %.3f)", feat, auc, max(auc, 1 - auc))
        logging.info("=== 5-fold CV logistic regression ===")
        logging.info("CV-AUC baseline (output confidence): %.3f", cv_auc(BASELINE, rows, y))
        logging.info("CV-AUC workspace (lens features):    %.3f", cv_auc(WORKSPACE, rows, y))
        logging.info("CV-AUC combined:                     %.3f", cv_auc(BASELINE + WORKSPACE, rows, y))
    logging.info("wrote %s", args.out)


if __name__ == "__main__":
    main()
