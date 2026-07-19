"""Quantify the covert-emotion signal: an emotion x lexicon confusion matrix.

For each covert condition (fury/terror/grief/euphoria/amusement), read the lens
at the fixed-sentence output positions in the workspace band, and score how
present each emotion's lexicon is (best rank across positions x band layers).
Subtract the neutral-control score => how much covertly feeling X raises the
rank of each emotion's vocabulary.

A clean diagonal (covert-X most raises X's own lexicon) = the workspace encodes
specific emotions, not a generic "affect" blob. That's the conclusion.

Usage:
    python analysis/quant_emotions.py --model google/gemma-4-E4B-it --lens out/gemma-4-e4b-it/lens.pt
    python analysis/quant_emotions.py --model google/gemma-4-26B-A4B-it --lens out/gemma-4-26b-a4b-it/lens.pt --tag moe
"""

import argparse
import json
import os

if os.path.isdir("E:/hf-cache"):  # author box keeps HF cache off the full C: drive
    os.environ.setdefault("HF_HOME", "E:/hf-cache")

import numpy as np  # noqa: E402
import torch  # noqa: E402
import transformers  # noqa: E402

import jlens  # noqa: E402
from fit import load_model  # noqa: E402

LEXICON = {
    "fury": [" angry", " anger", " rage", " furious", " fury", " irate", " seething", " hatred", " livid"],
    "terror": [" fear", " afraid", " terror", " scared", " dread", " panic", " horror", " terrified"],
    "grief": [" sad", " grief", " sorrow", " mourning", " weep", " grieving", " heartbreak", " tears"],
    "euphoria": [" happy", " joy", " joyful", " elated", " ecstatic", " thrilled", " delighted", " euphoric"],
    "amusement": [" funny", " laugh", " hilarious", " amusing", " giggle", " humor", " comedy"],
}
CONDITIONS = ["fury", "terror", "grief", "euphoria", "amusement"]
FIXED_SENTENCE = "The meeting has been moved to noon on Thursday."


def first_ids(tokenizer, words):
    ids = set()
    for w in words:
        toks = tokenizer(w, add_special_tokens=False).input_ids
        if toks:
            ids.add(toks[0])
    return sorted(ids)


def best_band_rank(lens, model, prompt, lex_ids, band, n_sent_pos):
    """Min rank (over the last n_sent_pos positions x band layers) of any lexicon
    token. Lower = more present in the workspace."""
    positions = list(range(-n_sent_pos, 0))
    lens_logits, _, _ = lens.apply(model, prompt, positions=positions)
    best = 10**9
    lex = torch.tensor(lex_ids)
    for layer in band:
        logits = lens_logits[layer].float()  # [n_pos, vocab]
        order = logits.argsort(dim=-1, descending=True)
        rankmat = torch.empty_like(order)
        ar = torch.arange(order.shape[-1]).expand_as(order)
        rankmat.scatter_(-1, order, ar)
        best = min(best, int(rankmat[:, lex].min()))
    return best


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="google/gemma-4-E4B-it")
    ap.add_argument("--lens", default="out/gemma-4-e4b-it/lens.pt")
    ap.add_argument("--suite", default="probes/emotions.json")
    ap.add_argument("--tag", default=None, help="label for output file")
    ap.add_argument("--out", default="out/emotion_matrix.json")
    args = ap.parse_args()

    tok = transformers.AutoTokenizer.from_pretrained(args.model)
    lens = jlens.JacobianLens.load(args.lens)
    model = jlens.from_hf(load_model(args.model), tok)
    band = [l for l in range(int(model.n_layers * 0.25), int(model.n_layers * 0.75))
            if l in lens.jacobians]
    lex_ids = {emo: first_ids(tok, words) for emo, words in LEXICON.items()}
    n_sent = len(tok(FIXED_SENTENCE, add_special_tokens=False).input_ids)

    with open(args.suite, encoding="utf-8") as f:
        probes = {p["slug"]: p for p in json.load(f)}

    def resolve(slug):
        p = probes[f"covert-{slug}"] if slug != "neutral" else probes["covert-neutral"]
        msgs = [{"role": "user", "content": p["user"]},
                {"role": "assistant", "content": p["assistant_prefill"]}]
        return tok.apply_chat_template(msgs, tokenize=False, continue_final_message=True)

    # log(rank) so a jump from rank 5000->50 counts like 50->0.5
    def score(slug):
        prompt = resolve(slug)
        return {emo: np.log1p(best_band_rank(lens, model, prompt, lex_ids[emo], band, n_sent))
                for emo in LEXICON}

    neutral = score("neutral")
    print(f"model={args.model}  band=L{band[0]}..L{band[-1]}  n_sent_pos={n_sent}")
    print(f"{'condition':<12}" + "".join(f"{e[:6]:>9}" for e in LEXICON))
    matrix = {}
    for cond in CONDITIONS:
        s = score(cond)
        # delta: how much covert-cond LOWERS each lexicon's log-rank vs neutral
        delta = {emo: neutral[emo] - s[emo] for emo in LEXICON}
        matrix[cond] = delta
        row = "".join(f"{delta[e]:>+9.2f}" for e in LEXICON)
        diag = "  <-- diagonal" if max(delta, key=delta.get) == cond else ""
        print(f"{cond:<12}{row}{diag}")

    hits = sum(1 for c in CONDITIONS if max(matrix[c], key=matrix[c].get) == c)
    print(f"\ndiagonal hits: {hits}/{len(CONDITIONS)} "
          f"(covert-X most raises X's own lexicon)")

    tag = args.tag or args.model.split("/")[-1].lower()
    out = args.out.replace(".json", f"_{tag}.json")
    with open(out, "w", encoding="utf-8") as f:
        json.dump({"model": args.model, "band": [band[0], band[-1]],
                   "neutral_logrank": neutral, "delta_matrix": matrix,
                   "diagonal_hits": hits}, f, indent=2)
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
