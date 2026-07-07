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
def merge_available(model_id: str) -> str:
    """Merge whatever lens-shard-*.pt exist for a model into lens.pt."""
    import glob
    import jlens

    d = f"/out/{_slug(model_id)}"
    paths = sorted(glob.glob(f"{d}/lens-shard-*.pt"))
    if not paths:
        return f"NO SHARDS for {model_id}"
    merged = jlens.JacobianLens.merge([jlens.JacobianLens.load(p) for p in paths])
    merged.save(f"{d}/lens.pt")
    out_vol.commit()
    return f"merged {len(paths)} shards -> {d}/lens.pt (n_prompts={merged.n_prompts})"


@app.local_entrypoint()
def merge(models: str):
    for m in models.split(","):
        print(merge_available.remote(m.strip()))


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


@app.function(
    image=image, gpu="A100-80GB", timeout=3600,
    volumes={"/hf": hf_cache, "/out": out_vol},
    secrets=[modal.Secret.from_name("huggingface")],
)
def emotion_matrix(model_id: str, covert_probes: dict) -> dict:
    """Emotion x lexicon confusion matrix, run where the big model + lens live."""
    import os
    os.environ["HF_HOME"] = "/hf"
    import numpy as np, torch, transformers, jlens

    LEXICON = {
        "fury": [" angry", " anger", " rage", " furious", " fury", " irate", " seething", " hatred", " livid"],
        "terror": [" fear", " afraid", " terror", " scared", " dread", " panic", " horror", " terrified"],
        "grief": [" sad", " grief", " sorrow", " mourning", " weep", " grieving", " heartbreak", " tears"],
        "euphoria": [" happy", " joy", " joyful", " elated", " ecstatic", " thrilled", " delighted", " euphoric"],
        "amusement": [" funny", " laugh", " hilarious", " amusing", " giggle", " humor", " comedy"],
    }
    tok = transformers.AutoTokenizer.from_pretrained(model_id)
    kwargs = dict(dtype=torch.bfloat16, device_map="cuda")
    try:
        hf = transformers.AutoModelForCausalLM.from_pretrained(model_id, **kwargs)
    except ValueError:
        hf = transformers.AutoModelForImageTextToText.from_pretrained(model_id, **kwargs)
    model = jlens.from_hf(hf, tok)
    lens = jlens.JacobianLens.load(f"/out/{_slug(model_id)}/lens.pt")
    band = [l for l in range(int(model.n_layers*0.25), int(model.n_layers*0.75)) if l in lens.jacobians]
    lex = {e: sorted({tok(w, add_special_tokens=False).input_ids[0]
                      for w in ws if tok(w, add_special_tokens=False).input_ids}) for e, ws in LEXICON.items()}
    n_sent = len(tok("The meeting has been moved to noon on Thursday.", add_special_tokens=False).input_ids)

    # word -> its first-token id, for per-word evidence
    word_ids = {emo: {w: tok(w, add_special_tokens=False).input_ids[0]
                      for w in ws if tok(w, add_special_tokens=False).input_ids}
                for emo, ws in LEXICON.items()}

    def score(probe):
        msgs = [{"role": "user", "content": probe["user"]},
                {"role": "assistant", "content": probe["assistant_prefill"]}]
        prompt = tok.apply_chat_template(msgs, tokenize=False, continue_final_message=True)
        ll, _, _ = lens.apply(model, prompt, positions=list(range(-n_sent, 0)))
        # best rank per lexicon (log) + best rank per individual word (for receipts)
        rankmats = {}
        for layer in band:
            order = ll[layer].float().argsort(dim=-1, descending=True)
            rm = torch.empty_like(order)
            rm.scatter_(-1, order, torch.arange(order.shape[-1]).expand_as(order))
            rankmats[layer] = rm
        out, word_ranks = {}, {}
        for emo, ids in lex.items():
            t = torch.tensor(ids)
            out[emo] = float(np.log1p(min(int(rankmats[l][:, t].min()) for l in band)))
        for emo, wd in word_ids.items():
            for w, wid in wd.items():
                word_ranks[w.strip()] = min(int(rankmats[l][:, wid].min()) for l in band)
        return out, word_ranks

    neutral, _ = score(covert_probes["neutral"])
    matrix, hits, evidence = {}, 0, {}
    for cond in ["fury", "terror", "grief", "euphoria", "amusement"]:
        s, wr = score(covert_probes[cond])
        matrix[cond] = {e: neutral[e] - s[e] for e in LEXICON}
        if max(matrix[cond], key=matrix[cond].get) == cond:
            hits += 1
        # receipts: this emotion's own words, best-ranked first
        own_words = [w.strip() for w in LEXICON[cond]]
        own = sorted(((w, wr[w]) for w in own_words if w in wr), key=lambda x: x[1])
        evidence[cond] = own[:6]
    return {"model": model_id, "band": [band[0], band[-1]], "diagonal_hits": hits,
            "n_layers": model.n_layers, "delta_matrix": matrix, "evidence": evidence}


@app.local_entrypoint()
def emotions(models: str = "google/gemma-4-26B-A4B-it", out: str = "out/emotion_all.json"):
    import json
    probes = {p["slug"].replace("covert-", ""): p
              for p in json.load(open("probes/emotions.json", encoding="utf-8"))
              if p["slug"].startswith("covert-")}
    model_list = [m.strip() for m in models.split(",")]
    results = list(emotion_matrix.map(model_list, kwargs={"covert_probes": probes}))
    for r in results:
        print(f"\n{r['model']}  band=L{r['band'][0]}..L{r['band'][1]} of {r['n_layers']}  hits={r['diagonal_hits']}/5")
        print(f"{'cond':<10}" + "".join(f"{e[:6]:>8}" for e in r['delta_matrix']['fury']))
        for c, row in r['delta_matrix'].items():
            mark = " <" if max(row, key=row.get) == c else ""
            print(f"{c:<10}" + "".join(f"{row[e]:>+8.2f}" for e in row) + mark)
        print("  evidence:", {c: [f"{w}#{rk}" for w, rk in ev[:3]] for c, ev in r["evidence"].items()})
    with open(out, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)
    print(f"\nwrote {out}")


@app.function(
    image=image, gpu="A100-80GB", timeout=4 * 3600,
    volumes={"/hf": hf_cache, "/out": out_vol},
    secrets=[modal.Secret.from_name("huggingface")],
)
def uncertainty_run(model_id: str, n: int = 500, questions: list | None = None,
                    tag: str = "trivia", max_new: int = 24) -> list:
    """Hallucination probe (Phase 1 of docs/HALLUCINATION_PLAN.md), cloud port
    of probe_uncertainty.py. Reads lens features at the answer position, greedy
    generation, labels vs aliases. `questions` overrides TriviaQA (each item:
    {"q": ..., "aliases": [...], **extra-fields-passed-through})."""
    import json
    import logging
    import os
    os.environ["HF_HOME"] = "/hf"
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
    import numpy as np, torch, transformers, jlens

    BAND_LO, BAND_HI = 0.25, 0.75
    HEDGE_WORDS = [" guess", " maybe", " unsure", " unknown", " perhaps",
                   " possibly", " unclear", " uncertain", "?", " hmm", " Hmm",
                   " probably"]

    tok = transformers.AutoTokenizer.from_pretrained(model_id)
    kwargs = dict(dtype=torch.bfloat16, device_map="cuda")
    try:
        hf = transformers.AutoModelForCausalLM.from_pretrained(model_id, **kwargs)
    except ValueError:
        hf = transformers.AutoModelForImageTextToText.from_pretrained(model_id, **kwargs)
    model = jlens.from_hf(hf, tok)
    lens = jlens.JacobianLens.load(f"/out/{_slug(model_id)}/lens.pt")
    band = [l for l in range(int(model.n_layers * BAND_LO), int(model.n_layers * BAND_HI))
            if l in lens.jacobians]
    hedge_ids = sorted({tid for w in HEDGE_WORDS
                        for tid in tok(w, add_special_tokens=False).input_ids[:1]})

    if questions is None:
        from datasets import load_dataset
        ds = load_dataset("mandarjoshi/trivia_qa", "rc.nocontext",
                          split="validation", streaming=True)
        questions = []
        for rec in ds:
            questions.append({"q": rec["question"],
                              "aliases": rec["answer"]["aliases"] + [rec["answer"]["value"]]})
            if len(questions) == n:
                break
    logging.info("%s: %d questions, band L%d..L%d", model_id, len(questions),
                 band[0], band[-1])

    stop_ids = {tok.eos_token_id}
    for t in ("<end_of_turn>", "<|im_end|>"):
        tid = tok.convert_tokens_to_ids(t)
        if isinstance(tid, int) and tid >= 0:
            stop_ids.add(tid)

    rows = []
    for i, item in enumerate(questions):
        if item.get("system"):
            msgs = [{"role": "system", "content": item["system"]},
                    {"role": "user", "content": item["q"]}]
        elif "clue_depth" in item:
            msgs = [{"role": "user", "content": item["q"]}]  # q self-contained
        else:
            msgs = [{"role": "user", "content":
                     f"Answer with just the answer, nothing else: {item['q']}"}]
        prompt = tok.apply_chat_template(
            msgs, tokenize=False, add_generation_prompt=True, enable_thinking=False)
        input_ids = model.encode(prompt, max_length=1536)
        ids, gen_ids, step_logprobs = input_ids, [], []
        for _ in range(max_new):
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
            ids = torch.cat([ids, torch.tensor([[nxt]], device=ids.device)], dim=1)
        if not gen_ids:
            continue
        answer = tok.decode(gen_ids, skip_special_tokens=True).strip()
        first_answer_id = gen_ids[0]
        baseline = {
            "bl_first_token_logprob": step_logprobs[0],
            "bl_mean_logprob": float(np.mean(step_logprobs)),
            "bl_min_logprob": float(np.min(step_logprobs)),
            "bl_answer_len": len(gen_ids),
        }
        norm = lambda s: "".join(c for c in s.lower() if c.isalnum() or c == " ").strip()
        correct = any(norm(a) and norm(a) in norm(answer)
                      for a in item.get("aliases", []))

        lens_logits, _, _ = lens.apply(model, prompt, positions=[-1])
        ranks_ans, ranks_hedge, entropies, top1s = [], [], [], []
        shape = {"top1_p": [], "rival_mass": [], "tail_mass": [], "eff_k20": []}
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
            # distribution shape: competition (rivals) vs diffuse noise (tail)
            p_sorted = probs[order]
            t1 = float(p_sorted[0]); t5 = float(p_sorted[:5].sum())
            t20 = p_sorted[:20]
            shape["top1_p"].append(round(t1, 5))
            shape["rival_mass"].append(round(t5 - t1, 5))       # mass held by rivals 2-5
            shape["tail_mass"].append(round(1.0 - float(t20.sum()), 5))  # smear beyond top20
            q = t20 / t20.sum()
            shape["eff_k20"].append(round(float(1.0 / (q * q).sum()), 3))  # participation ratio
        ranks_arr = np.array(ranks_ans)
        ignited = np.nonzero(ranks_arr <= 10)[0]
        features = {
            "ignition_frac": float((ranks_arr <= 10).mean()),
            "ignition_depth": float(ignited[0] / len(band)) if len(ignited) else 1.0,
            "mean_log_rank_answer": float(np.log1p(ranks_arr).mean()),
            "band_agreement": float(np.mean(np.array(top1s) == first_answer_id)),
            "mean_entropy": float(np.mean(entropies)),
            "best_hedge_rank_log": float(np.log1p(min(ranks_hedge))),
            "layer_entropies": [round(e, 4) for e in entropies],
            "shape": shape,
        }
        extra = {k: v for k, v in item.items() if k not in ("q", "aliases")}
        rows.append({"q": item["q"], "answer": answer, "correct": correct,
                     **extra, **baseline, **features})
        if (i + 1) % 25 == 0:
            acc = np.mean([r["correct"] for r in rows])
            logging.info("%s %d/%d, running accuracy %.3f", model_id, i + 1,
                         len(questions), acc)

    path = f"/out/{_slug(model_id)}/uncertainty_{tag}_{len(rows)}.jsonl"
    with open(path, "w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row) + "\n")
    out_vol.commit()
    logging.info("wrote %s", path)
    return rows


@app.function(
    image=image, gpu="A100-80GB", timeout=3600,
    volumes={"/hf": hf_cache, "/out": out_vol},
    secrets=[modal.Secret.from_name("huggingface")],
)
def dump_workspace(model_id: str, covert_probes: dict, topk: int = 12) -> dict:
    """Top-k workspace tokens per band layer per covert condition, for the
    interactive guess-the-emotion demo. Small output (tokens + ranks only)."""
    import os
    os.environ["HF_HOME"] = "/hf"
    import torch, transformers, jlens

    tok = transformers.AutoTokenizer.from_pretrained(model_id)
    kwargs = dict(dtype=torch.bfloat16, device_map="cuda")
    try:
        hf = transformers.AutoModelForCausalLM.from_pretrained(model_id, **kwargs)
    except ValueError:
        hf = transformers.AutoModelForImageTextToText.from_pretrained(model_id, **kwargs)
    model = jlens.from_hf(hf, tok)
    lens = jlens.JacobianLens.load(f"/out/{_slug(model_id)}/lens.pt")
    band = [l for l in range(int(model.n_layers * 0.25), int(model.n_layers * 0.75))
            if l in lens.jacobians]
    n_sent = len(tok("The meeting has been moved to noon on Thursday.",
                     add_special_tokens=False).input_ids)

    out = {"model": model_id, "n_layers": model.n_layers,
           "band": [band[0], band[-1]], "conditions": {}}
    for cond, probe in covert_probes.items():
        msgs = [{"role": "user", "content": probe["user"]},
                {"role": "assistant", "content": probe["assistant_prefill"]}]
        prompt = tok.apply_chat_template(msgs, tokenize=False,
                                         continue_final_message=True)
        ll, _, _ = lens.apply(model, prompt, positions=list(range(-n_sent, 0)))
        layers = {}
        for layer in band:
            logits = ll[layer].float()          # [n_sent, vocab]
            # rank-pool: a token's score is its BEST RANK at any position
            # (max-pooling logits lets formatting tokens at their own position
            # drown out emotion tokens that are rank-0 elsewhere)
            seen = {}
            for pos in range(logits.shape[0]):
                top = logits[pos].argsort(descending=True)[:6]
                for r, t in enumerate(top.tolist()):
                    if t not in seen or r < seen[t]:
                        seen[t] = r
            best = sorted(seen.items(), key=lambda x: x[1])[:topk]
            layers[str(layer)] = [tok.decode([t]) for t, _ in best]
        out["conditions"][cond] = layers
    return out


@app.function(
    image=image, gpu="A100-80GB", timeout=3600,
    volumes={"/hf": hf_cache, "/out": out_vol},
    secrets=[modal.Secret.from_name("huggingface")],
)
def dump_qa(model_id: str, questions: list, topk: int = 8) -> list:
    """Per-question workspace snapshot at the answer position: top-k tokens per
    band layer, per-layer entropy, and the rank of the model's own first answer
    token at every layer. For the confidently-right-vs-wrong figure."""
    import os
    os.environ["HF_HOME"] = "/hf"
    import torch, transformers, jlens

    tok = transformers.AutoTokenizer.from_pretrained(model_id)
    kwargs = dict(dtype=torch.bfloat16, device_map="cuda")
    try:
        hf = transformers.AutoModelForCausalLM.from_pretrained(model_id, **kwargs)
    except ValueError:
        hf = transformers.AutoModelForImageTextToText.from_pretrained(model_id, **kwargs)
    model = jlens.from_hf(hf, tok)
    lens = jlens.JacobianLens.load(f"/out/{_slug(model_id)}/lens.pt")
    band = [l for l in range(int(model.n_layers * 0.25), int(model.n_layers * 0.75))
            if l in lens.jacobians]

    out = []
    for item in questions:
        prompt = tok.apply_chat_template(
            [{"role": "user", "content":
              f"Answer with just the answer, nothing else: {item['q']}"}],
            tokenize=False, add_generation_prompt=True, enable_thinking=False)
        ids = model.encode(prompt, max_length=512)
        with torch.no_grad():
            hidden = model.forward(ids).last_hidden_state[:, -1]
            head = model._lm_head
            logits = head(hidden.to(head.weight.dtype).to(head.weight.device))
        first_id = int(logits.argmax(-1))
        ll, _, _ = lens.apply(model, prompt, positions=[-1])
        layers = []
        for layer in band:
            lg = ll[layer][0].float()
            order = lg.argsort(descending=True)
            rank_of = torch.empty_like(order)
            rank_of[order] = torch.arange(len(order))
            probs = lg.softmax(-1)
            ent = float(-(probs * probs.clamp_min(1e-12).log()).sum())
            layers.append({
                "layer": layer,
                "top": [tok.decode([int(t)]) for t in order[:topk]],
                "answer_rank": int(rank_of[first_id]),
                "entropy": round(ent, 3),
            })
        out.append({**item, "first_token": tok.decode([first_id]), "layers": layers})
    return out


@app.local_entrypoint()
def qa_dump(model: str = "google/gemma-4-E4B-it",
            questions_file: str = "out/qa_examples.json",
            out: str = "out/qa_dump.json"):
    import json
    qs = json.load(open(questions_file, encoding="utf-8"))
    res = dump_qa.remote(model, qs)
    with open(out, "w", encoding="utf-8") as f:
        json.dump(res, f, ensure_ascii=False, indent=1)
    print(f"wrote {out} ({len(res)} questions)")


@app.local_entrypoint()
def dump(models: str, out: str = "out/workspace_dump.json"):
    """modal run modal_fit.py::dump --models "a,b" -> demo token data"""
    import json
    probes = {p["slug"].replace("covert-", ""): p
              for p in json.load(open("probes/emotions.json", encoding="utf-8"))
              if p["slug"].startswith("covert-")}
    model_list = [m.strip() for m in models.split(",")]
    results = list(dump_workspace.map(model_list, kwargs={"covert_probes": probes}))
    with open(out, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False)
    print(f"wrote {out} ({len(results)} models)")


@app.local_entrypoint()
def uncertainty(models: str, n: int = 500, tag: str = "trivia",
                questions_file: str = "", max_new: int = 24):
    """modal run modal_fit.py::uncertainty --models "a,b" [--questions-file f.json]"""
    import json
    import os
    qs = None
    if questions_file:
        qs = json.load(open(questions_file, encoding="utf-8"))
    model_list = [m.strip() for m in models.split(",")]
    results = list(uncertainty_run.map(
        model_list, kwargs={"n": n, "questions": qs, "tag": tag,
                            "max_new": max_new}))
    os.makedirs("out", exist_ok=True)
    for mid, rows in zip(model_list, results):
        local = f"out/uncertainty_{tag}_{_slug(mid)}.jsonl"
        with open(local, "w", encoding="utf-8") as f:
            for row in rows:
                f.write(json.dumps(row) + "\n")
        import numpy as np
        acc = np.mean([r["correct"] for r in rows]) if rows else 0
        print(f"{mid}: {len(rows)} rows, accuracy {acc:.3f} -> {local}")


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
