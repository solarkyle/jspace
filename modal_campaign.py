"""Two-pass campaign trace runner (handoff section 9).

Pass 1: KV-cached generation (hf.generate) -> answer + per-token logprobs.
Pass 2: ONE teacher-forced forward over [prompt + answer], capturing residuals
        only at selected positions, transported through the fitted lens.

Under greedy decoding the teacher-forced sequence IS the generated sequence, so
the activation at sequence position (len_prompt - 1 + k) equals the
autoregressive workspace read that predicts answer token k. The --verify path
checks this numerically on the first N prompts before the optimization is
trusted (handoff 9.0).

Local entrypoints:
    modal run modal_campaign.py::run --manifest campaign/manifests/pilot.jsonl \
        --model google/gemma-4-12B-it --tag pilot --verify 50
    modal run modal_campaign.py::run --manifest campaign/manifests/stage1.jsonl \
        --model google/gemma-4-12B-it --tag stage1 --shard 0 --n-shards 8

Download traces:
    modal volume get jlens-out <slug>/campaign_<tag>_shard<k>.jsonl out/...
"""

import json
import os

import modal

app = modal.App("jlens-campaign")

GPU = os.environ.get("JLENS_GPU", "L40S")

image = (
    modal.Image.debian_slim(python_version="3.12")
    .apt_install("git")
    .pip_install("torch", "transformers>=5.5", "datasets", "accelerate",
                 "huggingface_hub", "bitsandbytes")
    .pip_install("git+https://github.com/anthropics/jacobian-lens")
)

hf_cache = modal.Volume.from_name("jlens-hf-cache", create_if_missing=True)
out_vol = modal.Volume.from_name("jlens-out", create_if_missing=True)

BAND_LO, BAND_HI = 0.25, 0.75
PREFIX_FRACS = [0.0, 0.5, 1.0]   # onset, mid-answer, last answer token
HEDGE_WORDS = [" guess", " maybe", " unsure", " unknown", " perhaps", " possibly",
               " unclear", " uncertain", "?", " hmm", " Hmm", " probably"]


def _slug(model_id: str) -> str:
    return model_id.split("/")[-1].lower()


@app.function(
    image=image, gpu=GPU, timeout=6 * 3600,
    volumes={"/hf": hf_cache, "/out": out_vol},
    secrets=[modal.Secret.from_name("huggingface")],
)
def run_shard(model_id: str, prompts: list, tag: str, shard: int,
              max_new: int = 64, quant: str = "", verify: int = 0) -> dict:
    import logging
    import time
    os.environ["HF_HOME"] = "/hf"
    os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
    import numpy as np, torch, transformers, jlens
    from jlens.hooks import ActivationRecorder

    tok = transformers.AutoTokenizer.from_pretrained(model_id)
    kwargs = dict(dtype=torch.bfloat16, device_map="cuda")
    if quant == "4bit":
        kwargs = dict(quantization_config=transformers.BitsAndBytesConfig(
            load_in_4bit=True, bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16), device_map="cuda")
    try:
        hf = transformers.AutoModelForCausalLM.from_pretrained(model_id, **kwargs)
    except ValueError:
        hf = transformers.AutoModelForImageTextToText.from_pretrained(model_id, **kwargs)
    hf.eval()
    hf_cache.commit()
    model = jlens.from_hf(hf, tok)
    lens = jlens.JacobianLens.load(f"/out/{_slug(model_id)}/lens.pt")
    band = [l for l in range(int(model.n_layers * BAND_LO), int(model.n_layers * BAND_HI))
            if l in lens.jacobians]
    final_layer = model.n_layers - 1
    record_at = sorted({*band, final_layer})
    hedge_ids = sorted({tid for w in HEDGE_WORDS
                        for tid in tok(w, add_special_tokens=False).input_ids[:1]})

    stop_ids = {tok.eos_token_id}
    for t in ("<end_of_turn>", "<|im_end|>"):
        tid = tok.convert_tokens_to_ids(t)
        if isinstance(tid, int) and tid >= 0:
            stop_ids.add(tid)

    def lens_logits_at(acts, pos):
        """Transport every band layer's residual at sequence position `pos`."""
        out = {}
        for layer in band:
            residual = lens.transport(acts[layer][0, pos:pos + 1].float(), layer)
            out[layer] = model.unembed(residual).float().cpu()[0]
        return out

    def features(lens_logits, target_id):
        ranks_ans, ranks_hedge, entropies, top1s = [], [], [], []
        shape = {"top1_p": [], "rival_mass": [], "tail_mass": [], "eff_k20": []}
        for layer in band:
            logits = lens_logits[layer].float()
            order = logits.argsort(descending=True)
            rank_of = torch.empty_like(order)
            rank_of[order] = torch.arange(len(order))
            ranks_ans.append(int(rank_of[target_id]))
            ranks_hedge.append(int(min(rank_of[t] for t in hedge_ids)))
            probs = logits.softmax(-1)
            entropies.append(float(-(probs * probs.clamp_min(1e-12).log()).sum()))
            top1s.append(int(order[0]))
            p_sorted = probs[order]
            t1 = float(p_sorted[0]); t5 = float(p_sorted[:5].sum())
            t20 = p_sorted[:20]
            shape["top1_p"].append(round(t1, 5))
            shape["rival_mass"].append(round(t5 - t1, 5))
            shape["tail_mass"].append(round(1.0 - float(t20.sum()), 5))
            q = t20 / t20.sum()
            shape["eff_k20"].append(round(float(1.0 / (q * q).sum()), 3))
        ranks_arr = np.array(ranks_ans)
        ignited = np.nonzero(ranks_arr <= 10)[0]
        return {
            "ignition_frac": float((ranks_arr <= 10).mean()),
            "ignition_depth": float(ignited[0] / len(band)) if len(ignited) else 1.0,
            "mean_log_rank_answer": float(np.log1p(ranks_arr).mean()),
            "band_agreement": float(np.mean(np.array(top1s) == target_id)),
            "mean_entropy": float(np.mean(entropies)),
            "best_hedge_rank_log": float(np.log1p(min(ranks_hedge))),
            "layer_entropies": [round(e, 4) for e in entropies],
            "shape": shape,
        }

    def build_user(p):
        if p.get("context"):
            return f"Context:\n{p['context']}\n\nQuestion: {p['prompt']}"
        return p["prompt"]

    rows, t0, gen_tok_total, prompt_tok_total = [], time.time(), 0, 0
    max_verify_err = 0.0
    verify_feat_err = {}
    for i, p in enumerate(prompts):
        msgs = []
        if p.get("system"):
            msgs.append({"role": "system", "content": p["system"]})
        msgs.append({"role": "user", "content": build_user(p)})
        prompt = tok.apply_chat_template(msgs, tokenize=False,
                                         add_generation_prompt=True, enable_thinking=False)
        input_ids = model.encode(prompt, max_length=2048)
        n_prompt = input_ids.shape[1]

        with torch.no_grad():
            gen = hf.generate(input_ids, max_new_tokens=max_new, do_sample=False,
                              return_dict_in_generate=True, output_scores=True,
                              pad_token_id=tok.eos_token_id)
        seq = gen.sequences[0]
        gen_ids = seq[n_prompt:].tolist()
        # trim at first stop token
        cut = len(gen_ids)
        for k, t in enumerate(gen_ids):
            if t in stop_ids:
                cut = k
                break
        gen_ids = gen_ids[:cut]
        if not gen_ids:
            continue
        step_logprobs = []
        for k in range(len(gen_ids)):
            lp = gen.scores[k][0].float().log_softmax(-1)
            step_logprobs.append(float(lp[gen_ids[k]]))
        answer = tok.decode(gen_ids, skip_special_tokens=True).strip()
        L = len(gen_ids)
        gen_tok_total += L
        prompt_tok_total += n_prompt

        # Pass 2: teacher-force [prompt + answer], capture selected positions.
        full = torch.cat([input_ids, torch.tensor([gen_ids], device=input_ids.device)], dim=1)
        with torch.no_grad():
            with ActivationRecorder(model.layers, at=record_at) as rec:
                model.forward(full)
                acts = {j: rec.activations[j].detach() for j in record_at}

        base_pos = n_prompt - 1  # predicts gen_ids[0]
        prefix_feats = []
        for frac in PREFIX_FRACS:
            k = int(round(frac * (L - 1)))
            pos = base_pos + k
            ll = lens_logits_at(acts, pos)
            f = features(ll, gen_ids[k])
            f["frac"] = frac
            f["token_index"] = k
            prefix_feats.append(f)
        onset = prefix_feats[0]

        # equivalence check: onset from teacher-force vs autoregressive snapshot.
        # Raw logits differ by bf16 kernel noise; what matters is whether the
        # CONSUMED features and the answer-token rank agree.
        if i < verify:
            with torch.no_grad():
                with ActivationRecorder(model.layers, at=record_at) as rec2:
                    model.forward(input_ids)
                    a2 = {j: rec2.activations[j].detach() for j in record_at}
            ll_auto = lens_logits_at(a2, n_prompt - 1)
            ll_tf = lens_logits_at(acts, base_pos)
            err = max(float(ll_tf[l].sub(ll_auto[l]).abs().max()) for l in band)
            max_verify_err = max(max_verify_err, err)
            f_auto = features(ll_auto, gen_ids[0])
            for key in ("mean_log_rank_answer", "ignition_frac", "mean_entropy",
                        "band_agreement", "ignition_depth"):
                verify_feat_err[key] = max(verify_feat_err.get(key, 0.0),
                                           abs(onset[key] - f_auto[key]))

        ans_lp = step_logprobs
        rows.append({
            "example_id": p["example_id"], "source_dataset": p["source_dataset"],
            "domain": p["domain"], "task_type": p["task_type"],
            "grader_type": p["grader_type"], "answerable": p.get("answerable", True),
            "split_group": p["split_group"], "upstream_group": p["upstream_group"],
            "prompt": p["prompt"], "context": p.get("context", ""),
            "references": p.get("references", []), "aliases": p.get("aliases", []),
            "answer": answer, "token_count": L,
            "logprob_features": {
                "bl_first_token_logprob": ans_lp[0],
                "bl_mean_logprob": float(np.mean(ans_lp)),
                "bl_min_logprob": float(np.min(ans_lp)),
                "bl_answer_len": L,
            },
            "onset_workspace_features": onset,
            "prefix_workspace_features": prefix_feats,
            "metadata": p.get("metadata", {}),
        })
        if (i + 1) % 25 == 0:
            dt = time.time() - t0
            logging.info("%d/%d  %.1f prompts/s  gen_tok=%d", i + 1, len(prompts),
                         (i + 1) / dt, gen_tok_total)

    dt = time.time() - t0
    out_dir = f"/out/{_slug(model_id)}"
    os.makedirs(out_dir, exist_ok=True)
    path = f"{out_dir}/campaign_{tag}_shard{shard}.jsonl"
    with open(path, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")
    out_vol.commit()
    stats = {
        "shard": shard, "n_in": len(prompts), "n_out": len(rows),
        "seconds": round(dt, 1), "prompts_per_s": round(len(rows) / dt, 3),
        "gen_tokens": gen_tok_total, "prompt_tokens": prompt_tok_total,
        "gen_tok_per_s": round(gen_tok_total / dt, 1),
        "max_verify_abs_err": max_verify_err, "verified": min(verify, len(rows)),
        "verify_feat_err": {k: round(v, 5) for k, v in verify_feat_err.items()},
        "path": path,
    }
    logging.info("STATS %s", json.dumps(stats))
    return stats


@app.local_entrypoint()
def run(manifest: str, model: str = "google/gemma-4-12B-it", tag: str = "pilot",
        shard: int = 0, n_shards: int = 1, max_new: int = 64, quant: str = "",
        verify: int = 0):
    prompts = [json.loads(l) for l in open(manifest, encoding="utf-8") if l.strip()]
    if n_shards > 1:
        prompts = prompts[shard::n_shards]
    print(f"{len(prompts)} prompts, shard {shard}/{n_shards}, model {model}")
    stats = run_shard.remote(model, prompts, tag, shard, max_new, quant, verify)
    print("STATS:", json.dumps(stats, indent=2))
    if stats["verified"]:
        fe = stats["verify_feat_err"]
        print(f"\nequivalence over {stats['verified']} prompts:")
        print(f"  raw logit max abs err: {stats['max_verify_abs_err']:.3f} (bf16 noise expected)")
        print(f"  consumed-feature max abs deltas: {json.dumps(fe)}")
        rank_ok = fe.get("mean_log_rank_answer", 9) < 0.05
        ok = rank_ok and fe.get("mean_entropy", 9) < 0.1
        print(f"  -> {'PASS (features stable; two-pass trusted)' if ok else 'INVESTIGATE'}")
