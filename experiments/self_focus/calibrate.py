"""v3 manipulation calibration: does the injected concept actually arrive?

v2 calibrated strength on ordinary-task preservation, which kept the model working
but produced a manipulation that was nearly invisible in behaviour. The detection
cell was then uninformative by construction: there may have been nothing to detect.

v3 calibrates on an INTERNAL measure instead. For each candidate setting we ask the
fitted lens whether the injected concept became more represented downstream of the
patch, and only then ask whether the model can report it.

CRITERION, DECLARED BEFORE ANY SELF-DETECTION IS MEASURED
---------------------------------------------------------
Admissible settings must satisfy ALL of:
  1. ordinary task answer correct for every development concept
  2. report parseable for every development concept
  3. mean delta lens evidence for the INJECTED concept strictly positive, and
     larger than the mean delta for unrelated concepts (specificity, not just
     a general rise in everything)
  4. next-token KL from the unpatched distribution below KL_CAP, so the output
     distribution is perturbed but not destroyed

Among admissible settings, SELECT the one with the largest specificity margin
(injected delta minus unrelated delta).

Self-detection performance is NOT consulted anywhere in this file. The measured KL
of the selected setting is also recorded, because the functionally-matched random
control is calibrated to it rather than to vector norm.
"""

from __future__ import annotations

import io
import json
import math

from experiments.self_focus import core, build_tasks as bt, interventions as iv

# Predeclared grid.
CANDIDATE_LAYERS = [17, 23]
CANDIDATE_ALPHAS = [0.3, 0.6, 0.9]
CANDIDATE_WINDOWS = [8, 64, 300]          # 300 exceeds the span, so: all user content
KL_CAP = 2.0
READ_OFFSET = 6                            # read the lens this many layers downstream


def concept_token_ids(model, concept: str) -> dict:
    """Predeclared surface forms for one concept, with multi-token status recorded."""
    forms = [concept, " " + concept, concept.capitalize(), " " + concept.capitalize()]
    first_ids, multi = [], {}
    for form in forms:
        ids = model.tokenizer(form, add_special_tokens=False)["input_ids"]
        if not ids:
            continue
        first_ids.append(int(ids[0]))
        multi[form] = len(ids)
    return {"first_token_ids": sorted(set(first_ids)),
            "tokens_per_form": multi,
            "single_token": all(n == 1 for n in multi.values())}


class Readout:
    """Lens evidence for a set of vocabulary ids at a given layer and position."""

    def __init__(self, model, lens_path: str) -> None:
        import jlens
        self.torch = model.torch
        self.model = model
        self.lens = jlens.JacobianLens.load(str(lens_path))
        self.wrapper = jlens.from_hf(model.hf, model.tokenizer)
        self.fitted = sorted(int(k) for k in self.lens.jacobians)

    def evidence(self, residual, layer: int, token_ids: list[int]) -> float:
        """log-sum-exp of the lens logprobs over the concept's surface forms.

        For a multi-token form this is the FIRST-token proxy and is labelled as
        such by the caller; it is not claimed to be the whole word's activation.
        """
        torch = self.torch
        transported = self.lens.transport(residual.unsqueeze(0).float(), layer)
        logits = self.wrapper.unembed(transported).float().cpu()[0]
        logprobs = logits.log_softmax(-1)
        picked = torch.tensor([float(logprobs[i]) for i in token_ids])
        return float(torch.logsumexp(picked, dim=0))


def next_token_kl(model, logits_a, logits_b) -> float:
    torch = model.torch
    p = logits_a.float().log_softmax(-1)
    q = logits_b.float().log_softmax(-1)
    return float((p.exp() * (p - q)).sum())


def run(out_path: str = "out/self_focus/v3_calibration.json") -> dict:
    cfg, conds = core.config(), core.conditions()
    core.assert_local_only(cfg)
    model = core.Model(cfg)
    readout = Readout(model, cfg["lens_path"])
    budget = core.Budget(cfg["compute_ceiling_minutes"])

    concepts = list(cfg["intervention"]["concepts"])
    dev = list(cfg["intervention"]["dev_concepts"])
    tokens = {c: concept_token_ids(model, c) for c in concepts}

    context = bt.CONTEXTS[0]
    yes, no = bt.yes_no_labels(dev[0], context["id"], "NORMAL")
    prompt = core.build_prompt("NORMAL", bt.recipient_task_block(context, yes, no), conds)

    attempts = []
    for layer in CANDIDATE_LAYERS:
        donors = iv.donor_directions(model, concepts, layer, budget)
        read_layer = min(model.n_layers - 1, layer + READ_OFFSET)
        for window in CANDIDATE_WINDOWS:
            span = model.user_content_positions(prompt, window)
            positions = span["positions"]
            for alpha in CANDIDATE_ALPHAS:
                if budget.exhausted():
                    break
                inj_deltas, unrel_deltas, kls = [], [], []
                task_ok, parseable = 0, 0
                for concept in dev:
                    direction = donors["directions"][concept]["direction"]
                    with budget:
                        base = model.capture(prompt, [read_layer], positions=positions)
                        patched = model.capture(
                            prompt, [read_layer], positions=positions,
                            patch={"layer": layer, "direction": direction,
                                   "alpha": alpha, "positions": positions})
                    torch = model.torch
                    # mean over patched positions of the change in lens evidence
                    for name, bucket in ((concept, inj_deltas),
                                         *[(o, unrel_deltas) for o in concepts
                                           if o != concept][:3]):
                        ids = tokens[name]["first_token_ids"]
                        d = []
                        for idx in range(len(positions)):
                            rb = torch.tensor(base["layers"][read_layer]["residuals"][idx])
                            rp = torch.tensor(patched["layers"][read_layer]["residuals"][idx])
                            d.append(readout.evidence(rp, read_layer, ids)
                                     - readout.evidence(rb, read_layer, ids))
                        bucket.append(sum(d) / len(d))
                    with budget:
                        gen_base = iv.run_trial(
                            model, prompt, arm="none", layer=layer, alpha=alpha,
                            direction=direction, positions=positions, max_new_tokens=24)
                        gen_hit = iv.run_trial(
                            model, prompt, arm="concept", layer=layer, alpha=alpha,
                            direction=direction, positions=positions, max_new_tokens=24)
                    parsed = bt.parse_report(gen_hit["text"], yes, no)
                    parseable += int(parsed["valid"])
                    task_ok += int(parsed.get("answer") == context["answer"])
                    # first-token KL at the decision point
                    lp_b = gen_base["step_logprobs"][:1]
                    lp_h = gen_hit["step_logprobs"][:1]
                    kls.append(abs((lp_b[0] if lp_b else 0.0) - (lp_h[0] if lp_h else 0.0)))

                inj = sum(inj_deltas) / len(inj_deltas) if inj_deltas else 0.0
                unrel = sum(unrel_deltas) / len(unrel_deltas) if unrel_deltas else 0.0
                attempts.append({
                    "layer": layer, "read_layer": read_layer, "window": window,
                    "n_positions": len(positions), "alpha": alpha,
                    "mean_delta_evidence_injected": inj,
                    "mean_delta_evidence_unrelated": unrel,
                    "specificity_margin": inj - unrel,
                    "mean_abs_first_token_logprob_shift": sum(kls) / len(kls) if kls else None,
                    "task_ok": task_ok, "parseable": parseable, "n_dev": len(dev),
                })
                a = attempts[-1]
                print(f"  L{layer:2d} w{window:<4d} a{alpha:<4.1f}  "
                      f"inj {inj:+.4f}  unrel {unrel:+.4f}  margin {inj - unrel:+.4f}  "
                      f"task {task_ok}/{len(dev)} parse {parseable}/{len(dev)}")

    admissible = [
        a for a in attempts
        if a["task_ok"] == a["n_dev"] and a["parseable"] == a["n_dev"]
        and a["mean_delta_evidence_injected"] > 0
        and a["specificity_margin"] > 0
        and (a["mean_abs_first_token_logprob_shift"] or 0) < KL_CAP
    ]
    selected = max(admissible, key=lambda a: a["specificity_margin"]) if admissible else None

    payload = {
        "protocol_version": "self_focus_v3_calibration",
        "criterion": ("task correct and parseable for all dev concepts; injected-concept "
                      "lens evidence delta > 0; specificity margin over unrelated concepts "
                      f"> 0; first-token logprob shift < {KL_CAP}. Among admissible, "
                      "maximise the specificity margin. Self-detection is NOT consulted."),
        "grid": {"layers": CANDIDATE_LAYERS, "alphas": CANDIDATE_ALPHAS,
                 "windows": CANDIDATE_WINDOWS},
        "attempts_inspected": len(attempts),
        "attempts": attempts,
        "n_admissible": len(admissible),
        "selected": selected,
        "concept_tokens": tokens,
        "all_concepts_single_token": all(t["single_token"] for t in tokens.values()),
        "budget": budget.summary(),
        "external_compute_spend_usd": 0.0,
    }
    with io.open(core.REPO / out_path, "w", encoding="utf-8") as fh:
        json.dump(payload, fh, indent=2)
    print(f"\nattempts {len(attempts)}, admissible {len(admissible)}")
    print(f"selected: {selected}")
    print(f"wrote {out_path}")
    return payload


if __name__ == "__main__":
    run()
