"""Hidden-concept intervention: donor directions, prefill patching, arms.

The mechanism facts that make or break this, both pinned by tests:

* Forward hooks fire in registration order, so a patch hook must be registered
  before any recorder on the same module.
* A patch at layer L and positions P changes layer L's output only at P. It
  reaches other positions only once the next layer attends across them, so the
  effect is read downstream of L.
* The patch is applied during PREFILL over user-content positions, and the hook is
  removed before the assistant emits its first token. Generation then continues
  from the live patched cache. An injection applied only in a post-generation
  replay could not test causal access at all.

A direction is never derived from the recipient prompt, its correct answer, or its
eventual report. Donor prompts are independent, and the baseline is the mean over
unrelated donor concepts.
"""

from __future__ import annotations

import time
from typing import Any

from experiments.self_focus import core

# Independent donor templates. Deliberately plain, and none of them mentions
# monitoring, introspection, detection or the recipient task.
DONOR_TEMPLATES = [
    "Write the single word: {c}",
    "The topic of this sentence is {c}.",
    "Here is a short note about a {c}.",
    "{c}",
    "Describe a {c} in one word.",
]

ARMS = ("none", "sham", "concept", "random", "steer")
ARM_NOTES = {
    "none": "no hook at all; also serves as the unmodified observer baseline "
            "with identical visible context",
    "sham": "hook registered at zero strength; isolates the hook itself",
    "concept": "the concept direction",
    "random": "an independently seeded random direction, same layer, timing and "
              "per-position norm",
    "steer": "no activation change; output logits biased toward the concept only "
             "AFTER the detection field is emitted, so mere concept mention "
             "cannot count as detection",
}


def donor_directions(model: Any, concepts: list[str], layer: int,
                     budget: core.Budget) -> dict:
    """Mean donor activation per concept at `layer`, minus the unrelated mean.

    Returns {concept: {"direction": tensor, "norm": float, "n_donors": int}} and
    counts every forward pass against the compute budget.
    """
    torch = model.torch
    per_concept: dict[str, Any] = {}
    passes = 0
    for concept in concepts:
        vectors = []
        for template in DONOR_TEMPLATES:
            if budget.exhausted():
                break
            prompt = template.format(c=concept)
            with budget:
                cap = model.capture(prompt, [layer])
            passes += 1
            # last recorded position: the end of the donor prompt
            residuals = cap["layers"][layer]["residuals"]
            vectors.append(torch.tensor(residuals[-1], dtype=torch.float32))
        if vectors:
            per_concept[concept] = torch.stack(vectors).mean(dim=0)

    if not per_concept:
        return {"directions": {}, "donor_passes": passes}

    stacked = torch.stack(list(per_concept.values()))
    grand_mean = stacked.mean(dim=0)
    out = {}
    for concept, vector in per_concept.items():
        # baseline is the mean over the OTHER concepts, so a concept is never its
        # own baseline
        others = [v for c, v in per_concept.items() if c != concept]
        baseline = torch.stack(others).mean(dim=0) if others else grand_mean
        direction = vector - baseline
        norm = float(direction.norm())
        if norm <= 1e-9:
            continue
        out[concept] = {
            "direction": direction / norm,
            "raw_norm": norm,
            "n_donors": len(DONOR_TEMPLATES),
            "baseline": "mean of the other donor concepts",
        }
    return {"directions": out, "donor_passes": passes,
            "hash": core.stable_id(sorted(concepts), layer, DONOR_TEMPLATES)}


def random_direction(model: Any, dim: int, seed: int):
    torch = model.torch
    gen = torch.Generator().manual_seed(int(seed))
    vector = torch.randn(dim, generator=gen, dtype=torch.float32)
    return vector / vector.norm()


class ConceptSteerer:
    """Bias the concept's tokens, but only after the detection field is closed.

    This is the control that stops "the model said lemon" from being read as
    "the model detected lemon". It changes output only, never activations.
    """

    def __init__(self, model: Any, concept: str, bias: float) -> None:
        self.tokenizer = model.tokenizer
        self.bias = float(bias)
        ids = set()
        for form in (concept, " " + concept, concept.capitalize(),
                     " " + concept.capitalize()):
            encoded = self.tokenizer(form, add_special_tokens=False)["input_ids"]
            if encoded:
                ids.add(int(encoded[0]))
        self.token_ids = sorted(ids)
        self.armed = False

    def observe(self, text_so_far: str) -> None:
        # arm once the decision has been emitted and terminated
        if not self.armed and "DETECTED=" in text_so_far and ";" in text_so_far.split("DETECTED=")[-1]:
            self.armed = True

    def apply(self, logits) -> Any:
        if not self.armed or not self.token_ids:
            return logits
        out = logits.clone()
        for tid in self.token_ids:
            out[0, tid] = out[0, tid] + self.bias
        return out


def run_trial(model: Any, prompt: str, *, arm: str, layer: int, alpha: float,
              direction: Any, positions: list[int], max_new_tokens: int,
              steerer: ConceptSteerer | None = None,
              observe_layers: list[int] | None = None) -> dict:
    """Prefill (optionally patched), drop the hook, then generate from the cache."""
    torch = model.torch
    rendered = model.render(prompt)
    input_ids = model.encode(rendered)
    n_prompt = int(input_ids.shape[1])

    recorded: dict[int, Any] = {}
    handles = []
    applied = {"relative_norms": [], "patched_positions": []}

    def patcher(_module, _inputs, output):
        hidden = output if torch.is_tensor(output) else output[0]
        unit = direction.to(device=hidden.device, dtype=torch.float32)
        unit = unit / unit.norm().clamp_min(1e-12)
        changed = hidden.clone()
        for pos in positions:
            if pos >= hidden.shape[1]:
                continue
            base = hidden[:, pos:pos + 1].float()
            base_norm = base.norm().clamp_min(1e-12)
            delta = unit * (effective_alpha * base_norm)
            changed[:, pos:pos + 1] = (base + delta).to(dtype=hidden.dtype)
            applied["relative_norms"].append(
                float(delta.norm().item() / base_norm.item()))
            applied["patched_positions"].append(int(pos))
        if torch.is_tensor(output):
            return changed
        return (changed,) + tuple(output[1:])

    def recorder(layer_index: int):
        def hook(_module, _inputs, output):
            hidden = output if torch.is_tensor(output) else output[0]
            recorded[layer_index] = hidden.detach()[0, -1].float().cpu()
            return output
        return hook

    # The sham arm registers the SAME hook at zero strength, so it isolates the
    # presence of the hook from the perturbation. Passing the caller's alpha here
    # made sham a duplicate of concept while still recording patch_alpha as 0.0,
    # which is a wrong-provenance bug as well as a dead control.
    effective_alpha = 0.0 if arm == "sham" else float(alpha)

    t0 = time.perf_counter()
    try:
        # patch first: a recorder registered earlier on the same module would
        # capture the pre-patch output
        if arm in ("concept", "random", "sham"):
            handles.append(model.blocks[layer].register_forward_hook(patcher))
        for observed in (observe_layers or []):
            handles.append(model.blocks[observed].register_forward_hook(recorder(observed)))
        with torch.no_grad():
            out = model.hf(input_ids, use_cache=True)
    finally:
        for handle in handles:
            handle.remove()

    # the hook is gone before a single report token exists; generation continues
    # from the live patched cache
    past = out.past_key_values
    logits = out.logits[:, -1, :].float()
    gen_ids: list[int] = []
    step_logprobs: list[float] = []
    stop = model._stop_ids()
    with torch.no_grad():
        for _ in range(int(max_new_tokens)):
            biased = steerer.apply(logits) if steerer else logits
            nxt = int(biased.argmax(dim=-1).item())
            if nxt in stop:
                break
            lp = logits.log_softmax(-1)
            step_logprobs.append(float(lp[0, nxt]))
            gen_ids.append(nxt)
            if steerer:
                steerer.observe(model.tokenizer.decode(gen_ids, skip_special_tokens=True))
            step = torch.tensor([[nxt]], device=input_ids.device, dtype=input_ids.dtype)
            out = model.hf(step, past_key_values=past, use_cache=True)
            past = out.past_key_values
            logits = out.logits[:, -1, :].float()
    elapsed = time.perf_counter() - t0

    return {
        "arm": arm,
        "rendered_prompt": rendered,
        "n_prompt_tokens": n_prompt,
        "prompt_token_ids": input_ids[0].tolist(),
        "text": model.tokenizer.decode(gen_ids, skip_special_tokens=True),
        "gen_token_ids": gen_ids,
        "step_logprobs": step_logprobs,
        "patch_layer": layer if arm in ("concept", "random", "sham") else None,
        "patch_alpha": effective_alpha if arm in ("concept", "random", "sham") else None,
        "patch_alpha_requested": float(alpha),
        "applied_relative_norms": applied["relative_norms"],
        "patched_positions": sorted(set(applied["patched_positions"])),
        "steer_armed": bool(steerer.armed) if steerer else False,
        "observed_last_token_norms": {str(k): float(v.norm()) for k, v in recorded.items()},
        "seconds": round(elapsed, 3),
    }


def forced_prefix_readout(model: Any, prompt: str, forced_text: str, *,
                          patch: dict | None = None,
                          read_layers: list[int] | None = None) -> dict:
    """Prefill (optionally patched), drop the hook, then feed a FORCED prefix.

    Returns the full next-token logits at the position immediately after
    `forced_text`, plus residuals there. This is how a decision field is measured
    without letting the model generate any text of its own first.

    The forced tokens are processed from the patched prefill cache, so the reporting
    computation genuinely descends from the intervention. No hook is active during
    the forced pass, which is the required lifetime: the patch touches prompt
    positions only.
    """
    torch = model.torch
    rendered = model.render(prompt)
    input_ids = model.encode(rendered)
    handles = []
    try:
        if patch is not None:
            layer = int(patch["layer"])
            positions = list(patch["positions"])
            alpha = float(patch["alpha"])
            direction = patch["direction"]

            def patcher(_module, _inputs, output):
                hidden = output if torch.is_tensor(output) else output[0]
                unit = direction.to(device=hidden.device, dtype=torch.float32)
                unit = unit / unit.norm().clamp_min(1e-12)
                changed = hidden.clone()
                for pos in positions:
                    if pos >= hidden.shape[1]:
                        continue
                    base = hidden[:, pos:pos + 1].float()
                    delta = unit * (alpha * base.norm().clamp_min(1e-12))
                    changed[:, pos:pos + 1] = (base + delta).to(dtype=hidden.dtype)
                if torch.is_tensor(output):
                    return changed
                return (changed,) + tuple(output[1:])

            handles.append(model.blocks[layer].register_forward_hook(patcher))
        with torch.no_grad():
            out = model.hf(input_ids, use_cache=True)
    finally:
        for handle in handles:
            handle.remove()

    past = out.past_key_values
    logits = out.logits[:, -1, :].float()
    forced_ids = model.tokenizer(forced_text, add_special_tokens=False)["input_ids"]

    recorded: dict[int, Any] = {}
    handles = []

    def recorder(layer_index: int):
        def hook(_module, _inputs, output):
            hidden = output if torch.is_tensor(output) else output[0]
            recorded[layer_index] = hidden.detach()[0, -1].float().cpu()
            return output
        return hook

    try:
        for layer_index in (read_layers or []):
            handles.append(model.blocks[layer_index].register_forward_hook(
                recorder(layer_index)))
        with torch.no_grad():
            for i, tok in enumerate(forced_ids):
                step = torch.tensor([[int(tok)]], device=input_ids.device,
                                    dtype=input_ids.dtype)
                out = model.hf(step, past_key_values=past, use_cache=True)
                past = out.past_key_values
                logits = out.logits[:, -1, :].float()
    finally:
        for handle in handles:
            handle.remove()

    return {"logits": logits[0].cpu(), "forced_ids": forced_ids,
            "n_prompt_tokens": int(input_ids.shape[1]),
            "reporting_residuals": {k: v for k, v in recorded.items()},
            "patched": patch is not None}


def full_vocab_kl(logits_p, logits_q) -> float:
    """KL(P || Q) over the WHOLE vocabulary, in float32.

    The calibration previously compared the logprob of the single chosen token and
    described it as a next-token KL. Those are different things: a formatting token
    can hold identical probability in both distributions while the preference
    between two answer labels reverses underneath it.
    """
    import torch
    p = logits_p.float().log_softmax(-1)
    q = logits_q.float().log_softmax(-1)
    return float((p.exp() * (p - q)).sum())
