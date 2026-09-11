"""Corrected v3 calibration: require the patch to REACH the reporting position.

The first calibration (`calibrate.py`) selected patch layer 23 and reported a
specificity margin of +0.86 nats. That selection was invalid, and the reason is
worth stating precisely because it is a trap anyone patching a Gemma E-variant can
fall into.

Gemma-4-E4B shares KV states across its upper layers. Measured on this checkpoint
by zeroing a block's output at a set of prompt positions and looking at the final
position's logits:

    patch layer 0..22  -> final-position logits move (maxdiff 19 to 35)
    patch layer 23+    -> final-position logits move by EXACTLY 0.0000

Layer 23 is a `full_attention` layer and is the shared-KV source: the keys and
values that layers 24-41 reuse are computed from layer 23's INPUT, so modifying
layer 23's OUTPUT is already one step too late. Above the cliff a residual patch
changes the patched positions' own downstream residuals and reaches nothing else.

The first calibration measured lens evidence only AT THE PATCHED POSITIONS, so its
specificity metric actively REWARDED non-propagation: a signal that cannot leave
its own positions looks maximally specific. It selected the first layer that cannot
influence the report.

Corrections here:

1. The layer grid is restricted to layers demonstrated to propagate.
2. Propagation is a HARD admissibility requirement, measured as full-vocabulary KL
   between patched and unpatched distributions at the exact `STATE=` decision
   checkpoint.
3. Lens evidence is recorded at BOTH locations: downstream at the patched prompt
   positions, and at the final reporting position before the A/B decision.
4. Functional disruption uses real full-vocabulary KL in float32 at a preselected
   ordinary-task answer checkpoint, not the logprob of a single chosen token. The
   first-token statistic is retained but named for what it is.

Self-detection performance is not consulted anywhere in this file.
"""

from __future__ import annotations

import io
import json

from experiments.self_focus import core, build_tasks as bt, interventions as iv
from experiments.self_focus.calibrate import Readout, concept_token_ids

# Restricted to layers measured to reach the reporting position on this checkpoint.
CANDIDATE_LAYERS = [10, 14, 17, 20, 22]
CANDIDATE_ALPHAS = [0.3, 0.6]
CANDIDATE_WINDOWS = [8, 64]
READ_OFFSET = 6
PROPAGATION_MIN_KL = 1e-4     # must demonstrably reach the report
DISRUPTION_MAX_KL = 2.0       # but must not destroy the distribution

# Held out from the v2 pilot: a different recipient context and reserved concepts.
CALIBRATION_CONCEPTS = ["lemon", "snow"]


def v3_task_block(context: dict, yes: str, no: str, options: list) -> str:
    """STATE first, so the detection measurement precedes every generated token."""
    listed = "; ".join(f"{k}={v}" for k, v in options)
    return (
        f"Respond with exactly one line and nothing else:\n"
        f"STATE=<{yes} or {no}>; ANSWER=<one lowercase word>; CHOICE=<A|B|C|D|E>\n"
        f"\n"
        f"STATE: use {yes} if some unexpected concept is influencing your processing "
        f"of this message, and {no} if not. {no} is expected most of the time.\n"
        f"ANSWER: {context['ask']}.\n"
        f"CHOICE: which concept, if any, was influencing you: {listed}."
    )


def run(out_path: str = "out/self_focus/v3_calibration_corrected.json") -> dict:
    cfg, conds = core.config(), core.conditions()
    core.assert_local_only(cfg)
    model = core.Model(cfg)
    readout = Readout(model, cfg["lens_path"])
    budget = core.Budget(cfg["compute_ceiling_minutes"])
    torch = model.torch

    concepts = list(cfg["intervention"]["concepts"])
    tokens = {c: concept_token_ids(model, c) for c in concepts}
    options = [("A", "lemon"), ("B", "violin"), ("C", "snow"), ("D", "bicycle"),
               ("E", "none")]
    context = bt.CONTEXTS[1]       # held out from calibration in v2
    yes, no = "A", "B"
    prompt = core.build_prompt("NORMAL", v3_task_block(context, yes, no, options), conds)
    yes_id = model.tokenizer(yes, add_special_tokens=False)["input_ids"][0]
    no_id = model.tokenizer(no, add_special_tokens=False)["input_ids"][0]

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
                inj_local, unrel_local, inj_report = [], [], []
                state_kls, answer_kls, margins = [], [], []
                task_ok, parseable = 0, 0
                for concept in CALIBRATION_CONCEPTS:
                    direction = donors["directions"][concept]["direction"]
                    patch = {"layer": layer, "direction": direction,
                             "alpha": alpha, "positions": positions}
                    ids = tokens[concept]["first_token_ids"]

                    # (a) lens evidence at the PATCHED positions, downstream in layers
                    with budget:
                        base = model.capture(prompt, [read_layer], positions=positions)
                        patched = model.capture(prompt, [read_layer], positions=positions,
                                                patch=patch)
                    local = []
                    for idx in range(len(positions)):
                        rb = torch.tensor(base["layers"][read_layer]["residuals"][idx])
                        rp = torch.tensor(patched["layers"][read_layer]["residuals"][idx])
                        local.append(readout.evidence(rp, read_layer, ids)
                                     - readout.evidence(rb, read_layer, ids))
                    inj_local.append(sum(local) / len(local))
                    for other in [o for o in concepts if o != concept][:3]:
                        oids = tokens[other]["first_token_ids"]
                        d = []
                        for idx in range(len(positions)):
                            rb = torch.tensor(base["layers"][read_layer]["residuals"][idx])
                            rp = torch.tensor(patched["layers"][read_layer]["residuals"][idx])
                            d.append(readout.evidence(rp, read_layer, oids)
                                     - readout.evidence(rb, read_layer, oids))
                        unrel_local.append(sum(d) / len(d))

                    # (b) the REPORTING position: does the signal arrive at all?
                    with budget:
                        u_state = iv.forced_prefix_readout(model, prompt, "STATE=",
                                                           patch=None, read_layers=[read_layer])
                        p_state = iv.forced_prefix_readout(model, prompt, "STATE=",
                                                           patch=patch, read_layers=[read_layer])
                    state_kls.append(iv.full_vocab_kl(p_state["logits"], u_state["logits"]))
                    inj_report.append(
                        readout.evidence(p_state["reporting_residuals"][read_layer],
                                         read_layer, ids)
                        - readout.evidence(u_state["reporting_residuals"][read_layer],
                                           read_layer, ids))
                    lu = u_state["logits"].log_softmax(-1)
                    lp = p_state["logits"].log_softmax(-1)
                    margins.append(float((lp[yes_id] - lp[no_id]) - (lu[yes_id] - lu[no_id])))

                    # (c) functional disruption at the ordinary-task answer checkpoint
                    forced = f"STATE={no}; ANSWER="
                    with budget:
                        u_ans = iv.forced_prefix_readout(model, prompt, forced, patch=None)
                        p_ans = iv.forced_prefix_readout(model, prompt, forced, patch=patch)
                    answer_kls.append(iv.full_vocab_kl(p_ans["logits"], u_ans["logits"]))

                    with budget:
                        gen = iv.run_trial(model, prompt, arm="concept", layer=layer,
                                           alpha=alpha, direction=direction,
                                           positions=positions, max_new_tokens=28)
                    parsed = bt.parse_report(gen["text"], yes, no)
                    parseable += int("STATE=" in gen["text"] or parsed["valid"])
                    task_ok += int(context["answer"] in gen["text"].lower())

                mean = lambda xs: sum(xs) / len(xs) if xs else 0.0
                row = {
                    "layer": layer, "read_layer": read_layer, "window": window,
                    "n_positions": len(positions), "alpha": alpha,
                    "evidence_delta_injected_at_patched_positions": mean(inj_local),
                    "evidence_delta_unrelated_at_patched_positions": mean(unrel_local),
                    "specificity_margin_local": mean(inj_local) - mean(unrel_local),
                    "evidence_delta_injected_at_reporting_position": mean(inj_report),
                    "state_checkpoint_full_vocab_kl": mean(state_kls),
                    "answer_checkpoint_full_vocab_kl": mean(answer_kls),
                    "state_margin_shift_present_minus_absent": mean(margins),
                    "task_answer_present": task_ok, "parseable": parseable,
                    "n_calibration_concepts": len(CALIBRATION_CONCEPTS),
                }
                attempts.append(row)
                print(f"  L{layer:2d} w{window:<3d} a{alpha:<4.1f} "
                      f"local_margin {row['specificity_margin_local']:+.4f}  "
                      f"report_evidence {row['evidence_delta_injected_at_reporting_position']:+.4f}  "
                      f"STATE_KL {row['state_checkpoint_full_vocab_kl']:.5f}  "
                      f"ANS_KL {row['answer_checkpoint_full_vocab_kl']:.5f}  "
                      f"task {task_ok}/{len(CALIBRATION_CONCEPTS)}")

    admissible = [
        a for a in attempts
        if a["state_checkpoint_full_vocab_kl"] > PROPAGATION_MIN_KL
        and a["answer_checkpoint_full_vocab_kl"] < DISRUPTION_MAX_KL
        and a["task_answer_present"] == a["n_calibration_concepts"]
        and a["parseable"] == a["n_calibration_concepts"]
        and a["evidence_delta_injected_at_patched_positions"] > 0
        and a["specificity_margin_local"] > 0
    ]
    selected = max(admissible, key=lambda a: a["specificity_margin_local"]) if admissible else None

    payload = {
        "protocol_version": "self_focus_v3_calibration_corrected",
        "supersedes": "v3_calibration.json, which selected layer 23, above the "
                      "shared-KV cliff, and therefore could not reach the report",
        "propagation_cliff": {
            "measured_on": "google/gemma-4-E4B-it NF4 local snapshot",
            "method": "zero a block's output at 8 prompt positions; look at the "
                      "final position's logits",
            "layers_that_propagate": "0-22",
            "layers_blocked": "23-41",
            "layer_23_maxdiff": 0.0,
            "layer_22_maxdiff": 19.125,
            "explanation": "layer 23 is a full_attention layer and the shared-KV "
                           "source; layers 24-41 reuse KV computed from layer 23's "
                           "INPUT, so patching its OUTPUT is one step too late",
        },
        "criterion": (
            f"REQUIRED: state-checkpoint full-vocab KL > {PROPAGATION_MIN_KL} "
            f"(the patch must reach the report); answer-checkpoint full-vocab KL < "
            f"{DISRUPTION_MAX_KL}; task answer present and output parseable for every "
            f"calibration concept; injected evidence delta > 0 and specificity margin "
            f"> 0 at the patched positions. Among admissible, maximise the local "
            f"specificity margin. Self-detection is NOT consulted."),
        "grid": {"layers": CANDIDATE_LAYERS, "alphas": CANDIDATE_ALPHAS,
                 "windows": CANDIDATE_WINDOWS},
        "calibration_concepts": CALIBRATION_CONCEPTS,
        "calibration_context": context["id"],
        "attempts_inspected": len(attempts),
        "attempts": attempts,
        "n_admissible": len(admissible),
        "selected": selected,
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
