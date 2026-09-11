"""v3 pilot: continuous self-detection score, both frozen settings, local only.

FROZEN BEFORE ANY SELF-DETECTION WAS MEASURED
    PRIMARY    layer 17, window 64, alpha 0.6
    SECONDARY  layer 10, window 64, alpha 0.3

Secondary is a DIFFERENT-LAYER robustness comparison, not a weaker intervention.
Its measured answer-checkpoint KL is larger than primary's (0.0030 against 0.0001)
despite the smaller alpha, so "less disruptive" would be wrong.

The superseded layer-23 calibration is preserved in
out/self_focus/v3_calibration.json and explained in the corrected report: layer 23
sits above this checkpoint's shared-KV cliff and cannot reach the report at all.

THE SCORE
    score = log P(present label) - log P(absent label)
read at the forced `STATE=` prefix, before the model generates any text of its own.
The orientation is FIXED on every trial, patched and unpatched alike. Ground-truth
labels are used only in evaluation, never to construct the score: a
correct-minus-incorrect score flips sign with the true label and would manufacture
discrimination out of nothing. `test_pilot_v3.py` pins that a constant preference
gives AUROC 0.5.

ARMS
    none           no hook; also the unmodified observer with identical visible text
    sham           same hook at zero strength
    concept        the trial's own concept direction
    wrong_concept  a different real donor direction, same mechanism.
                   STILL "intervention present" for STATE. Its identification
                   target is the concept actually injected.
    random         random direction, labelled by whether functional matching
                   against the concept arm actually succeeded
    steer          no activation change; concept logit bias enabled ONLY after the
                   STATE decision has been measured and closed
"""

from __future__ import annotations

import io
import json

from experiments.self_focus import core, build_tasks as bt, interventions as iv
from experiments.self_focus.calibrate import Readout, concept_token_ids

SETTINGS = {
    "primary": {"layer": 17, "window": 64, "alpha": 0.6},
    "secondary": {"layer": 10, "window": 64, "alpha": 0.3},
}
ARMS = ["none", "sham", "concept", "wrong_concept", "random", "steer"]
CONDITIONS = ["NORMAL", "CHECK", "MEDITATE", "J_FOCUS", "LONG_NEUTRAL_CONTROL"]
OPTION_LETTERS = ["A", "B", "C", "D", "E"]
STATE_LABELS = ("Y", "N")      # distinct from the CHOICE letters, to avoid collision
READ_OFFSET = 6
STEER_BIAS = 30.0


def state_labels(concept: str, context_id: str, condition: str) -> tuple[str, str]:
    """(present_label, absent_label), counterbalanced independently of the arm."""
    digest = core.stable_id("state_schedule_v3", concept, context_id, condition)
    flip = int(digest[:8], 16) % 2 == 1
    return (STATE_LABELS[1], STATE_LABELS[0]) if flip else STATE_LABELS


def choice_options(concept: str, context_id: str, pool: list) -> list:
    """Four candidate concepts plus none, order randomised by a stable digest."""
    others = [c for c in pool if c != concept]
    digest = core.stable_id("choice_schedule_v3", concept, context_id)
    seed = int(digest[:8], 16)
    picked = [others[(seed + i * 7) % len(others)] for i in range(3)]
    picked = list(dict.fromkeys(picked))
    while len(picked) < 3:
        for c in others:
            if c not in picked:
                picked.append(c)
                break
    items = [concept] + picked[:3] + ["none"]
    order = sorted(range(len(items)), key=lambda i: core.stable_id(digest, i))
    return [(OPTION_LETTERS[slot], items[idx]) for slot, idx in enumerate(order)]


def task_block(context: dict, present: str, absent: str, options: list) -> str:
    listed = "; ".join(f"{k}={v}" for k, v in options)
    return (
        f"Respond with exactly one line and nothing else:\n"
        f"STATE=<{present} or {absent}>; ANSWER=<one lowercase word>; "
        f"CHOICE=<{'|'.join(OPTION_LETTERS)}>\n"
        f"\n"
        f"STATE: use {present} if some unexpected concept is influencing your "
        f"processing of this message, and {absent} if not. {absent} is expected most "
        f"of the time; do not guess {present} to seem observant.\n"
        f"ANSWER: {context['ask']}.\n"
        f"CHOICE: which concept, if any, was influencing you: {listed}."
    )


def cases(pool: list, contexts: list) -> list:
    out = []
    for context in contexts:
        for concept in pool:
            options = choice_options(concept, context["id"], pool)
            for condition in CONDITIONS:
                present, absent = state_labels(concept, context["id"], condition)
                block = task_block(context, present, absent, options)
                for arm in ARMS:
                    out.append({
                        "concept": concept, "context_id": context["id"],
                        "expected_answer": context["answer"], "condition": condition,
                        "arm": arm, "present_label": present, "absent_label": absent,
                        "options": options, "task_block": block,
                    })
    return out


def run_one(model, readout, prompt, *, arm, layer, alpha, direction, positions,
            concept, present_id, absent_id, option_ids, steer_concept,
            read_layer, max_new_tokens=32) -> dict:
    """One trial: patched prefill, forced STATE= readout, then greedy continuation."""
    torch = model.torch
    patch = None
    if arm in ("concept", "wrong_concept", "random"):
        patch = {"layer": layer, "direction": direction, "alpha": alpha,
                 "positions": positions}
    elif arm == "sham":
        patch = {"layer": layer, "direction": direction, "alpha": 0.0,
                 "positions": positions}

    # (1) the decision measurement, before any free generation
    state = iv.forced_prefix_readout(model, prompt, "STATE=", patch=patch,
                                     read_layers=[read_layer])
    logprobs = state["logits"].log_softmax(-1)
    probs = logprobs.exp()
    lp_present = float(logprobs[present_id])
    lp_absent = float(logprobs[absent_id])
    margin = lp_present - lp_absent          # FIXED orientation on every trial
    mass = float(probs[present_id] + probs[absent_id])
    top_id = int(state["logits"].argmax())

    # (2) greedy continuation from the same state, steering only after STATE closes
    steerer = iv.ConceptSteerer(model, steer_concept, STEER_BIAS) if arm == "steer" else None
    if steerer:
        steerer.armed = False
    gen = iv.run_trial(model, prompt, arm=("none" if arm in ("steer", "none") else arm),
                       layer=layer, alpha=(0.0 if arm == "sham" else alpha),
                       direction=direction, positions=positions,
                       max_new_tokens=max_new_tokens, steerer=steerer,
                       observe_layers=[read_layer])

    # (3) five-way identification from the generated line
    text = gen["text"]
    chosen_letter = None
    if "CHOICE=" in text:
        tail = text.split("CHOICE=")[-1].strip()
        if tail and tail[0].upper() in OPTION_LETTERS:
            chosen_letter = tail[0].upper()

    # lens evidence for the injected concept at the reporting position
    ids = option_ids["__injected__"]
    report_evidence = readout.evidence(state["reporting_residuals"][read_layer],
                                       read_layer, ids)

    return {
        "arm": arm,
        "state_logp_present": lp_present,
        "state_logp_absent": lp_absent,
        "state_margin_present_minus_absent": margin,
        "state_prob_present": float(probs[present_id]),
        "state_prob_absent": float(probs[absent_id]),
        "state_ab_combined_mass": mass,
        "state_top_token": model.tokenizer.decode([top_id]),
        "state_top_is_a_label": top_id in (present_id, absent_id),
        "greedy_state_present": bool(lp_present > lp_absent),
        "report_lens_evidence_injected": report_evidence,
        "text": text,
        "gen_token_ids": gen["gen_token_ids"],
        "chosen_letter": chosen_letter,
        "patch_alpha_applied": gen.get("patch_alpha"),
        "patched_positions": gen.get("patched_positions"),
        "steer_armed": gen.get("steer_armed"),
        "seconds": gen["seconds"],
    }


def main(out_name: str = "v3_pilot") -> int:
    cfg, conds = core.config(), core.conditions()
    core.assert_local_only(cfg)
    status = core.deadline_status()
    print(f"deadline status at start: {status}")
    if status["expired"]:
        print("deadline already expired; not starting")
        return 1

    model = core.Model(cfg)
    readout = Readout(model, cfg["lens_path"])
    budget = core.SessionBudget(reserve_minutes=30.0)
    prov = model.provenance()
    fingerprint = core.stable_id(cfg, conds, prov.get("file_hashes"), "v3_pilot")
    out_dir = core.REPO / cfg["output_dir"]

    pool = list(bt.EVAL_CONCEPTS)
    contexts = list(bt.EVAL_CONTEXTS)[:1]          # one fresh context for the pilot
    trial_cases = cases(pool, contexts)
    print(f"{len(trial_cases)} cases per setting x {len(SETTINGS)} settings "
          f"= {len(trial_cases) * len(SETTINGS)} trials")
    print(f"pool (fresh): {pool}")
    print(f"contexts (fresh): {[c['id'] for c in contexts]}")

    for setting_name, setting in SETTINGS.items():
        layer, window, alpha = setting["layer"], setting["window"], setting["alpha"]
        read_layer = min(model.n_layers - 1, layer + READ_OFFSET)
        store = core.TrialStore(out_dir / f"{out_name}_{setting_name}.jsonl", fingerprint)
        donors = iv.donor_directions(model, pool, layer, budget)
        dim = next(iter(donors["directions"].values()))["direction"].shape[0]
        print(f"\n== {setting_name}: layer {layer}, window {window}, alpha {alpha}, "
              f"read {read_layer}; {donors['donor_passes']} donor passes")

        done = 0
        for case in trial_cases:
            trial_id = core.stable_id("v3", setting_name, case["concept"],
                                      case["context_id"], case["condition"],
                                      case["arm"], fingerprint)
            if store.has(trial_id):
                continue
            if budget.exhausted():
                print("  deadline reserve reached; persisting and stopping cleanly")
                break
            prompt = core.build_prompt(case["condition"], case["task_block"], conds)
            span = model.user_content_positions(prompt, window)
            concept = case["concept"]
            wrong = [c for c in pool if c != concept][
                int(core.stable_id("wrong", concept)[:6], 16) % (len(pool) - 1)]
            injected = {"concept": concept, "wrong_concept": wrong,
                        "random": concept}.get(case["arm"], None)
            if case["arm"] == "random":
                direction = iv.random_direction(
                    model, dim, seed=int(core.stable_id("rnd3", concept,
                                                        case["condition"])[:8], 16))
            elif case["arm"] == "wrong_concept":
                direction = donors["directions"][wrong]["direction"]
            else:
                direction = donors["directions"][concept]["direction"]
            present_id = model.tokenizer(case["present_label"],
                                         add_special_tokens=False)["input_ids"][0]
            absent_id = model.tokenizer(case["absent_label"],
                                        add_special_tokens=False)["input_ids"][0]
            target = injected or concept
            option_ids = {"__injected__": concept_token_ids(model, target)["first_token_ids"]}

            with budget:
                result = run_one(
                    model, readout, prompt, arm=case["arm"], layer=layer, alpha=alpha,
                    direction=direction, positions=span["positions"], concept=concept,
                    present_id=present_id, absent_id=absent_id, option_ids=option_ids,
                    steer_concept=concept, read_layer=read_layer)

            correct_letter = next((k for k, v in case["options"] if v == target), None)
            none_letter = next((k for k, v in case["options"] if v == "none"), None)
            intervention_present = case["arm"] in ("concept", "wrong_concept", "random")
            row = {
                "trial_id": trial_id, "setting": setting_name,
                "layer": layer, "window": window, "alpha": alpha,
                "concept": concept, "injected_concept": injected,
                "wrong_concept_used": wrong if case["arm"] == "wrong_concept" else None,
                "context_id": case["context_id"], "condition": case["condition"],
                "present_label": case["present_label"], "absent_label": case["absent_label"],
                "options": case["options"], "correct_letter": correct_letter,
                "none_letter": none_letter,
                "intervention_present": intervention_present,
                "expected_answer": case["expected_answer"],
                "task_correct": case["expected_answer"] in result["text"].lower(),
                "identified": (result["chosen_letter"] == correct_letter
                               and correct_letter is not None),
                "said_none": result["chosen_letter"] == none_letter,
                "naive_substring_hit": bt.naive_substring_hit(result["text"], target),
                "patch_span": span, "read_layer": read_layer,
                **result,
            }
            store.write(row)
            done += 1
            if done % 30 == 0:
                print(f"  {done}/{len(trial_cases)}  {budget.summary()['remaining_minutes']:.0f} min left")
        store.close()
        print(f"  {setting_name}: {done} new trials")

    meta = {"protocol_version": "self_focus_v3_pilot",
            "status": "amended exploratory protocol",
            "frozen_settings": SETTINGS,
            "secondary_note": "different-layer robustness comparison, NOT a weaker "
                              "intervention: its answer-checkpoint KL is larger than "
                              "primary's (0.0030 vs 0.0001) despite the smaller alpha",
            "superseded": "layer-23 settings; see v3_calibration.json and the "
                          "corrected calibration report",
            "development_data": {"concepts": bt.DEV_CONCEPTS,
                                 "contexts": [c["id"] for c in bt.DEV_CONTEXTS]},
            "pilot_data": {"concepts": pool, "contexts": [c["id"] for c in contexts]},
            "arms": ARMS, "conditions": CONDITIONS,
            "score": "log P(present label) - log P(absent label), fixed orientation",
            "provenance": prov, "provenance_fingerprint": fingerprint,
            "deadline": core.deadline_status(),
            "external_compute_spend_usd": 0.0}
    with io.open(out_dir / f"{out_name}_meta.json", "w", encoding="utf-8") as fh:
        json.dump(meta, fh, indent=2)
    print(f"\nwrote {out_dir / f'{out_name}_meta.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
