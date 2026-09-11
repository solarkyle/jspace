"""Six-condition QA error-awareness study. Local only.

Two phases, and the protocol is frozen between them:

    dev   20 items x 6 conditions = 120 trials   (inspect, fix implementation only)
    eval  80 items x 6 conditions = 480 trials   (scored once, never tuned on)

Families never span the splits, and the splits come from separate family seeds.

ANSWER-ONSET MEASUREMENT
The option distribution is read at a forced `CHOICE=` prefix, before the model has
generated any answer token. That distribution, and the internal features recorded at
the same position, are the only inputs an onset detector may use. The model's own
stated CONFIDENCE is generated AFTER the choice, so it is recorded and reported
separately but never fed to the onset detector.

Grading is strict designated-field parsing. No substring matching: the v3 pilot used
substring matching for task correctness and its figures were upper bounds as a result.
Invalid and truncated outputs are counted, never dropped.
"""

from __future__ import annotations

import io
import json

from experiments.self_focus import core, interventions as iv, qa_tasks as qt
from experiments.self_focus.calibrate import Readout

CONDITIONS = ["NORMAL", "CHECK", "MEDITATE", "J_INFO", "J_FOCUS", "J_ROLEPLAY"]
READ_LAYERS = [10, 17, 23, 30]
LENS_LAYER = 17
MAX_NEW = 24


def onset_features(model, readout, state, option_ids) -> dict:
    """Everything available at the forced CHOICE= position, and nothing later."""
    torch = model.torch
    logits = state["logits"]
    picked = torch.tensor([float(logits[i]) for i in option_ids])
    probs = picked.softmax(0)
    ordered = probs.sort(descending=True).values
    entropy = float(-(probs * probs.clamp_min(1e-12).log()).sum())
    feats = {
        "opt_p_max": float(ordered[0]),
        "opt_margin_top1_top2": float(ordered[0] - ordered[1]),
        "opt_entropy": entropy,
    }
    for layer, residual in state["reporting_residuals"].items():
        feats[f"resid_norm_L{layer}"] = float(residual.norm())
    if LENS_LAYER in state["reporting_residuals"]:
        lens_logits = readout.lens.transport(
            state["reporting_residuals"][LENS_LAYER].unsqueeze(0).float(), LENS_LAYER)
        lp = readout.wrapper.unembed(lens_logits).float().cpu()[0].log_softmax(-1)
        p = lp.exp()
        feats["lens_entropy_L17"] = float(-(p * lp).sum())
        feats["lens_top1_p_L17"] = float(p.max())
    return feats


def run_split(model, readout, conds, items, split, store, budget) -> int:
    done = 0
    option_id_cache = {}
    for item in items:
        block = qt.qa_task_block(item)
        for condition in CONDITIONS:
            trial_id = core.stable_id("qa", split, item["item_id"], condition)
            if store.has(trial_id):
                continue
            if budget.exhausted():
                print("    deadline reserve reached; stopping cleanly")
                return done
            prompt = core.build_prompt(condition, block, conds)
            if "letters" not in option_id_cache:
                option_id_cache["letters"] = [
                    model.tokenizer(k, add_special_tokens=False)["input_ids"][0]
                    for k in qt.OPTION_LETTERS]
            option_ids = option_id_cache["letters"]

            with budget:
                state = iv.forced_prefix_readout(model, prompt, "CHOICE=",
                                                 read_layers=READ_LAYERS)
                feats = onset_features(model, readout, state, option_ids)
                gen = model.generate(prompt, MAX_NEW)

            parsed = qt.parse_qa(gen["text"])
            chosen = parsed["choice"]
            correct = (chosen == item["correct_letter"]) if parsed["valid"] else None
            # the onset distribution's own top option, independent of generation
            top_letter = qt.OPTION_LETTERS[
                max(range(5), key=lambda i: state["logits"][option_ids[i]])]
            store.write({
                "trial_id": trial_id, "split": split, "condition": condition,
                "item_id": item["item_id"], "family_id": item["family_id"],
                "item_type": item["item_type"], "answerable": item["answerable"],
                "correct_letter": item["correct_letter"],
                "correct_value": item["correct_value"],
                "options": item["options"],
                "parsed_valid": parsed["valid"], "parse_reason": parsed["reason"],
                "chosen_letter": chosen, "stated_confidence": parsed["confidence"],
                "is_correct": correct,
                "onset_top_letter": top_letter,
                "onset_top_is_correct": top_letter == item["correct_letter"],
                "features": feats,
                "n_prompt_tokens": gen["n_prompt_tokens"],
                "text": gen["text"], "seconds": gen["seconds"],
            })
            done += 1
            if done % 40 == 0:
                print(f"    {done} trials, {budget.summary()['remaining_minutes']:.0f} min left")
    return done


def main() -> int:
    cfg, conds = core.config(), core.conditions()
    core.assert_local_only(cfg)
    status = core.deadline_status()
    print(f"deadline: {status}")
    if status["expired"]:
        print("deadline expired; not starting")
        return 1

    model = core.Model(cfg)
    readout = Readout(model, cfg["lens_path"])
    budget = core.SessionBudget(reserve_minutes=30.0)
    prov = model.provenance()
    fingerprint = core.stable_id(cfg, conds, prov.get("file_hashes"), "qa_v1")
    out_dir = core.REPO / cfg["output_dir"]

    data = qt.build()
    print(f"dev {len(data['dev'])} items x {len(CONDITIONS)} conditions = "
          f"{len(data['dev']) * len(CONDITIONS)} trials")
    print(f"eval {len(data['eval'])} items x {len(CONDITIONS)} conditions = "
          f"{len(data['eval']) * len(CONDITIONS)} trials")
    with io.open(out_dir / "qa_items.json", "w", encoding="utf-8") as fh:
        json.dump(data, fh, indent=2)

    for split in ("dev", "eval"):
        store = core.TrialStore(out_dir / f"qa_{split}.jsonl", fingerprint)
        print(f"\n== {split}")
        n = run_split(model, readout, conds, data[split], split, store, budget)
        store.close()
        print(f"  {split}: {n} new trials")
        if split == "dev":
            # protocol freeze: prompts, grading, features and settings are fixed here
            freeze = {"frozen_at": "after development, before evaluation",
                      "conditions": CONDITIONS, "schema": qt.QA_SCHEMA,
                      "read_layers": READ_LAYERS, "lens_layer": LENS_LAYER,
                      "max_new_tokens": MAX_NEW,
                      "grading": "strict designated-field parsing; no substring matching",
                      "onset_rule": "option distribution and internal features at the "
                                    "forced CHOICE= position only; stated CONFIDENCE is "
                                    "generated later and is never an onset feature",
                      "item_hash": core.stable_id(data)}
            with io.open(out_dir / "qa_protocol_freeze.json", "w", encoding="utf-8") as fh:
                json.dump(freeze, fh, indent=2)
            print(f"  froze protocol -> qa_protocol_freeze.json")

    meta = {"protocol_version": "qa_v1", "conditions": CONDITIONS,
            "provenance": prov, "provenance_fingerprint": fingerprint,
            "deadline": core.deadline_status(),
            "external_compute_spend_usd": 0.0}
    with io.open(out_dir / "qa_meta.json", "w", encoding="utf-8") as fh:
        json.dump(meta, fh, indent=2)
    print("\nwrote qa_meta.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
