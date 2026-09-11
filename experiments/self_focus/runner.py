"""Self-Focus Lab runner. Local only.

Commands:
    preflight            verify model, template, layers, memory, provenance; no generation
    demo                 the first deliverable: six free reports + fixed-continuation readouts
    qa-dev               20 questions x 6 conditions
    qa-eval              80 questions x 6 conditions (only after the protocol is frozen)
    intervention-pilot   hidden-concept injection pilot
    report               build the Markdown and HTML reports from saved trials

Every command accepts --dry-run, which prints exact case counts, model identity,
limits, output paths, and confirms execution is local, without loading the model.
"""

from __future__ import annotations

import argparse
import io
import json
import sys
from pathlib import Path

from experiments.self_focus import core


def _out_dir(cfg: dict) -> Path:
    path = core.REPO / cfg["output_dir"]
    path.mkdir(parents=True, exist_ok=True)
    return path


def _fingerprint(cfg: dict, conds: dict, provenance: dict) -> str:
    """Identity of the whole run: config, prompts, and what actually loaded."""
    return core.stable_id(
        cfg, conds,
        provenance.get("loaded_from"), provenance.get("precision"),
        provenance.get("architecture"), provenance.get("n_layers"),
        provenance.get("file_hashes"),
    )


def _dry_run(command: str, cfg: dict, conds: dict, cases: int, detail: str) -> int:
    out = _out_dir(cfg)
    print("DRY RUN, no model loaded, no compute used")
    print(f"  command            {command}")
    print(f"  protocol_version   {cfg['protocol_version']} / {conds['protocol_version']}")
    print(f"  model_id           {cfg['model_id']}")
    print(f"  loaded from        {cfg.get('local_model_path') or cfg['model_id']}")
    print(f"  precision          {cfg['precision']} (held constant; never pooled with other precisions)")
    print(f"  conditions         {', '.join(conds['conditions'])}")
    print(f"  cases to run       {cases}")
    if detail:
        print(f"  breakdown          {detail}")
    print(f"  max layers         {cfg['max_recorded_layers']}")
    print(f"  max positions      {cfg['max_recorded_positions']}")
    print(f"  prompt limit       {cfg['max_prompt_tokens']} tokens (oversized prompts REJECTED)")
    print(f"  compute ceiling    {cfg['compute_ceiling_minutes']} minutes")
    print(f"  output dir         {out}")
    print(f"  execution          LOCAL ONLY  (allow_remote={cfg['allow_remote']}, "
          f"allow_paid_judge={cfg['allow_paid_judge']}, "
          f"cloud_spend_authorization_usd={cfg['cloud_spend_authorization_usd']})")
    core.assert_local_only(cfg)
    print("  local-only assertion passed")
    return 0


# --------------------------------------------------------------------------- #
# preflight
# --------------------------------------------------------------------------- #

def cmd_preflight(args) -> int:
    cfg, conds = core.config(), core.conditions()
    if args.dry_run:
        return _dry_run("preflight", cfg, conds, 0, "no trials; verification only")

    core.assert_local_only(cfg)
    print("loading model ...")
    model = core.Model(cfg)
    prov = model.provenance()
    band, lens_info = core.lens_band_from(cfg["lens_path"], model.n_layers)
    layers = core.select_layers(model.n_layers, cfg, band)

    # the rendered template must be real and stable, not silently fallen back
    rendered = model.render("Say OK.")
    ids = model.encode(rendered)

    checks = {
        "model_loaded": True,
        "architecture": prov["architecture"],
        "n_layers": prov["n_layers"],
        "block_path": prov["block_path"],
        "device": prov["device"],
        "precision": prov["precision"],
        "template_rendered": rendered,
        "template_token_count": int(ids.shape[1]),
        "lens": lens_info,
        "layer_selection": layers,
        "module_names": [model.module_name(l) for l in layers["layers"]],
        "gpu": prov["gpu"],
    }
    fingerprint = _fingerprint(cfg, conds, prov)
    payload = {"provenance": prov, "checks": checks, "provenance_fingerprint": fingerprint,
               "protocol_version": cfg["protocol_version"]}
    out = _out_dir(cfg) / "preflight.json"
    with io.open(out, "w", encoding="utf-8") as fh:
        json.dump(payload, fh, indent=2)

    print(f"  architecture    {prov['architecture']}  ({prov['n_layers']} layers at {prov['block_path']})")
    print(f"  device          {prov['device']}   precision {prov['precision']}")
    print(f"  GPU             {prov['gpu'].get('name')}  free {prov['gpu'].get('free_gb')} GB "
          f"of {prov['gpu'].get('total_gb')} GB, peak alloc {prov['gpu'].get('peak_allocated_gb')} GB")
    print(f"  lens            available={lens_info.get('available')} "
          f"{'band=' + str(lens_info.get('band')) if lens_info.get('available') else lens_info.get('reason')}")
    print(f"  layers          {layers['layers']}  ({layers['basis']}, zero-based)")
    print(f"  modules         {checks['module_names']}")
    print(f"  template        {rendered!r}")
    print(f"  fingerprint     {fingerprint}")
    print(f"wrote {out}")
    return 0


# --------------------------------------------------------------------------- #
# demo: the first deliverable
# --------------------------------------------------------------------------- #

def cmd_demo(args) -> int:
    cfg, conds = core.config(), core.conditions()
    names = list(conds["conditions"])
    n_reports = len(names)
    n_traces = len(conds["fixed_continuations"]) * len(names)
    if args.dry_run:
        return _dry_run("demo", cfg, conds, n_reports + n_traces,
                        f"{n_reports} free reports (<=128 tokens) + {n_traces} "
                        f"teacher-forced fixed-continuation traces")

    core.assert_local_only(cfg)
    out_dir = _out_dir(cfg)
    budget = core.Budget(cfg["compute_ceiling_minutes"])
    model = core.Model(cfg)
    prov = model.provenance()
    band, lens_info = core.lens_band_from(cfg["lens_path"], model.n_layers)
    layers = core.select_layers(model.n_layers, cfg, band)["layers"]
    fingerprint = _fingerprint(cfg, conds, prov)

    # ---- A. six free reports, exploratory only --------------------------- #
    store = core.TrialStore(out_dir / "demo_reports.jsonl", fingerprint)
    task_block = conds["free_report_suffix"]
    print(f"A. six free reports (max {cfg['max_new_tokens']['free_report']} tokens)")
    for name in names:
        trial_id = core.stable_id("demo_report", cfg["protocol_version"], name,
                                  task_block, fingerprint)
        if store.has(trial_id):
            print(f"  {name:11s} already recorded, skipping")
            continue
        if budget.exhausted():
            print("  compute ceiling reached; stopping cleanly")
            break
        prompt = core.build_prompt(name, task_block, conds)
        with budget:
            result = model.generate(prompt, cfg["max_new_tokens"]["free_report"])
        store.write({
            "trial_id": trial_id, "kind": "demo_report", "condition": name,
            "prompt": prompt, **result,
        })
        print(f"  {name:11s} {result['n_prompt_tokens']:4d} prompt tok, "
              f"{len(result['gen_token_ids']):3d} gen tok, {result['seconds']:.1f}s")
    store.close()

    # ---- B. fixed-continuation readouts ---------------------------------- #
    store = core.TrialStore(out_dir / "demo_traces.jsonl", fingerprint)
    print(f"\nB. teacher-forced fixed continuation, layers {layers}")
    for p_index, passage in enumerate(conds["fixed_continuations"]):
        # ONE tokenization of the passage, reused for every condition, so the
        # forced continuation token ids are identical across conditions
        cont_ids = model.tokenizer(passage, add_special_tokens=False)["input_ids"]
        for name in names:
            trial_id = core.stable_id("demo_trace", cfg["protocol_version"], name,
                                      p_index, tuple(cont_ids), tuple(layers), fingerprint)
            if store.has(trial_id):
                print(f"  passage {p_index} {name:11s} already recorded, skipping")
                continue
            if budget.exhausted():
                print("  compute ceiling reached; stopping cleanly")
                break
            task_block = f"{conds['continuation_task']}\n\n{passage}"
            prompt = core.build_prompt(name, task_block, conds)
            with budget:
                cap = model.capture(prompt, layers, forced_continuation_ids=cont_ids)
            row = {
                "trial_id": trial_id, "kind": "demo_trace", "condition": name,
                "passage_index": p_index, "passage": passage,
                "label": "teacher-forced fixed continuation",
                "continuation_token_ids": cont_ids,
                "prompt": prompt,
                "n_prompt_tokens": cap["n_prompt_tokens"],
                "total_tokens": cap["total_tokens"],
                "seconds": cap["seconds"],
                "layers": {
                    str(l): {"module": cap["layers"][l]["module"],
                             "positions": cap["layers"][l]["positions"],
                             "residual_norms": cap["layers"][l]["residual_norms"]}
                    for l in cap["layers"]
                },
            }
            # full residual vectors go to a separate, ignored file
            with io.open(out_dir / "demo_residuals.jsonl", "a", encoding="utf-8") as fh:
                fh.write(json.dumps({
                    "trial_id": trial_id, "condition": name, "passage_index": p_index,
                    "layers": {str(l): cap["layers"][l]["residuals"] for l in cap["layers"]},
                }) + "\n")
            store.write(row)
            print(f"  passage {p_index} {name:11s} {cap['n_prompt_tokens']:4d}+"
                  f"{len(cont_ids):3d} tok, {len(cap['layers'])} layers, {cap['seconds']:.2f}s")
    store.close()

    meta = {
        "protocol_version": cfg["protocol_version"],
        "provenance": prov, "provenance_fingerprint": fingerprint,
        "lens": lens_info, "layers": layers,
        "budget": budget.summary(),
        "external_compute_spend_usd": 0.0,
    }
    with io.open(out_dir / "demo_meta.json", "w", encoding="utf-8") as fh:
        json.dump(meta, fh, indent=2)
    print(f"\nbudget {budget.summary()}")
    print(f"wrote {out_dir / 'demo_meta.json'}")
    return 0


# --------------------------------------------------------------------------- #
# hidden-intervention pilot
# --------------------------------------------------------------------------- #

CANDIDATE_LAYERS = [17, 23]
CANDIDATE_STRENGTHS = [0.15, 0.30]


def _pilot_setup(cfg, conds):
    from experiments.self_focus import build_tasks as bt
    ivc = cfg["intervention"]
    cases = bt.pilot_cases(ivc["conditions"], ivc["arms"], int(ivc["n_contexts"]))
    return ivc, cases


def cmd_select(args) -> int:
    """Inspect at most two layers and two strengths, count every attempt, freeze."""
    import json as _json
    from experiments.self_focus import build_tasks as bt, interventions as iv

    cfg, conds = core.config(), core.conditions()
    ivc = cfg["intervention"]
    n_attempts = len(CANDIDATE_LAYERS) * len(CANDIDATE_STRENGTHS)
    if args.dry_run:
        return _dry_run("select-intervention", cfg, conds, n_attempts,
                        f"{len(CANDIDATE_LAYERS)} layers x {len(CANDIDATE_STRENGTHS)} "
                        f"strengths, inspected on development concepts only")

    core.assert_local_only(cfg)
    out_dir = _out_dir(cfg)
    budget = core.Budget(cfg["compute_ceiling_minutes"])
    model = core.Model(cfg)
    dev = list(ivc["dev_concepts"])[:2]
    context = bt.CONTEXTS[0]
    yes, no = bt.yes_no_labels(dev[0], context["id"], "NORMAL")
    task = bt.recipient_task_block(context, yes, no)
    prompt = core.build_prompt("NORMAL", task, conds)
    span = model.user_content_positions(prompt, int(ivc["patch_window"]))
    observe = [30]   # downstream of both candidate layers

    attempts = []
    for layer in CANDIDATE_LAYERS:
        donors = iv.donor_directions(model, ivc["concepts"], layer, budget)
        for alpha in CANDIDATE_STRENGTHS:
            valid, changed, deltas = 0, 0, []
            for concept in dev:
                if budget.exhausted():
                    break
                base = iv.run_trial(
                    model, prompt, arm="none", layer=layer, alpha=0.0,
                    direction=donors["directions"][concept]["direction"],
                    positions=span["positions"],
                    max_new_tokens=cfg["max_new_tokens"]["intervention_report"],
                    observe_layers=observe)
                with budget:
                    pass
                hit = iv.run_trial(
                    model, prompt, arm="concept", layer=layer, alpha=alpha,
                    direction=donors["directions"][concept]["direction"],
                    positions=span["positions"],
                    max_new_tokens=cfg["max_new_tokens"]["intervention_report"],
                    observe_layers=observe)
                parsed = bt.parse_report(hit["text"], yes, no)
                valid += int(parsed["valid"])
                a = base["observed_last_token_norms"].get("30")
                b = hit["observed_last_token_norms"].get("30")
                if a and b:
                    deltas.append(abs(b - a) / a)
                changed += int(hit["text"] != base["text"])
            attempts.append({
                "layer": layer, "alpha": alpha, "n_dev_trials": len(dev),
                "format_valid": valid, "output_changed": changed,
                "mean_rel_downstream_delta": (sum(deltas) / len(deltas)) if deltas else None,
                "donor_passes": donors["donor_passes"],
            })
            print(f"  layer {layer} alpha {alpha:.2f}: format_valid {valid}/{len(dev)}, "
                  f"output_changed {changed}/{len(dev)}, downstream delta "
                  f"{attempts[-1]['mean_rel_downstream_delta']}")

    # prefer an attempt that keeps the format AND moves the computation
    usable = [a for a in attempts if a["format_valid"] == a["n_dev_trials"]
              and (a["mean_rel_downstream_delta"] or 0) > 0]
    choice = max(usable, key=lambda a: a["mean_rel_downstream_delta"]) if usable else None
    payload = {"candidate_layers": CANDIDATE_LAYERS,
               "candidate_strengths": CANDIDATE_STRENGTHS,
               "attempts_inspected": len(attempts), "attempts": attempts,
               "selected": {"layer": choice["layer"], "alpha": choice["alpha"]} if choice else None,
               "selection_rule": "largest downstream change among attempts that kept "
                                 "every development output parseable",
               "budget": budget.summary()}
    with io.open(out_dir / "intervention_selection.json", "w", encoding="utf-8") as fh:
        _json.dump(payload, fh, indent=2)
    print(f"\nattempts inspected: {len(attempts)} (ceiling 2 layers x 2 strengths)")
    print(f"selected: {payload['selected']}")
    print(f"wrote {out_dir / 'intervention_selection.json'}")
    if choice is None:
        print("NO usable setting found; the pilot must not run on an unfrozen value")
        return 1
    return 0


def cmd_pilot(args) -> int:
    from experiments.self_focus import build_tasks as bt, interventions as iv

    cfg, conds = core.config(), core.conditions()
    ivc, cases = _pilot_setup(cfg, conds)
    if args.dry_run:
        import collections
        by_arm = collections.Counter(c["arm"] for c in cases)
        return _dry_run("intervention-pilot", cfg, conds, len(cases),
                        f"{len(ivc['concepts'])} concepts x {len(ivc['conditions'])} "
                        f"conditions x {len(ivc['arms'])} arms x {ivc['n_contexts']} "
                        f"context = {len(cases)}; by arm {dict(by_arm)}")

    core.assert_local_only(cfg)
    out_dir = _out_dir(cfg)
    sel_path = out_dir / "intervention_selection.json"
    if not sel_path.exists():
        print("no frozen layer/strength; run select-intervention first")
        return 1
    selection = core.load_json(sel_path)
    if not selection.get("selected"):
        print("selection file contains no usable setting; refusing to run")
        return 1
    layer = int(selection["selected"]["layer"])
    alpha = float(selection["selected"]["alpha"])

    budget = core.Budget(cfg["compute_ceiling_minutes"])
    model = core.Model(cfg)
    prov = model.provenance()
    fingerprint = _fingerprint(cfg, conds, prov)
    store = core.TrialStore(out_dir / "intervention.jsonl", fingerprint)

    donors = iv.donor_directions(model, ivc["concepts"], layer, budget)
    print(f"donor directions: {len(donors['directions'])} concepts, "
          f"{donors['donor_passes']} forward passes, layer {layer}, alpha {alpha}")
    dim = next(iter(donors["directions"].values()))["direction"].shape[0]
    observe = [min(model.n_layers - 1, layer + 6)]

    done = 0
    for case in cases:
        trial_id = core.stable_id("intervention", cfg["protocol_version"],
                                  case["concept"], case["context_id"],
                                  case["condition"], case["arm"], layer, alpha,
                                  fingerprint)
        if store.has(trial_id):
            continue
        if budget.exhausted():
            print("compute ceiling reached; persisting and stopping cleanly")
            break
        prompt = core.build_prompt(case["condition"], case["task_block"], conds)
        span = model.user_content_positions(prompt, int(ivc["patch_window"]))
        concept = case["concept"]
        arm = case["arm"]
        if arm == "concept":
            direction = donors["directions"][concept]["direction"]
        elif arm == "random":
            direction = iv.random_direction(
                model, dim, seed=int(core.stable_id("rand", concept, case["condition"])[:8], 16))
        else:
            direction = donors["directions"][concept]["direction"]
        steerer = (iv.ConceptSteerer(model, concept, float(ivc["steer_bias"]))
                   if arm == "steer" else None)
        with budget:
            result = iv.run_trial(
                model, prompt, arm=arm, layer=layer, alpha=alpha,
                direction=direction, positions=span["positions"],
                max_new_tokens=cfg["max_new_tokens"]["intervention_report"],
                steerer=steerer, observe_layers=observe)
        parsed = bt.parse_report(result["text"], case["yes_label"], case["no_label"])
        row = {
            "trial_id": trial_id, "kind": "intervention",
            "concept": concept, "context_id": case["context_id"],
            "condition": case["condition"], "arm": arm,
            "yes_label": case["yes_label"], "no_label": case["no_label"],
            "expected_answer": case["expected_answer"],
            "patch_span": span, "observe_layers": observe,
            "parsed": parsed,
            "identified": bt.identification_correct(parsed, concept),
            "naive_substring_hit": bt.naive_substring_hit(result["text"], concept),
            "task_correct": (parsed.get("answer") == case["expected_answer"]),
            **{k: v for k, v in result.items() if k != "rendered_prompt"},
        }
        store.write(row)
        done += 1
        if done % 25 == 0:
            print(f"  {done}/{len(cases)} trials, {budget.summary()['used_minutes']:.1f} min used")
    store.close()

    meta = {"protocol_version": cfg["protocol_version"], "provenance": prov,
            "provenance_fingerprint": fingerprint, "selection": selection["selected"],
            "donor_hash": donors.get("hash"), "donor_passes": donors["donor_passes"],
            "arms": iv.ARM_NOTES, "budget": budget.summary(),
            "external_compute_spend_usd": 0.0}
    with io.open(out_dir / "intervention_meta.json", "w", encoding="utf-8") as fh:
        json.dump(meta, fh, indent=2)
    print(f"\n{done} new trials. budget {budget.summary()}")
    return 0


# --------------------------------------------------------------------------- #

def main(argv=None) -> int:
    parser = argparse.ArgumentParser(prog="self_focus")
    sub = parser.add_subparsers(dest="command", required=True)
    for name, fn in (("preflight", cmd_preflight), ("demo", cmd_demo)):
        p = sub.add_parser(name)
        p.add_argument("--dry-run", action="store_true")
        p.set_defaults(func=fn)
    for name, fn in (("select-intervention", cmd_select), ("intervention-pilot", cmd_pilot)):
        p = sub.add_parser(name)
        p.add_argument("--dry-run", action="store_true")
        p.set_defaults(func=fn)
    for name in ("qa-dev", "qa-eval", "report"):
        p = sub.add_parser(name)
        p.add_argument("--dry-run", action="store_true")
        p.set_defaults(func=lambda a, n=name: _not_yet(n))
    args = parser.parse_args(argv)
    return args.func(args)


def _not_yet(name: str) -> int:
    print(f"{name}: not implemented in this commit. "
          f"Implemented so far: preflight, demo, select-intervention, "
          f"intervention-pilot.")
    return 2


if __name__ == "__main__":
    sys.exit(main())
