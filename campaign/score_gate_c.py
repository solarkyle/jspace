"""Score Gate C: does a mid-answer prefix preserve the full-answer increment?

Registered criterion (campaign/PREREG_STAGE1.md):

    the risk score from the 50 percent answer-prefix preserves at least 90
    percent of the full-answer LODO AUROC increment. Falsified if the 50 percent
    prefix retains < 90 percent.

THIS IS A REDUCED-SCOPE FOLLOW-UP, NOT THE ORIGINAL PLAN EXECUTED UNCHANGED.
The differences are deliberate and are reported in the output payload:

  * Sources are restricted to grader_type=exact, so labels are free and need no
    judge. That drops halubench, medhallu and esconv from the original coverage.
  * Rows are restricted to answers of at least 32 tokens. Early warning is
    undefined where there is no room to warn: legal_hallucinations never reaches
    8 tokens and trivia_qa's median answer is about 4.
  * Answers are freshly generated and freshly graded. The July generations did
    not reproduce in a September comparison, so a July label does not describe a
    September answer. Why they diverged has not been isolated.
  * The primary model is logistic regression, matching the 2026-09-05 audit.
    LightGBM is reported beside it when installed, since the campaign's frozen
    classifier was LightGBM and the two were effectively tied at Stage 1.

"Increment" means workspace-over-logprob, so the ratio compares two
self-consistent pairs:

    full   increment = AUROC(full LP + full workspace)    - AUROC(full LP)
    prefix increment = AUROC(prefix LP + workspace @ 50%) - AUROC(prefix LP)
    ratio            = prefix increment / full increment

The two logprob baselines are FEATURE-MATCHED one-to-one: first, mean, min and a
length term on each side. An earlier version gave the prefix arm an extra
last-token probability with no full-answer counterpart, which makes the arms
differ by more than timing. That feature is still reported, as a separate
`prefix_plus` arm, rather than being folded into the primary comparison.

Fixed absolute-index checkpoints are scored too. The 50 percent checkpoint needs
the eventual answer length, so it is an oracle observation; token 8 and token 16
are what a live monitor could actually read.

Everything is leave-one-dataset-out. Pooled numbers are not reported: Stage 1
measured identity leakage of 0.811 against a 0.185 majority, and the registered
interpretation says pooled numbers are not to be trusted.

Usage:
    python -m campaign.score_gate_c --input out/campaign/gatec_graded.jsonl \
        --out out/campaign/gate_c.json
"""

import argparse
import collections
import io
import json
import os

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

SEED = 20260910
BOOTSTRAP = 500

WS_SCALARS = ("ignition_frac", "ignition_depth", "mean_log_rank_answer",
              "band_agreement", "mean_entropy", "best_hedge_rank_log")
# One-to-one with FULL_LP: first, mean, min, length.
PFX_LP = ("pfxlp_first", "pfxlp_mean", "pfxlp_min", "pfxlp_tokens_so_far")
PFX_LP_EXTRA = PFX_LP + ("pfxlp_last",)
FULL_LP = ("bl_first_token_logprob", "bl_mean_logprob", "bl_min_logprob",
           "bl_answer_len")

MIN_TEST_ROWS = 30
MIN_MINORITY = 5


def models() -> dict:
    out = {"logistic": lambda: make_pipeline(
        StandardScaler(),
        LogisticRegression(C=1.0, max_iter=2000, class_weight="balanced",
                           random_state=SEED))}
    try:
        from lightgbm import LGBMClassifier
    except Exception:
        return out
    out["lightgbm"] = lambda: LGBMClassifier(
        n_estimators=200, num_leaves=31, min_child_samples=20, subsample=0.9,
        colsample_bytree=0.9, class_weight="balanced", random_state=0,
        deterministic=True, verbose=-1)
    return out


def _checkpoint(row: dict, frac: float):
    for p in row.get("prefix_workspace_features") or []:
        if abs(float(p.get("frac", -1)) - frac) < 1e-9:
            return p
    return None


def _fixed(row: dict, index: int):
    for p in row.get("fixed_checkpoint_features") or []:
        if int(p.get("token_index", -1)) == index:
            return p
    return None


def arms(row: dict) -> dict:
    """Feature vectors per arm, or {} if a required checkpoint is missing."""
    mid = _checkpoint(row, 0.5)
    end = _checkpoint(row, 1.0)
    if mid is None or end is None:
        return {}
    lp_full = row.get("logprob_features") or {}
    if not all(k in lp_full for k in FULL_LP):
        return {}
    try:
        out = {
            "full_lp": [float(lp_full[k]) for k in FULL_LP],
            "full_lp_ws": [float(lp_full[k]) for k in FULL_LP]
                          + [float(end[k]) for k in WS_SCALARS],
            "prefix_lp": [float(mid[k]) for k in PFX_LP],
            "prefix_lp_ws": [float(mid[k]) for k in PFX_LP]
                            + [float(mid[k]) for k in WS_SCALARS],
            "prefix_plus_lp": [float(mid[k]) for k in PFX_LP_EXTRA],
            "prefix_plus_lp_ws": [float(mid[k]) for k in PFX_LP_EXTRA]
                                 + [float(mid[k]) for k in WS_SCALARS],
        }
    except KeyError:
        return {}
    for index in (8, 16):
        ck = _fixed(row, index)
        if ck is None or not all(k in ck for k in PFX_LP + WS_SCALARS):
            continue
        out[f"fixed{index}_lp"] = [float(ck[k]) for k in PFX_LP]
        out[f"fixed{index}_lp_ws"] = ([float(ck[k]) for k in PFX_LP]
                                      + [float(ck[k]) for k in WS_SCALARS])
    return out


def lodo(rows: list, arm: str, make_model) -> dict:
    """Leave-one-dataset-out AUROC per source, with a row-level bootstrap."""
    by_source = collections.defaultdict(list)
    for row in rows:
        if arm in row["_arms"]:
            by_source[row["source_dataset"]].append(row)

    per_source, skipped, draws = {}, {}, {}
    for held in sorted(by_source):
        test = by_source[held]
        train = [r for s, rs in by_source.items() if s != held for r in rs]
        y_test = np.array([int(r["_label"]) for r in test])
        if (len(test) < MIN_TEST_ROWS
                or min(int(y_test.sum()), int((1 - y_test).sum())) < MIN_MINORITY):
            skipped[held] = (f"n={len(test)} pos={int(y_test.sum())} "
                             f"neg={int((1 - y_test).sum())} below floor")
            continue
        X_train = np.array([r["_arms"][arm] for r in train])
        y_train = np.array([int(r["_label"]) for r in train])
        X_test = np.array([r["_arms"][arm] for r in test])
        scores = make_model().fit(X_train, y_train).predict_proba(X_test)[:, 1]
        per_source[held] = float(roc_auc_score(y_test, scores))

        rng = np.random.default_rng(SEED)
        boot = []
        for _ in range(BOOTSTRAP):
            idx = rng.integers(0, len(y_test), len(y_test))
            if len(set(y_test[idx])) < 2:
                continue
            boot.append(roc_auc_score(y_test[idx], scores[idx]))
        draws[held] = boot
    return {"per_source": per_source, "skipped": skipped,
            "mean": float(np.mean(list(per_source.values()))) if per_source else None,
            "_draws": draws}


def increment_ci(lo_arm: dict, hi_arm: dict) -> dict:
    """CI on the mean increment, resampling rows inside each held-out source.

    This reflects row-level noise only. With three or four sources there is no
    usable source-level distribution, so between-source variation is NOT in this
    interval and the mean stays a small-sample estimate.
    """
    shared = sorted(set(lo_arm["_draws"]) & set(hi_arm["_draws"]))
    if not shared:
        return {}
    width = min(min(len(lo_arm["_draws"][s]), len(hi_arm["_draws"][s])) for s in shared)
    if width < 50:
        return {}
    diffs = [float(np.mean([hi_arm["_draws"][s][b] - lo_arm["_draws"][s][b]
                            for s in shared])) for b in range(width)]
    return {"increment_p2.5": float(np.percentile(diffs, 2.5)),
            "increment_p97.5": float(np.percentile(diffs, 97.5)),
            "bootstrap_draws": width, "sources_in_ci": shared}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    rows, unresolved, dropped = [], [], collections.Counter()
    excluded_by_source = collections.Counter()
    excluded_methods = collections.Counter()
    with io.open(args.input, encoding="utf-8") as fh:
        for line in fh:
            if not line.strip():
                continue
            row = json.loads(line)
            a = arms(row)
            if not a:
                dropped["missing_checkpoint"] += 1
                continue
            row["_arms"] = a
            # Label from the FRESH grade of THIS run's answer. The manifest also
            # carries the July label, but regeneration reproduced almost none of
            # the July answers, so that label describes different text.
            grade = row.get("deterministic_grade") or {}
            if grade.get("correct") is None:
                # Unresolved, not discarded. Kept so the verdict can be bounded
                # against every possible assignment of these rows.
                dropped["unlabeled_needs_judge"] += 1
                excluded_by_source[row["source_dataset"]] += 1
                excluded_methods[grade.get("method", "unknown")] += 1
                unresolved.append(row)
                continue
            row["_label"] = not bool(grade["correct"])
            rows.append(row)

    arm_names = ["full_lp", "full_lp_ws", "prefix_lp", "prefix_lp_ws",
                 "prefix_plus_lp", "prefix_plus_lp_ws",
                 "fixed8_lp", "fixed8_lp_ws", "fixed16_lp", "fixed16_lp_ws"]

    payload = {
        "gate": "C",
        "scope": "reduced-scope follow-up; see module docstring for every deviation",
        "criterion": "50%-prefix increment >= 90% of full-answer increment (LODO mean)",
        "n_rows": len(rows),
        "dropped": dict(dropped),
        "excluded_unresolved_by_source": dict(excluded_by_source),
        "excluded_unresolved_by_method": dict(excluded_methods),
        "n_unresolved_kept_for_sensitivity": len(unresolved),
        "rows_by_source": dict(collections.Counter(r["source_dataset"] for r in rows)),
        "error_rate_by_source": {},
        "models": {},
        # Set after scoring. Stays provisional while any row is unadjudicated,
        # regardless of what the labelled subset says.
        "full_population_conclusion": None,
    }
    by_src = collections.defaultdict(list)
    for r in rows:
        by_src[r["source_dataset"]].append(int(r["_label"]))
    for src, labels in by_src.items():
        payload["error_rate_by_source"][src] = round(float(np.mean(labels)), 4)

    for model_name, make_model in models().items():
        results = {a: lodo(rows, a, make_model) for a in arm_names}
        block = {}
        for a, res in results.items():
            block[a] = {"mean": res["mean"], "per_source": res["per_source"],
                        "skipped": res["skipped"]}

        def ratio(lo: str, hi: str) -> dict:
            lo_m, hi_m = results[lo]["mean"], results[hi]["mean"]
            if lo_m is None or hi_m is None:
                return {"verdict": "UNDECIDABLE", "note": "an arm had no scoreable source"}
            inc = hi_m - lo_m
            full = results["full_lp_ws"]["mean"] - results["full_lp"]["mean"]
            out = {"increment": inc, "full_answer_increment": full,
                   **increment_ci(results[lo], results[hi])}
            if full is None or full <= 0:
                out["verdict"] = "UNDECIDABLE"
                out["note"] = ("full-answer increment is <= 0 on this subset, so the "
                               "registered ratio has no denominator")
                return out
            out["retention_ratio"] = inc / full
            out["verdict"] = "HIT" if inc / full >= 0.90 else "MISS"
            return out

        block["registered_primary"] = ratio("prefix_lp", "prefix_lp_ws")
        # The measurement and the claim are different things. The first is what the
        # retained, labelled rows say. The second is what can be asserted about the
        # population, which stays provisional while any row is unadjudicated.
        block["registered_primary"]["scope"] = "labelled_subset_only"
        block["labelled_subset_verdict"] = block["registered_primary"]["verdict"]
        block["secondary_prefix_plus_last_token"] = ratio("prefix_plus_lp", "prefix_plus_lp_ws")
        block["deployable_token8"] = ratio("fixed8_lp", "fixed8_lp_ws")
        block["deployable_token16"] = ratio("fixed16_lp", "fixed16_lp_ws")
        # Probe the registered verdict against the unresolved rows. These are
        # SCENARIOS, NOT BOUNDS: AUROC is not monotone in label flips and the
        # statistic is a ratio of two AUROC differences, so a mixed assignment can
        # land outside both uniform ones. test_uniform_scenarios.py carries a
        # four-row counterexample. Agreement here is weak reassurance, never proof.
        if unresolved:
            bounds = {}
            for name, as_error in (("all_unresolved_are_errors", True),
                                   ("all_unresolved_are_correct", False)):
                for row in unresolved:
                    row["_label"] = as_error
                augmented = rows + unresolved
                lo = lodo(augmented, "prefix_lp", make_model)
                hi = lodo(augmented, "prefix_lp_ws", make_model)
                f_lo = lodo(augmented, "full_lp", make_model)
                f_hi = lodo(augmented, "full_lp_ws", make_model)
                if None in (lo["mean"], hi["mean"], f_lo["mean"], f_hi["mean"]):
                    bounds[name] = {"verdict": "UNDECIDABLE"}
                    continue
                full = f_hi["mean"] - f_lo["mean"]
                inc = hi["mean"] - lo["mean"]
                entry = {"increment": inc, "full_answer_increment": full}
                if full <= 0:
                    entry["verdict"] = "UNDECIDABLE"
                else:
                    entry["retention_ratio"] = inc / full
                    entry["verdict"] = "HIT" if inc / full >= 0.90 else "MISS"
                bounds[name] = entry
            for row in unresolved:
                row.pop("_label", None)
            verdicts = {b.get("verdict") for b in bounds.values()}
            primary = block["registered_primary"].get("verdict")
            bounds["uniform_scenarios_agree"] = (
                len(verdicts) == 1 and primary in verdicts)
            bounds["note"] = (
                "Unresolved rows are those the deterministic grader routed to a "
                "judge, and no judge has adjudicated them. These two uniform "
                "assignments are SENSITIVITY SCENARIOS, NOT BOUNDS: a mixed "
                "assignment can fall outside both, so agreement here does not "
                "certify the gate. Only adjudication can.")
            block["unresolved_sensitivity"] = bounds
        block["full_population_conclusion"] = (
            "PROVISIONAL_UNADJUDICATED" if unresolved
            else block["labelled_subset_verdict"])
        payload["models"][model_name] = block

    payload["full_population_conclusion"] = (
        "PROVISIONAL_UNADJUDICATED" if unresolved else "FINAL")
    payload["caveats"] = [
        "Reduced-scope follow-up, not the original Gate C plan executed unchanged.",
        "LODO only; pooled numbers untrusted per the registered identity-leakage rule.",
        "Logprob baselines are feature-matched one-to-one across prefix and full arms.",
        "Bootstrap intervals resample rows within each held-out source and therefore "
        "exclude between-source variation; with three or four sources the mean is a "
        "small-sample estimate.",
        "Labels are fresh deterministic grades of this run's own answers, produced by "
        "a grader whose numeric handling was corrected on 2026-09-10.",
        "The 50% checkpoint needs the eventual answer length and is an oracle "
        "observation; token 8 and token 16 are the deployable ones.",
        "Sources below the size or minority-class floor are skipped and listed.",
        "Rows the grader routed to a judge have NOT been adjudicated. They are "
        "excluded from the primary result and counted by source and method. The "
        "two uniform imputations reported are sensitivity scenarios, NOT bounds: "
        "AUROC is not monotone in label flips and the statistic is a ratio of two "
        "AUROC differences, so a mixed assignment can fall outside both.",
        "While any row is unadjudicated the full-population conclusion is "
        "PROVISIONAL_UNADJUDICATED, whatever the labelled subset says.",
    ]

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    with io.open(args.out, "w", encoding="utf-8") as fh:
        json.dump(payload, fh, indent=2)

    print(f"rows={len(rows)} dropped={dict(dropped)}")
    if excluded_by_source:
        print(f"UNRESOLVED (routed to a judge, none adjudicated): "
              f"{sum(excluded_by_source.values())}")
        print(f"  by source: {dict(excluded_by_source)}")
        print(f"  by method: {dict(excluded_methods)}")
    print(f"by source: {payload['rows_by_source']}")
    print(f"error rate: {payload['error_rate_by_source']}")
    for model_name, block in payload["models"].items():
        print(f"\n[{model_name}]")
        for a in arm_names:
            res = block[a]
            if res["mean"] is None:
                continue
            print(f"  {a:20s} LODO mean {res['mean']:.4f} over {len(res['per_source'])} sources")
            if res["skipped"]:
                print(f"    skipped: {res['skipped']}")
        for key in ("registered_primary", "secondary_prefix_plus_last_token",
                    "deployable_token8", "deployable_token16"):
            r = block[key]
            ratio_txt = (f"{r['retention_ratio']:.3f}" if r.get("retention_ratio") is not None
                         else "n/a")
            ci = ""
            if "increment_p2.5" in r:
                ci = f" CI[{r['increment_p2.5']:+.4f},{r['increment_p97.5']:+.4f}]"
            print(f"  {key:34s} increment {r.get('increment', float('nan')):+.4f}{ci} "
                  f"ratio {ratio_txt} -> {r['verdict']}")
        sens = block.get("unresolved_sensitivity")
        if sens:
            for name in ("all_unresolved_are_errors", "all_unresolved_are_correct"):
                b = sens[name]
                rt = (f"{b['retention_ratio']:.3f}" if b.get("retention_ratio") is not None
                      else "n/a")
                print(f"  scenario {name:35s} ratio {rt} -> {b['verdict']}")
            print(f"  uniform scenarios agree: {sens['uniform_scenarios_agree']} "
                  f"(scenarios, NOT bounds; a mixed assignment can fall outside both)")
        print(f"  labelled-subset verdict:     {block['labelled_subset_verdict']}")
        print(f"  full-population conclusion:  {block['full_population_conclusion']}")
    print(f"\nfull-population conclusion: {payload['full_population_conclusion']}")
    if unresolved:
        print(f"  {len(unresolved)} rows remain unadjudicated; the labelled-subset "
              f"verdict above is a measurement, not a settled result")
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
