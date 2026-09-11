"""Build the Gate C manifest from the existing Stage 1 campaign traces.

Gate C is the one registered gate the campaign never tested: can a monitor warn
before the answer finishes? Two facts from the saved traces shape the design:

  * Early warning is only meaningful where there is room to warn. trivia_qa's
    median answer is ~4 tokens and legal_hallucinations never reaches 8, so
    those sources cannot carry the test at all.
  * The original traces saved whole-answer logprob aggregates and stripped
    answer text. Decoded answers re-tokenize to the original length only ~8% of
    the time, so the original token IDs are unrecoverable and a replay pass
    would be scoring sequences the model never actually produced.

So we re-run generation. Greedy decoding is deterministic within a fixed
environment, but MEASURED AGAINST THE JULY TRACES IT IS NOT: regenerating the
same prompts on the same model id reproduced only 1 of 30 saved answers (fresh
output adds markdown emphasis and different continuations). The original
environment was unpinned and recorded nothing, and the cause of the divergence
has not been isolated. Ruled out so far: the hub commits since July touched
README, tokenizer_config and the chat template rather than weights, and the
rendered prompt is byte-identical across those revisions.

Therefore the July labels cannot be reused -- a grade describes the answer it was
given, and these are different answers. This manifest still carries the old
answer and label in metadata, but only as a comparison artifact; the gate is
scored against FRESH grades of the new answers. Sources are restricted to
grader_type=exact so that re-grading is free and needs no judge.

Usage:
    python -m campaign.build_gatec_manifest --min-tokens 32 --out campaign/manifests/gatec.jsonl
    python -m campaign.build_gatec_manifest --min-tokens 32 --limit-per-source 40 \
        --out campaign/manifests/gatec_smoke.jsonl
"""

import argparse
import collections
import glob
import io
import json
import os

# Stage 1 traces live wherever the campaign output tree is. Default to the repo's
# own out/ path and let JSPACE_ARCHIVE point elsewhere (out/ may be a symlink to
# an external drive on a machine that cannot hold the full corpus).
ARCHIVE = os.environ.get("JSPACE_ARCHIVE", os.path.join("out", "campaign"))
# esconv: single label class (unscoreable) and CC BY-NC, excluded from the public
# dataset release. Never include it in a published gate.
EXCLUDE_SOURCES = {"esconv"}
MIN_JUDGE_CONFIDENCE = 0.7
CARRY = ("example_id", "source_dataset", "domain", "task_type", "grader_type",
         "answerable", "split_group", "upstream_group", "prompt", "context",
         "references", "aliases")


def load_verdicts() -> dict:
    out = {}
    for path in glob.glob(os.path.join(ARCHIVE, "verdicts_*stage1*.jsonl")):
        with io.open(path, encoding="utf-8") as fh:
            for line in fh:
                if not line.strip():
                    continue
                row = json.loads(line)
                eid = row.get("example_id")
                if eid:
                    out[eid] = row
    return out


def label_of(row: dict, verdicts: dict) -> tuple:
    """Return (is_error, source_of_label) or (None, reason)."""
    grade = row.get("deterministic_grade") or {}
    if grade.get("correct") is not None:
        return (not bool(grade["correct"])), "deterministic"
    v = verdicts.get(row["example_id"])
    if v is None:
        return None, "unjudged"
    if float(v.get("confidence") or 0.0) < MIN_JUDGE_CONFIDENCE:
        return None, "low_confidence"
    verdict = str(v.get("verdict") or "").strip().lower()
    if verdict in {"hit", "error", "hallucination", "incorrect", "miss_fact", "wrong"}:
        return True, f"judge:{verdict}"
    if verdict in {"miss", "no_error", "correct", "supported", "ok"}:
        return False, f"judge:{verdict}"
    return None, f"unmapped:{verdict}"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", default=os.path.join(ARCHIVE, "stage1_graded.jsonl"))
    ap.add_argument("--min-tokens", type=int, default=32)
    ap.add_argument("--limit-per-source", type=int, default=0)
    # Measured cost on this subset is ~$5.42/1k prompts, not the pilot's $2.02/1k
    # blended rate -- these are deliberately the longest-context rows in the
    # corpus. Restricting sources is therefore a budget lever, and the sources
    # worth paying for are the ones with enough minority-class rows to give a
    # stable per-source AUROC.
    ap.add_argument("--sources", default="", help="comma-separated allowlist")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    allow = {x.strip() for x in args.sources.split(",") if x.strip()}
    verdicts = load_verdicts()
    kept, reasons, per_source, errors = [], collections.Counter(), collections.Counter(), collections.Counter()
    with io.open(args.input, encoding="utf-8") as fh:
        for line in fh:
            if not line.strip():
                continue
            row = json.loads(line)
            src = row["source_dataset"]
            if src in EXCLUDE_SOURCES:
                reasons["excluded_source"] += 1
                continue
            if allow and src not in allow:
                reasons["not_in_allowlist"] += 1
                continue
            if row["token_count"] < args.min_tokens:
                reasons["too_short"] += 1
                continue
            is_error, why = label_of(row, verdicts)
            if is_error is None:
                reasons[why] += 1
                continue
            if args.limit_per_source and per_source[src] >= args.limit_per_source:
                reasons["over_source_cap"] += 1
                continue
            out = {k: row.get(k) for k in CARRY}
            # Carried through run_shard untouched, so the new trace can be checked
            # against the old one without a second join.
            out["metadata"] = {
                "gatec": True,
                "prior_answer": row["answer"],
                "prior_token_count": row["token_count"],
                "label_is_error": bool(is_error),
                "label_source": why,
            }
            kept.append(out)
            per_source[src] += 1
            errors[src] += int(bool(is_error))

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    with io.open(args.out, "w", encoding="utf-8") as fh:
        for row in kept:
            fh.write(json.dumps(row, ensure_ascii=False) + "\n")

    print(f"wrote {len(kept)} rows -> {args.out}")
    print(f"{'source':24s} {'n':>6s} {'errors':>7s} {'err_rate':>9s}")
    for src, n in per_source.most_common():
        print(f"{src:24s} {n:6d} {errors[src]:7d} {errors[src] / n:9.3f}")
    print("dropped:", dict(reasons.most_common()))


if __name__ == "__main__":
    main()
