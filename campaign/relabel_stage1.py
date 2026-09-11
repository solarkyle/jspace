"""Re-grade the Stage 1 rows the deterministic grader labelled, and nothing else.

The grader was corrected on 2026-09-10 (numeric fidelity, contradiction handling,
final-answer extraction). Any published result computed from its labels therefore
needs rechecking. This script produces the corrected label set without touching
anything a judge decided.

Provenance in stage1_judged.jsonl, by deterministic_grade.method:

    alias              13500   deterministic, re-graded here
    llm_prepass_alias    606   deterministic, re-graded here
    judge               2072   an LLM judge decided this; PRESERVED untouched
    needs_judge         2042   never labelled; left unlabelled

A row the corrected grader now finds ambiguous loses its label and joins the
unresolved set. That shrinks the labelled population, which is itself a result:
the old label set was partly built on evidence the grader should not have trusted.

This does not establish that the corrected labels are right. It establishes a
second label set, and the difference between the two bounds how much the published
numbers depend on grading decisions.

    python -m campaign.relabel_stage1 \
        --input out/campaign/stage1_judged.jsonl \
        --out out/campaign/stage1_judged_relabelled.jsonl
"""

import argparse
import collections
import io
import json
import os

from campaign.grade_deterministic import grade_row

# Methods the deterministic grader owns. Everything else is left alone.
REGRADEABLE = {"alias", "llm_prepass_alias", "unanswerable", "tool"}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--report", default="")
    args = ap.parse_args()

    counts = collections.Counter()
    per_source = collections.defaultdict(collections.Counter)
    examples = []

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    with io.open(args.input, encoding="utf-8") as fh, \
            io.open(args.out, "w", encoding="utf-8") as out:
        for line in fh:
            if not line.strip():
                continue
            row = json.loads(line)
            old = row.get("deterministic_grade") or {}
            method = old.get("method")
            source = row["source_dataset"]
            counts["rows"] += 1

            if method not in REGRADEABLE:
                counts[f"preserved:{method}"] += 1
                out.write(json.dumps(row) + "\n")
                continue

            new = grade_row(row)
            counts["regraded"] += 1
            per_source[source]["regraded"] += 1
            if new["correct"] is None:
                counts["lost_label_to_ambiguity"] += 1
                per_source[source]["lost_label"] += 1
            elif old.get("correct") is None:
                counts["gained_label"] += 1
                per_source[source]["gained_label"] += 1
            elif bool(new["correct"]) != bool(old["correct"]):
                counts["flipped"] += 1
                per_source[source]["flipped"] += 1
                if len(examples) < 60:
                    examples.append({
                        "example_id": row["example_id"],
                        "source_dataset": source,
                        "references": (row.get("references") or row.get("aliases"))[:4],
                        "answer": row["answer"][:300],
                        "old_correct": old.get("correct"),
                        "new_correct": new["correct"],
                        "old_method": method,
                        "new_method": new["method"],
                    })
            row["deterministic_grade"] = new
            row.setdefault("metadata", {})
            row["metadata"]["prior_deterministic_grade"] = old
            out.write(json.dumps(row) + "\n")

    print(f"rows={counts['rows']} regraded={counts['regraded']}")
    print(f"  flipped               {counts['flipped']}")
    print(f"  lost label (ambiguous) {counts['lost_label_to_ambiguity']}")
    print(f"  gained label           {counts['gained_label']}")
    for key in sorted(k for k in counts if k.startswith("preserved:")):
        print(f"  {key:24s} {counts[key]}")
    print(f"\n{'source':16s} {'regraded':>9s} {'flipped':>8s} {'lost_label':>11s}")
    for source in sorted(per_source):
        c = per_source[source]
        print(f"{source:16s} {c['regraded']:9d} {c['flipped']:8d} {c['lost_label']:11d}")

    if args.report:
        os.makedirs(os.path.dirname(args.report) or ".", exist_ok=True)
        with io.open(args.report, "w", encoding="utf-8") as fh:
            json.dump({"counts": dict(counts),
                       "per_source": {k: dict(v) for k, v in per_source.items()},
                       "flip_examples": examples}, fh, indent=2)
        print(f"\nwrote {args.report} ({len(examples)} flip examples for adjudication)")


if __name__ == "__main__":
    main()
