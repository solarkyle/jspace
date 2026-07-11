"""Validate a campaign manifest before it costs GPU time.

Checks the invariants that protect the science:
  - unique example_id (no accidental dup prompts)
  - every record has a split_group (leakage control) and pinned revision
  - grounded_qa / unanswerable rows carry context
  - answerable rows have a way to be graded (references or aliases), except
    llm-graded rows which are judged
  - reports per-source counts and answerable balance

Usage:
    python -m campaign.validate_manifest campaign/manifests/pilot.jsonl
"""

from __future__ import annotations

import json
import sys
from collections import Counter, defaultdict


def main(path: str) -> int:
    rows = [json.loads(l) for l in open(path, encoding="utf-8") if l.strip()]
    errors, warnings = [], []
    ids = Counter(r["example_id"] for r in rows)
    dups = [k for k, v in ids.items() if v > 1]
    if dups:
        errors.append(f"{len(dups)} duplicate example_id(s)")

    by_src = defaultdict(list)
    for r in rows:
        by_src[r["source_dataset"]].append(r)
        if not r.get("split_group"):
            errors.append(f"{r['example_id']}: missing split_group")
        if not r.get("source_revision"):
            errors.append(f"{r['example_id']}: unpinned revision")
        if r["task_type"] in ("grounded_qa", "unanswerable") and not r.get("context"):
            # unanswerable squad rows still have a context passage
            if r["source_dataset"] != "esconv":
                warnings.append(f"{r['example_id']}: {r['task_type']} without context")
        if r["grader_type"] == "exact" and r.get("answerable") and not (
                r.get("references") or r.get("aliases")):
            errors.append(f"{r['example_id']}: exact-graded but no references/aliases")

    print(f"{len(rows)} records, {len(by_src)} sources")
    for src, rs in sorted(by_src.items()):
        ans = sum(1 for r in rs if r.get("answerable"))
        groups = len({r["split_group"] for r in rs})
        print(f"  {src:>12}: {len(rs):>4}  answerable={ans:>3}  groups={groups:>4}  "
              f"grader={rs[0]['grader_type']}")

    # split_group overlap across sources would break leave-one-dataset-out
    grp_src = defaultdict(set)
    for r in rows:
        grp_src[r["split_group"]].add(r["source_dataset"])
    shared = [g for g, s in grp_src.items() if len(s) > 1]
    if shared:
        warnings.append(f"{len(shared)} split_group value(s) shared across sources")

    for w in warnings:
        print("WARN ", w)
    for e in errors:
        print("ERROR", e)
    print("OK" if not errors else f"FAILED with {len(errors)} error(s)")
    return 1 if errors else 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1] if len(sys.argv) > 1 else "campaign/manifests/pilot.jsonl"))
