"""Report how often regeneration reproduced the original campaign answer.

This is not a pass/fail gate, it is a measurement worth publishing. Greedy
decoding is deterministic within a fixed environment, but the campaign's Modal
image pinned neither transformers nor a model revision and recorded no
environment, so "same model id, same decoding" does not imply the same output
months later.
"""

import argparse
import collections
import io
import json


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True)
    args = ap.parse_args()

    per = collections.defaultdict(lambda: [0, 0, 0])
    with io.open(args.input, encoding="utf-8") as fh:
        for line in fh:
            if not line.strip():
                continue
            row = json.loads(line)
            meta = row.get("metadata") or {}
            prior = meta.get("prior_answer")
            if prior is None:
                continue
            slot = per[row["source_dataset"]]
            slot[0] += 1
            slot[1] += int(row["answer"] == prior)
            slot[2] += int(row["token_count"] == meta.get("prior_token_count"))

    print(f"{'source':24s} {'n':>6s} {'exact':>7s} {'same_len':>9s}")
    tot = [0, 0, 0]
    for src, (n, same, samelen) in sorted(per.items()):
        print(f"{src:24s} {n:6d} {same:7d} {samelen:9d}")
        for i, v in enumerate((n, same, samelen)):
            tot[i] += v
    if tot[0]:
        print(f"{'TOTAL':24s} {tot[0]:6d} {tot[1]:7d} {tot[2]:9d}"
              f"   ({100.0 * tot[1] / tot[0]:.1f}% of answers reproduced)")


if __name__ == "__main__":
    main()
