"""Report how often regeneration reproduced the original campaign answer.

This measures EXACT TEXT EQUALITY and answer length, and nothing else. It does
not measure whether accuracy, detector performance or any published finding
reproduces, and a low score here is not evidence that the original results were
wrong: those runs were internally consistent against whatever environment existed
at the time.

Greedy decoding is deterministic within a fixed environment. The campaign's image
pinned neither transformers nor a model revision and recorded no environment, so
there is no way to re-create the original conditions and confirm what changed.
The cause of the divergence has not been isolated.
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
