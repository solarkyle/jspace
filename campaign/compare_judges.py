"""Judge-model agreement: Fable vs Sonnet 5 on the same blinded requests.

If Sonnet 5 agrees with Fable closely on the derived binary error label, Stage 1
judging can run on Sonnet at ~1/10th the cost. Reports raw agreement, Cohen's
kappa on the binary label, and per-schema breakdown, plus the categorical
disagreement cases for spot inspection.

Usage:
    python -m campaign.compare_judges --a "out/campaign/verdicts_shard*.jsonl" \
        --b "out/campaign/verdicts_sonnet_shard*.jsonl" --a-name fable --b-name sonnet
"""

from __future__ import annotations

import argparse
import glob
import json

from campaign.grade_claude import _label_from_verdict


def _load(pattern):
    out = {}
    for path in sorted(glob.glob(pattern)):
        for line in open(path, encoding="utf-8"):
            if line.strip():
                v = json.loads(line)
                out[v["example_id"]] = v
    return out


def _kappa(a, b):
    """Cohen's kappa for two binary label lists."""
    n = len(a)
    if n == 0:
        return float("nan")
    po = sum(x == y for x, y in zip(a, b)) / n
    pa1 = sum(a) / n
    pb1 = sum(b) / n
    pe = pa1 * pb1 + (1 - pa1) * (1 - pb1)
    return (po - pe) / (1 - pe) if pe < 1 else 1.0


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--a", required=True)
    ap.add_argument("--b", required=True)
    ap.add_argument("--a-name", default="A")
    ap.add_argument("--b-name", default="B")
    args = ap.parse_args()

    A, B = _load(args.a), _load(args.b)
    shared = sorted(set(A) & set(B))
    print(f"{args.a_name}: {len(A)} verdicts, {args.b_name}: {len(B)}, shared: {len(shared)}")

    la, lb, disagree = [], [], []
    per_schema = {}
    for eid in shared:
        va, vb = A[eid], B[eid]
        ya, yb = _label_from_verdict(va), _label_from_verdict(vb)
        sch = va.get("_schema", "?")
        d = per_schema.setdefault(sch, {"n": 0, "agree": 0, "a": [], "b": []})
        d["n"] += 1
        # treat ambiguous(None) as its own bucket for agreement counting
        if ya == yb:
            d["agree"] += 1
        if ya is not None and yb is not None:
            la.append(ya); lb.append(yb)
            d["a"].append(ya); d["b"].append(yb)
            if ya != yb:
                disagree.append((eid, sch, _verdict_key(va), _verdict_key(vb)))

    print(f"\noverall binary-label agreement (excl. ambiguous): "
          f"{sum(x==y for x,y in zip(la,lb))}/{len(la)} = "
          f"{sum(x==y for x,y in zip(la,lb))/max(1,len(la)):.3f}")
    print(f"Cohen's kappa: {_kappa(la, lb):.3f}")
    for sch, d in sorted(per_schema.items()):
        k = _kappa(d["a"], d["b"]) if d["a"] else float("nan")
        print(f"  {sch:>9}: n={d['n']:>3}  raw agree={d['agree']/d['n']:.3f}  "
              f"kappa={k:.3f}  (label-comparable n={len(d['a'])})")

    print(f"\n{len(disagree)} binary disagreements:")
    for eid, sch, ka, kb in disagree[:20]:
        print(f"  {eid} [{sch}] {args.a_name}={ka} {args.b_name}={kb}")


def _verdict_key(v):
    if v.get("_schema") == "therapy":
        return f"fact={v.get('factuality')},fab={v.get('fabricated_resource')}"
    return v.get("verdict")


if __name__ == "__main__":
    main()
