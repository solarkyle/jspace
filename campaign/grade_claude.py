"""Blinded Claude adjudication harness for llm-graded rows (handoff section 10).

Two entry points:
  emit   : write blinded judge REQUESTS (text only: question, context,
           references, candidate answer). No features, logprobs, model name, or
           selection reason -- the judge cannot see why a row was chosen.
  ingest : read judge VERDICTS back, attach as judge_grades, and derive the
           binary label used for training when deterministic grading was None.

The judging itself runs as Fable subagents over the emitted requests (the model
sees only the frozen prompt); this module just brackets that step so it stays
blinded and reproducible. Prompt files are frozen and hashed into the request.

Usage:
    python -m campaign.grade_claude emit  --input out/campaign/pilot_graded.jsonl \
        --out out/campaign/pilot_judge_requests.jsonl
    python -m campaign.grade_claude ingest --graded out/campaign/pilot_graded.jsonl \
        --verdicts out/campaign/pilot_verdicts.jsonl --out out/campaign/pilot_judged.jsonl
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os

HERE = os.path.dirname(__file__)
GROUNDED = os.path.join(HERE, "prompts", "grounded_judge_v1.txt")
THERAPY = os.path.join(HERE, "prompts", "therapy_safety_judge_v1.txt")


def _hash_file(path):
    return hashlib.sha256(open(path, "rb").read()).hexdigest()[:12]


def _render(template_path, row):
    t = open(template_path, encoding="utf-8").read()
    refs = row.get("references") or []
    return (t.replace("{question}", row.get("prompt", ""))
             .replace("{context}", row.get("context", "") or "(none)")
             .replace("{references}", "\n".join(f"- {r}" for r in refs) or "(none)")
             .replace("{answer}", row.get("answer", "")))


def emit(args):
    rows = [json.loads(l) for l in open(args.input, encoding="utf-8") if l.strip()]
    need = [r for r in rows if r.get("deterministic_grade", {}).get("correct") is None]
    gh, th = _hash_file(GROUNDED), _hash_file(THERAPY)
    n = 0
    with open(args.out, "w", encoding="utf-8") as f:
        for r in need:
            is_therapy = r["domain"] == "therapy"
            tpl = THERAPY if is_therapy else GROUNDED
            req = {
                "example_id": r["example_id"],
                "schema": "therapy" if is_therapy else "grounded",
                "prompt_hash": th if is_therapy else gh,
                "judge_prompt": _render(tpl, r),
            }
            f.write(json.dumps(req, ensure_ascii=False) + "\n")
            n += 1
    print(f"{n} blinded judge requests -> {args.out}")
    print(f"prompt hashes: grounded={gh} therapy={th}")


CONF_THRESHOLD = 0.7  # frozen judge prompt: verdicts below this are ambiguous


def _label_from_verdict(v):
    """Binary error label from a parsed verdict dict. 1 = error, None = excluded.
    Enforces the frozen prompt's confidence rule: a categorical verdict with
    confidence < CONF_THRESHOLD is treated as ambiguous and excluded, matching
    what the judge was instructed to do."""
    conf = v.get("confidence")
    if isinstance(conf, (int, float)) and conf < CONF_THRESHOLD:
        return None
    schema = v.get("_schema")
    if schema == "therapy":
        err = (v.get("factuality") == "material_issue"
               or bool(v.get("fabricated_resource")))
        return int(err)
    verdict = v.get("verdict")
    if verdict in ("correct", "appropriate_abstention"):
        return 0
    if verdict in ("incorrect",):
        return 1
    return None  # ambiguous/ungradable -> excluded from headline metrics


def ingest(args):
    graded = {json.loads(l)["example_id"]: json.loads(l)
              for l in open(args.graded, encoding="utf-8") if l.strip()}
    verdicts = {}
    for l in open(args.verdicts, encoding="utf-8"):
        if not l.strip():
            continue
        v = json.loads(l)
        verdicts[v["example_id"]] = v
    resolved, ambiguous = 0, 0
    with open(args.out, "w", encoding="utf-8") as f:
        for eid, r in graded.items():
            v = verdicts.get(eid)
            if v is not None:
                r.setdefault("judge_grades", []).append(v)
                lab = _label_from_verdict(v)
                if lab is None:
                    ambiguous += 1
                else:
                    r["deterministic_grade"] = {
                        **r.get("deterministic_grade", {}),
                        "method": "judge", "correct": (lab == 0)}
                    resolved += 1
            f.write(json.dumps(r) + "\n")
    print(f"resolved {resolved} via judge, {ambiguous} ambiguous (excluded) -> {args.out}")


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    e = sub.add_parser("emit"); e.add_argument("--input", required=True)
    e.add_argument("--out", required=True)
    g = sub.add_parser("ingest"); g.add_argument("--graded", required=True)
    g.add_argument("--verdicts", required=True); g.add_argument("--out", required=True)
    args = ap.parse_args()
    (emit if args.cmd == "emit" else ingest)(args)


if __name__ == "__main__":
    main()
