"""Deterministic grading for exact/numeric/unanswerable rows (handoff 10.1).

Cheapest reliable method first. Rows whose grader_type is "llm" are left for
blinded Claude adjudication and only get a deterministic pre-pass here when a
reference obviously matches (to save judge calls).

Produces, per response row, a `deterministic_grade` dict:
  {"method": ..., "correct": bool|None, "abstained": bool}
`correct=None` means "needs a judge". The final training label
`eventual_response_error` is derived downstream from correct once judges resolve
the None cases.

Usage:
    python -m campaign.grade_deterministic --input out/campaign/pilot.jsonl \
        --out out/campaign/pilot_graded.jsonl
"""

from __future__ import annotations

import argparse
import json
import re

_ARTICLES = {"a", "an", "the"}
_ABSTAIN = re.compile(
    r"\b(cannot answer|can't answer|not (?:enough|sufficient) (?:information|context)|"
    r"unanswerable|no answer|not (?:provided|specified|mentioned|stated|found)|"
    r"does not (?:contain|provide|mention)|unable to answer|i don'?t know)\b", re.I)


def normalize(s: str) -> str:
    s = s.lower()
    s = re.sub(r"[^a-z0-9 ]", " ", s)
    toks = [t for t in s.split() if t not in _ARTICLES]
    return " ".join(toks).strip()


def _num(s: str):
    m = re.search(r"-?\d[\d,]*\.?\d*", s.replace(",", ""))
    return float(m.group()) if m else None


# Numbers must be read from RAW text. normalize() deletes "." and "-", so running
# _num() on a normalized string turned "3.14" into "3 14" and read it as 3.0,
# which made 3.15 and 3.14 compare equal, and turned "-5" into "5".
_NUMBER = re.compile(r"-?\d[\d,]*(?:\.\d+)?")
_PURE_NUMBER = re.compile(r"^\s*-?\d[\d,]*(?:\.\d+)?\s*$")

# A reference can appear inside an answer that denies it. Containment alone reads
# "Not Paris; the answer is London." as a hit for "Paris". Rather than guess, such
# rows are reported ambiguous and sent to a judge.
_CONTRADICTION = re.compile(
    r"\b(not|isn'?t|aren'?t|wasn'?t|weren'?t|doesn'?t|didn'?t|don'?t|cannot|"
    r"rather than|instead of|incorrect|mistaken|false)\b", re.I)


def _numbers(raw):
    """Every number in raw text, signs and decimals intact."""
    return [float(m.group().replace(",", "")) for m in _NUMBER.finditer(raw or "")]


def _is_pure_number(raw):
    return bool(_PURE_NUMBER.match(raw or ""))


# Where an answer states its conclusion. Matching against the span after the last
# such marker means earlier working, abandoned candidates and restated questions
# are not treated as the answer.
_FINAL_MARKER = re.compile(
    r"\b(?:my\s+|the\s+)?final\s+answer(?:\s+is|\s+would\s+be)?\s*[:=]?|"
    r"\bthe\s+answer\s+is\b|"
    r"\banswer\s*[:=]|"
    r"\bin\s+conclusion\b,?|"
    r"\bso\s+the\s+answer\s+is\b|"
    r"\btherefore\b,?", re.I)


def extract_final_answer(answer: str):
    """Return (span_to_grade, how) where how is "marked" or "whole".

    "marked" means the answer named its own conclusion and the span is only that
    conclusion. "whole" means no marker was found and the entire answer is graded,
    which is the looser case and is why contradiction cues still matter there.
    """
    text = (answer or "").strip()
    if not text:
        return text, "whole"
    marks = list(_FINAL_MARKER.finditer(text))
    if marks:
        span = text[marks[-1].end():].strip()
        if span:
            return span, "marked"
    return text, "whole"


def alias_match_detail(answer: str, refs: list) -> str:
    """Return "hit", "miss" or "ambiguous".

    "ambiguous" means surface evidence is not trustworthy on its own and the row
    needs a judge. Grading an ambiguous row correct is the failure that matters
    here: it silently deletes a real error from the labels.
    """
    span, how = extract_final_answer(answer)
    na = normalize(span)
    if not na:
        return "miss"
    na_tokens = set(na.split())
    span_numbers = _numbers(span)
    matched = False

    for r in refs:
        # A numeric reference is decided numerically, never by string containment,
        # and a numeric miss stays a miss rather than falling through to text rules.
        if _is_pure_number(r):
            target = _numbers(r)[0]
            if any(abs(value - target) < 1e-9 for value in span_numbers):
                matched = True
            continue
        nr = normalize(r)
        if not nr:
            continue
        if len(nr) < 4:
            if nr in na_tokens:
                matched = True
            continue
        if nr in na or (len(na) >= 4 and na in nr):
            matched = True

    if not matched:
        return "miss"
    # A contradiction inside the graded span is decisive either way: the span is
    # what the answer asserts. When nothing marked the conclusion, a cue anywhere
    # in the answer is enough to stop trusting surface containment.
    scope = span if how == "marked" else answer
    return "ambiguous" if _CONTRADICTION.search(scope or "") else "hit"


def alias_match(answer: str, refs: list[str]) -> bool:
    """Backwards-compatible boolean view. Ambiguous counts as NOT a hit."""
    return alias_match_detail(answer, refs) == "hit"


def _extract_json_object(text: str):
    """First balanced {...} block in text that parses as JSON, else None.
    Tolerates markdown fences and prose around the object."""
    import json as _json
    start = text.find("{")
    while start != -1:
        depth = 0
        for i in range(start, len(text)):
            c = text[i]
            if c == "{":
                depth += 1
            elif c == "}":
                depth -= 1
                if depth == 0:
                    try:
                        return _json.loads(text[start:i + 1])
                    except _json.JSONDecodeError:
                        break
        start = text.find("{", start + 1)
    return None


def _loose_eq(a, b) -> bool:
    """Value equality with numeric tolerance and case/whitespace-insensitive
    strings; lists element-wise (order-sensitive, per BFCL AST convention)."""
    if isinstance(a, bool) or isinstance(b, bool):
        return a is b or str(a).lower() == str(b).lower()
    if isinstance(a, (int, float)) and isinstance(b, (int, float)):
        return abs(float(a) - float(b)) < 1e-6
    if isinstance(a, str) and isinstance(b, (int, float)):
        return _loose_eq(b, a)
    if isinstance(a, (int, float)) and isinstance(b, str):
        n = _num(b)
        return n is not None and abs(float(a) - n) < 1e-6
    if isinstance(a, str) and isinstance(b, str):
        return a.strip().lower() == b.strip().lower()
    if isinstance(a, list) and isinstance(b, list):
        return len(a) == len(b) and all(_loose_eq(x, y) for x, y in zip(a, b))
    if isinstance(a, dict) and isinstance(b, dict):
        return (set(a) == set(b)
                and all(_loose_eq(a[k], b[k]) for k in a))
    return a == b


def grade_tool_call(answer: str, reference_json: str) -> dict:
    """Structural match of the model's emitted call vs the reference call.
    Wrong function, unparseable output, or mismatched arguments = error."""
    import json as _json
    ref = _json.loads(reference_json)
    pred = _extract_json_object(answer)
    if not isinstance(pred, dict):
        return {"method": "tool_ast", "correct": False, "abstained": False,
                "tool_fail": "no_json"}
    name = pred.get("name")
    args = pred.get("arguments", pred.get("parameters"))
    if isinstance(args, str):
        args = _extract_json_object(args)
    if name != ref["name"]:
        return {"method": "tool_ast", "correct": False, "abstained": False,
                "tool_fail": "wrong_function"}
    if not isinstance(args, dict) or not _loose_eq(args, ref["arguments"]):
        return {"method": "tool_ast", "correct": False, "abstained": False,
                "tool_fail": "wrong_arguments"}
    return {"method": "tool_ast", "correct": True, "abstained": False}


def grade_row(row: dict) -> dict:
    answer = row.get("answer", "")
    refs = row.get("references") or row.get("aliases") or []
    answerable = row.get("answerable", True)
    grader = row.get("grader_type", "exact")
    abstained = bool(_ABSTAIN.search(answer))

    if grader == "tool":
        return grade_tool_call(answer, refs[0])

    if grader == "exact":
        if not answerable:
            # correct iff the model abstained (SQuAD 2.0 unanswerable)
            return {"method": "unanswerable", "correct": abstained,
                    "abstained": abstained}
        verdict = alias_match_detail(answer, refs)
        if verdict == "hit":
            return {"method": "alias", "correct": True, "abstained": abstained}
        if verdict == "ambiguous":
            # The reference appears, but inside a contradiction. Surface matching
            # cannot tell "Paris" from "not Paris" here, and guessing either way
            # writes a wrong label, so abstain and let a judge decide.
            return {"method": "needs_judge", "correct": None, "abstained": abstained}
        # no alias hit and the model abstained on an answerable q -> wrong (miss)
        if abstained:
            return {"method": "alias", "correct": False, "abstained": True}
        # confident non-matching answer: usually wrong, but alias lists are noisy
        return {"method": "alias", "correct": False, "abstained": False}

    # llm-graded sources: cheap pre-pass only, and only on unambiguous evidence
    if refs and alias_match_detail(answer, refs) == "hit":
        return {"method": "llm_prepass_alias", "correct": True, "abstained": abstained}
    return {"method": "needs_judge", "correct": None, "abstained": abstained}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    rows = [json.loads(l) for l in open(args.input, encoding="utf-8") if l.strip()]
    from collections import Counter
    methods, need_judge = Counter(), 0
    with open(args.out, "w", encoding="utf-8") as f:
        for r in rows:
            g = grade_row(r)
            r["deterministic_grade"] = g
            methods[g["method"]] += 1
            if g["correct"] is None:
                need_judge += 1
            f.write(json.dumps(r) + "\n")
    graded = len(rows) - need_judge
    print(f"{len(rows)} rows; {graded} deterministically graded, {need_judge} need a judge")
    print("methods:", dict(methods))
    # accuracy on the deterministically graded exact rows
    det = [r for r in rows if r["deterministic_grade"]["correct"] is not None
           and r["grader_type"] == "exact"]
    if det:
        acc = sum(r["deterministic_grade"]["correct"] for r in det) / len(det)
        print(f"exact-source accuracy: {acc:.3f} (n={len(det)})")


if __name__ == "__main__":
    main()
