from __future__ import annotations

import math
from typing import Any

import numpy as np


DEFAULT_THRESHOLDS = {
    "residual_cosine_min": 0.995,
    "lens_jsd_max": 0.02,
    "topk_overlap_min": 0.80,
    # Cosine is scale invariant, so direction agreement alone cannot detect a
    # candidate whose residuals are uniformly rescaled. Compare norms too.
    "residual_norm_ratio_max": 1.05,
}

# A comparison has three outcomes, not two. "incomplete" means the candidate did
# not supply the evidence the contract asks for, which is NOT the same as
# agreeing with the reference. Only "pass" sets passed=True.
PASS, FAIL, INCOMPLETE = "pass", "fail", "incomplete"


def sparse_distribution(entry: dict[str, Any]) -> dict[int | str, float]:
    ids = [int(value) for value in entry.get("top_ids", [])]
    probs = [float(value) for value in entry.get("top_probs", [])]
    out: dict[int | str, float] = {
        token_id: probability for token_id, probability in zip(ids, probs)
    }
    out["<tail>"] = max(0.0, 1.0 - sum(out.values()))
    return out


def jensen_shannon(left: dict[int | str, float], right: dict[int | str, float]) -> float:
    keys = set(left) | set(right)
    total = 0.0
    for key in keys:
        p = max(0.0, float(left.get(key, 0.0)))
        q = max(0.0, float(right.get(key, 0.0)))
        midpoint = (p + q) / 2.0
        if p > 0:
            total += 0.5 * p * math.log(p / max(midpoint, 1e-12))
        if q > 0:
            total += 0.5 * q * math.log(q / max(midpoint, 1e-12))
    return float(total)


def cosine(left: list[float], right: list[float]) -> float | None:
    if not left or len(left) != len(right):
        return None
    a = np.asarray(left, dtype=np.float64)
    b = np.asarray(right, dtype=np.float64)
    denom = float(np.linalg.norm(a) * np.linalg.norm(b))
    if denom <= 1e-12:
        return None
    return float(np.dot(a, b) / denom)


def compare_captures(
    reference: dict[str, Any],
    candidate: dict[str, Any],
    *,
    thresholds: dict[str, float] | None = None,
) -> dict[str, Any]:
    """Compare a reference capture against a candidate, failing closed.

    This gate used to accept four kinds of invalid pair (see
    analysis/research_prefix_audit.py::conformance_negative_controls): a required
    layer missing from the candidate, missing residual vectors, both captures
    lacking final-output evidence, and residuals rescaled tenfold. Each of those
    returned passed=true, because the comparison only looked at the intersection
    of available evidence and treated "nothing to compare" as agreement.
    """
    limits = {**DEFAULT_THRESHOLDS, **(thresholds or {})}
    ref_layers = {int(row["layer"]): row for row in reference.get("layers", [])}
    cand_layers = {int(row["layer"]): row for row in candidate.get("layers", [])}
    aligned = sorted(set(ref_layers) & set(cand_layers))
    rows: list[dict[str, Any]] = []
    problems: list[str] = []

    for layer in aligned:
        ref = ref_layers[layer]
        cand = cand_layers[layer]
        ref_res = ref.get("residual") or []
        cand_res = cand.get("residual") or []
        residual_cosine = cosine(ref_res, cand_res)
        norm_ratio = None
        if ref_res and cand_res and len(ref_res) == len(cand_res):
            a = float(np.linalg.norm(np.asarray(ref_res, dtype=np.float64)))
            b = float(np.linalg.norm(np.asarray(cand_res, dtype=np.float64)))
            if min(a, b) > 1e-12:
                norm_ratio = max(a, b) / min(a, b)

        ref_dist = sparse_distribution(ref)
        cand_dist = sparse_distribution(cand)
        jsd = jensen_shannon(ref_dist, cand_dist)
        ref_top = set(int(v) for v in ref.get("top_ids", []))
        cand_top = set(int(v) for v in cand.get("top_ids", []))
        overlap = len(ref_top & cand_top) / max(1, len(ref_top | cand_top))

        if residual_cosine is None or norm_ratio is None:
            status = INCOMPLETE
            problems.append(f"layer {layer}: residual evidence missing or unusable")
        elif residual_cosine < limits["residual_cosine_min"]:
            status = FAIL
        elif norm_ratio > limits["residual_norm_ratio_max"]:
            status = FAIL
            problems.append(f"layer {layer}: residual norms differ by {norm_ratio:.3f}x")
        elif jsd > limits["lens_jsd_max"] or overlap < limits["topk_overlap_min"]:
            status = FAIL
        else:
            status = PASS

        rows.append({
            "layer": layer,
            "residual_cosine": residual_cosine,
            "residual_norm_ratio": norm_ratio,
            "lens_jsd": jsd,
            "topk_overlap": overlap,
            "status": status,
            "passed": status == PASS,
        })

    missing_reference = sorted(set(cand_layers) - set(ref_layers))
    missing_candidate = sorted(set(ref_layers) - set(cand_layers))
    if missing_candidate:
        problems.append(f"candidate is missing required layers {missing_candidate}")
    if not rows:
        problems.append("no layers could be compared")

    ref_final = (reference.get("final") or {}).get("next_token_id")
    cand_final = (candidate.get("final") or {}).get("next_token_id")
    if ref_final is None or cand_final is None:
        final_match = None
        problems.append("final-output evidence missing from at least one capture")
    else:
        final_match = ref_final == cand_final

    reference_tokens = reference.get("input_token_ids")
    candidate_tokens = candidate.get("input_token_ids")
    if reference_tokens is None or candidate_tokens is None:
        input_tokens_match = None
        problems.append("input_token_ids missing from at least one capture")
    else:
        input_tokens_match = list(reference_tokens) == list(candidate_tokens)

    if any(row["status"] == INCOMPLETE for row in rows) or not rows             or final_match is None or input_tokens_match is None or missing_candidate:
        status = INCOMPLETE
    elif all(row["status"] == PASS for row in rows) and final_match and input_tokens_match:
        status = PASS
    else:
        status = FAIL

    return {
        "schema_version": 2,
        "status": status,
        "passed": status == PASS,
        "problems": problems,
        "final_token_match": final_match,
        "input_tokens_match": input_tokens_match,
        "aligned_layers": len(rows),
        "missing_reference_layers": missing_reference,
        "missing_candidate_layers": missing_candidate,
        "thresholds": limits,
        "layers": rows,
    }
