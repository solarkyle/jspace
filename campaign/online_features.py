"""Features a live monitor could actually compute, with no numpy dependency.

Kept in its own pure module so CPU tests can import it without Modal or torch,
and so the remote runner and any offline analysis share one implementation
instead of drifting apart (the live sidecar router and the frozen campaign
classifier already drifted to 14 vs 31 features).
"""

from __future__ import annotations


def prefix_lp(step_logprobs: list[float], k: int) -> dict[str, float]:
    """Output-confidence aggregates using ONLY tokens 0..k inclusive.

    The campaign's bl_* features aggregate the whole answer and include its final
    length, so they cannot legitimately appear in an early-warning model.
    Everything returned here is available at the moment token k is emitted.
    """
    if not step_logprobs:
        raise ValueError("step_logprobs is empty")
    if not 0 <= k < len(step_logprobs):
        raise IndexError(f"checkpoint {k} outside answer of {len(step_logprobs)} tokens")
    window = [float(x) for x in step_logprobs[: k + 1]]
    return {
        "pfxlp_first": float(step_logprobs[0]),
        "pfxlp_last": window[-1],
        "pfxlp_mean": sum(window) / len(window),
        "pfxlp_min": min(window),
        "pfxlp_tokens_so_far": float(len(window)),
    }
