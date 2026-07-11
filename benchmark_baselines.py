"""Compare the workspace router against cheap and optional expensive baselines.

This script is intentionally trace-only by default: it uses the committed
`data/uncertainty_trivia_*.jsonl` runs and the published router weights. If you
later run semantic entropy, P(True), verbalized confidence, or a hidden-state
probe, pass their scores as JSONL with `--extra-scores`.

Extra-score JSONL format:
    {"model": "gemma-4-e4b-it", "q": "...", "method": "semantic_entropy",
     "score": 1.23, "cost_x": 10}

`score` is interpreted as higher = more likely wrong.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any


ORDER = [
    "gemma-4-e4b-it",
    "gemma-4-12b-it",
    "huihui-gemma-4-12b-it-abliterated",
    "gemma-4-26b-a4b-it",
    "qwen3.6-27b",
]
NAME = {
    "gemma-4-e4b-it": "E4B",
    "gemma-4-12b-it": "12B",
    "huihui-gemma-4-12b-it-abliterated": "12B-ablit",
    "gemma-4-26b-a4b-it": "26B-MoE",
    "qwen3.6-27b": "Qwen-27B",
}
BASE = [
    "bl_first_token_logprob",
    "bl_mean_logprob",
    "bl_min_logprob",
    "bl_answer_len",
]
WS = [
    "ws_mean_entropy",
    "ws_max_entropy",
    "ws_late_entropy",
    "ws_entropy_slope",
    "ws_entropy_std",
    "ws_ignition_frac",
    "ws_ignition_depth",
    "ws_mean_log_rank",
    "ws_band_agreement",
    "ws_hedge_rank",
]


def load_rows(slug: str) -> list[dict[str, Any]]:
    path = Path(f"data/uncertainty_trivia_{slug}.jsonl")
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]


def featurize(row: dict[str, Any]) -> dict[str, float]:
    e = [float(v) for v in row["layer_entropies"]]
    n = len(e)
    x = list(range(n))
    x_mean = sum(x) / n
    e_mean = sum(e) / n
    denom = sum((v - x_mean) ** 2 for v in x)
    slope = (
        sum((x[i] - x_mean) * (e[i] - e_mean) for i in range(n)) / denom
        if denom
        else 0.0
    )
    late = e[2 * n // 3 :]
    return {
        "bl_first_token_logprob": float(row["bl_first_token_logprob"]),
        "bl_mean_logprob": float(row["bl_mean_logprob"]),
        "bl_min_logprob": float(row["bl_min_logprob"]),
        "bl_answer_len": float(row["bl_answer_len"]),
        "ws_mean_entropy": e_mean,
        "ws_max_entropy": max(e),
        "ws_late_entropy": sum(late) / len(late),
        "ws_entropy_slope": slope,
        "ws_entropy_std": (sum((v - e_mean) ** 2 for v in e) / n) ** 0.5,
        "ws_ignition_frac": float(row["ignition_frac"]),
        "ws_ignition_depth": float(row["ignition_depth"]),
        "ws_mean_log_rank": float(row["mean_log_rank_answer"]),
        "ws_band_agreement": float(row["band_agreement"]),
        "ws_hedge_rank": float(row["best_hedge_rank_log"]),
    }


def sigmoid(x: float) -> float:
    if x >= 0:
        return 1.0 / (1.0 + math.exp(-x))
    exp_x = math.exp(x)
    return exp_x / (1.0 + exp_x)


def router_score(router: dict[str, Any], norms: dict[str, list[float]], feat: dict[str, float]) -> float:
    total = float(router["bias"])
    for name, weight in zip(router["features"], router["weights"]):
        mean, std = norms.get(name, [0.0, 1.0])
        if abs(std) < 1e-9:
            std = 1.0
        total += float(weight) * ((feat.get(name, 0.0) - mean) / std)
    return sigmoid(total)


def auc(scores: list[float], labels: list[bool]) -> float:
    n_pos = sum(labels)
    n_neg = len(labels) - n_pos
    if not n_pos or not n_neg:
        return float("nan")
    order = sorted(range(len(scores)), key=lambda i: scores[i])
    ranks = [0] * len(scores)
    for rank, idx in enumerate(order, start=1):
        ranks[idx] = rank
    pos_rank_sum = sum(ranks[i] for i, label in enumerate(labels) if label)
    return (pos_rank_sum - n_pos * (n_pos + 1) / 2) / (n_pos * n_neg)


def catch_at(scores: list[float], labels: list[bool], budget: float) -> float:
    n_wrong = sum(labels)
    if not n_wrong:
        return float("nan")
    k = max(1, int(len(scores) * budget))
    chosen = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:k]
    return sum(1 for i in chosen if labels[i]) / n_wrong


def load_extra_scores(path: Path | None) -> dict[tuple[str, str, str], tuple[float, float]]:
    if path is None:
        return {}
    out: dict[tuple[str, str, str], tuple[float, float]] = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        rec = json.loads(line)
        model = str(rec.get("model") or "")
        method = str(rec["method"])
        question = str(rec["q"])
        out[(model, question, method)] = (float(rec["score"]), float(rec.get("cost_x", 1)))
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--routers", default="data/workspace_routers_all5.json")
    parser.add_argument("--zero-shot-router", default="data/workspace_router_e4b.json")
    parser.add_argument("--extra-scores", type=Path, default=None)
    parser.add_argument("--out-json", type=Path, default=Path("data/baseline_benchmark.json"))
    parser.add_argument("--out-md", type=Path, default=Path("docs/BASELINE_BENCHMARK.md"))
    args = parser.parse_args()

    routers = json.loads(Path(args.routers).read_text(encoding="utf-8"))
    zero_shot = json.loads(Path(args.zero_shot_router).read_text(encoding="utf-8"))
    extra = load_extra_scores(args.extra_scores)
    extra_methods = sorted({key[2] for key in extra})
    results: list[dict[str, Any]] = []

    for slug in ORDER:
        rows = load_rows(slug)
        feats = [featurize(row) for row in rows]
        wrong = [not bool(row["correct"]) for row in rows]
        model_router = routers["routers"][slug]
        norms = routers.get("norm_stats", {}).get(slug) or model_router.get("norm_stats", {})
        methods: dict[str, tuple[list[float], float]] = {
            "first-token logprob": ([-f["bl_first_token_logprob"] for f in feats], 1.0),
            "mean logprob": ([-f["bl_mean_logprob"] for f in feats], 1.0),
            "min logprob": ([-f["bl_min_logprob"] for f in feats], 1.0),
            "workspace E4B zero-shot": (
                [
                    router_score(
                        zero_shot["models"]["workspace_only"],
                        norms,
                        f,
                    )
                    for f in feats
                ],
                1.0,
            ),
            "combined E4B zero-shot": (
                [
                    router_score(
                        zero_shot["models"]["combined"],
                        norms,
                        f,
                    )
                    for f in feats
                ],
                1.0,
            ),
            "workspace router": (
                [router_score(model_router["workspace_only"], norms, f) for f in feats],
                1.0,
            ),
            "combined router": (
                [router_score(model_router["combined"], norms, f) for f in feats],
                1.0,
            ),
        }
        for method in extra_methods:
            scores: list[float] = []
            labels: list[bool] = []
            cost = 1.0
            for row, is_wrong in zip(rows, wrong):
                key = (slug, row["q"], method)
                if key not in extra:
                    key = ("", row["q"], method)
                if key not in extra:
                    continue
                score, cost = extra[key]
                scores.append(score)
                labels.append(is_wrong)
            if scores:
                methods[method] = (scores, cost)
                method_wrong = labels
            else:
                method_wrong = wrong

        for method, (scores, cost) in methods.items():
            labels = wrong
            if len(scores) != len(wrong):
                labels = [
                    wrong[i]
                    for i, row in enumerate(rows)
                    if (slug, row["q"], method) in extra or ("", row["q"], method) in extra
                ]
            results.append(
                {
                    "model": slug,
                    "model_name": NAME[slug],
                    "method": method,
                    "cost_x": cost,
                    "auc": auc(scores, labels),
                    "catch_30": catch_at(scores, labels, 0.30),
                    "catch_50": catch_at(scores, labels, 0.50),
                    "n": len(scores),
                    "wrong": int(sum(labels)),
                }
            )

    args.out_json.write_text(json.dumps(results, indent=2) + "\n", encoding="utf-8")
    lines = [
        "# Baseline benchmark",
        "",
        "Trace-only comparison on committed TriviaQA runs. Higher AUC/catch is better; cost is relative to one greedy local-model answer.",
        "",
        "`workspace/combined E4B zero-shot` uses the E4B-trained router weights with each target model's frozen normalization stats. `workspace/combined router` uses the per-model released router weights and is a deployment sanity check, not an out-of-fold estimate. For the 5-fold CV proof layer, use `python analyze_router.py`.",
        "",
        "| model | method | cost | AUC | wrong caught @30% | wrong caught @50% |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for rec in results:
        lines.append(
            f"| {rec['model_name']} | {rec['method']} | {rec['cost_x']:.1f}x | "
            f"{rec['auc']:.3f} | {rec['catch_30']:.0%} | {rec['catch_50']:.0%} |"
        )
    args.out_md.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"wrote {args.out_json}")
    print(f"wrote {args.out_md}")


if __name__ == "__main__":
    main()
