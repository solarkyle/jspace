# Hallucination prediction: where it stands and the plan to keep pushing

> **STATUS UPDATE 2026-07-07 (evening): Phases 1 and 2 are DONE.** Phase 1 gate
> passed 3/4 (fails on Qwen 27B, whose output confidence is already calibrated);
> threshold transfers across all Gemmas untuned. Phase 2 was a negative result
> for workspace-vs-logprob on fake entities, with a striking behavioral finding:
> abliteration converts refusal into fabrication (17/50 -> 49/50). Full numbers
> in [FINDINGS.md](FINDINGS.md) Part 4, `analysis/analyze_crossmodel.py` reproduces.
> Update after repo hardening: Phase 3 trajectory features are done
> (`analysis/analyze_router.py`), the sidecar is running with a prefix-read guardrail,
> and `analysis/benchmark_baselines.py` now provides the trace-only cost/baseline table.
> `analysis/score_expensive_baselines.py` can generate P(True)-style and sampled-answer
> entropy scores for import. The remaining open item is to run those expensive
> baselines at scale, add true semantic-entropy clustering if desired, and run
> the new causal intervention script.

Status after v2 (2026-07-07): on 500 TriviaQA questions through Gemma 4 E4B,
workspace features predict correctness at CV-AUC 0.746 (baseline 0.713,
combined 0.778). The strongest single result is the quadrant cut: answers where
the output logit is confident but the workspace is noisy are 42% correct vs 75%
when the workspace is clean. Confound checks passed (not answer length, not
question type, only weakly correlated with output confidence).

The honest gap: on TriviaQA the combined escalation router only beats a
logprob-only router by about a point at any fixed budget, because TriviaQA
output confidence is already well calibrated. The whole bet of this line of
work is that the workspace advantage concentrates where output confidence is
miscalibrated. So the plan is built around finding out if that is true.

## Phase 1: does it replicate across models? (~$15-25 Modal, no new lenses needed)

All 5 lenses are already fit. probe_uncertainty is forward-only, so this is
cheap compared to fitting.

1. Run `analysis/probe_uncertainty.py` (ported into analysis/modal_fit.py as a new entrypoint)
   for 12B, 12B-abliterated, 26B-MoE, Qwen-27B. 500 TriviaQA each.
   A10G for the 12Bs, A100-80GB for MoE and Qwen (bf16 inference).
2. Report per-model: baseline AUC, workspace AUC, combined AUC, and the
   quadrant table. The claim "workspace catches overconfident hallucinations"
   needs to hold on at least 3 of 4 new models to survive.
3. Threshold transfer: calibrate the entropy threshold on model A (z-scored),
   apply to model B. If a single normalized threshold transfers, deployment
   gets much easier (no per-model calibration set needed).
4. Bonus question nobody has asked: is the abliterated model MORE prone to
   confident hallucination than its base, and does the workspace see it?

## Phase 2: datasets where output confidence is known to lie (~$10-20)

TriviaQA is the easy case. The money is where logprobs are miscalibrated:

1. **PopQA** (long-tail entities): small models hallucinate confidently about
   rare entities. Prediction: baseline AUC drops, workspace holds.
2. **Fake-entity probes**: questions about nonexistent people/places/papers
   ("What year did the physicist Elena Morvath win the Nobel Prize?"). The
   model cannot be right; measure how often the workspace flags fabrication
   while the output sounds fluent. This is the purest overconfident-
   hallucination test and it is easy to generate at scale.
3. **SQuAD v2 unanswerables**: context provided, answer absent. Does the
   workspace distinguish "retrieving" from "improvising"?
4. Keep 5-fold CV + report per-dataset AUCs, never pooled-only.

## Phase 3: better features (free, local, data already collected) - DONE

Current features are hand-defined stats on the mid-band. Not yet tried:

1. **Layerwise entropy trajectory**: slope and collapse depth (where entropy
   drops), not just the band mean. The "ignition" story suggests WRONG answers
   ignite late or never; the band mean blurs this.
2. **Cross-layer agreement**: KL divergence between adjacent layer readouts.
   A workspace that keeps changing its mind should predict wrongness.
3. **Per-token volatility during generation** (not just pre-answer): rank-1
   token churn across the answer span.
4. Small logistic on the full layerwise entropy vector (with CV) instead of
   hand-picked scalars, to see how much signal the hand-picking leaves behind.

## Phase 4: cost-matched comparison against published methods - SCAFFOLD DONE, EXPENSIVE BASELINES OPEN

To be taken seriously this has to be positioned against the literature:

- **Semantic entropy** (Farquhar et al., Nature 2024): needs 5-10 sampled
  generations per query (5-10x inference cost). Ours: single forward pass.
- **P(True) / verbalized confidence**: one extra forward pass, weak on
  overconfident errors (the model is asked to audit itself with the same
  weights that hallucinated).
- The pitch is not "highest AUC ever", it is "AUC competitive with sampling
  methods at 1x inference cost, using only the model's own internal state,
  no trained probe, no labels". If Phase 2 shows the fake-entity result, that
  plus cost is the paper/post.

`analysis/benchmark_baselines.py` now writes the trace-only table to
`docs/BASELINE_BENCHMARK.md` and accepts optional JSONL scores for semantic
entropy, P(True), verbalized confidence, or hidden-state probes.
`analysis/score_expensive_baselines.py` generates two such files directly:
P(True)-style self-evaluation and sampled-answer entropy. True semantic entropy
still needs clustering/judging over sampled answers.

## Phase 5: ship the escalation sidecar (the demo people can run) - DONE, OVERHEAD BENCHMARK OPEN

MVP: FastAPI, OpenAI-compatible `/v1/chat/completions` wrapper around a local
HF model + fitted lens. Every response includes
`x_workspace_confidence: float`; optional `escalate_to:` config forwards
low-confidence queries to a bigger model (any OpenAI-compatible endpoint) and
returns that answer instead, tagged with which model answered. A
r/LocalLLaMA-ready demo: Gemma E4B answering with Qwen 27B (or a cloud model)
as big brother, plus a live log showing which queries got escalated and why.

The implemented sidecar returns the score in the response `jspace` block, with
four modes (`detect`, `escalate`, `refuse`, `tag`) and a chat UI. It now reads
the first `WORKSPACE_READ_TOKENS` answer-token workspaces by default and routes
on the highest-risk one. Real serving-stack overhead still needs measurement.

## Phase 6: causal bridge - SCAFFOLD DONE, RUN OPEN

`analysis/causal_hint_patch.py` implements the first causal bridge: for noisy wrong
cases, add a correct-answer hint, patch the hinted residual delta into the
original answer-onset run across workspace layers, and measure whether the
correct answer logit rises. This is not yet a sparse J-space coordinate swap,
but it is the next concrete step from correlation toward causality.

## Decision gates (written before the runs so we stay honest)

- If Phase 1 replication fails (workspace adds nothing on 3+ models): publish
  the negative result with the E4B quadrant finding as a single-model
  observation. No cherry-picking the one model where it works.
- If Phase 2 shows no widening of the gap on miscalibrated datasets: the
  practical case for the router dies (logprobs are cheaper); the interp
  finding (workspace state correlates with knowledge access) still stands.
- If both hold: write it up properly (blog post + maybe arXiv note), because
  "single-pass hallucination flag from the model's own workspace"
  would be genuinely useful and, as far as we know from a same-day literature
  check, not published in this form.

## Budget

Phases 1-2 are cheap (forward passes only, no fitting: roughly $25-45 of
cloud GPU time total). Phase 3 is free (local). Phase 5 is free (local).
A new lens fit only becomes worth it if a new model matters (e.g. a Qwen MoE
for the dense-vs-MoE hallucination comparison).
