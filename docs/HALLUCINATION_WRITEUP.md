# Workspace Noise Catches Overconfident Hallucinations

Draft public writeup. Lead with this result; keep the emotion probes as a
secondary gallery until they have more prompts per condition.

## Thesis

Anthropic's J-space paper shows that verbalizable vectors form a causal
workspace inside language models. This repo asks a practical follow-up:

> When a small open model is about to answer, does the shape of its workspace
> tell us whether it is retrieving cleanly or fabricating?

On Gemma-family models, yes. A noisy, high-entropy workspace trajectory predicts
confident wrong answers better than output confidence alone in output
confidence's own blind spot.

## What is new here

Not new: decoding hidden states into words. Logit lens, tuned lens, and
Anthropic's Jacobian lens already do that.

New enough to lead with:

- A single-answer internal signal for noisy retrieval, using the model's own
  vocabulary space and no trained hidden-state probe.
- The signal works where logprobs are supposed to fail: confident wrong answers.
- The router transfers across Gemmas with fixed normalization/weights, and it
  fails visibly on Qwen, whose output confidence is already well calibrated.
- The deployment path is concrete: an OpenAI-compatible sidecar can flag,
  refuse, tag, or escalate local-model answers.

## Core result

On 500 TriviaQA questions per model, the most legible cut is the confident
quadrant:

| Model | confident + clean | confident + noisy | gap |
|---|---:|---:|---:|
| Gemma E4B | 77% correct | 42% correct | +35 pt |
| Gemma 12B | 83% | 47% | +36 pt |
| Gemma 12B abliterated | 79% | 63% | +16 pt |
| Gemma 26B MoE | 91% | 71% | +20 pt |
| Qwen 27B | 85% | 87% | -3 pt |

The honest read is not "J-space detects truth." It detects noisy retrieval. A
model that confidently believes a false fact can still look clean.

## Why output confidence is not enough

Output confidence is strong on easy unanswerables and fake entities. That is a
negative result in this repo: unfamiliar fabricated names are not the workspace
use case because capable models already show uncertainty in their logits.

The workspace matters on real-but-hard questions where the output token is
confident anyway. In those cases the final logits have already saturated, while
the mid-layer workspace can still look like name-soup: many unrelated candidate
concepts with probability smeared into the tail.

## Current benchmark surface

Use:

```bash
python analysis/analyze_router.py
python analysis/benchmark_baselines.py
python analysis/score_expensive_baselines.py --n 100
python analysis/benchmark_baselines.py --extra-scores data/expensive_baselines.jsonl
```

`analysis/analyze_router.py` is the proof layer: 5-fold CV, out-of-fold scores, output
baselines vs workspace trajectory features vs combined.

`analysis/benchmark_baselines.py` is the deployment/cost table. It uses committed traces,
published router weights, and can import future semantic-entropy or P(True)
scores as JSONL. The pitch to test next is not "best possible AUC"; it is
"competitive signal at roughly 1x answer cost, while semantic entropy costs
multiple sampled generations."

`analysis/score_expensive_baselines.py` generates two optional baselines now:
P(True)-style self-evaluation and sampled-answer entropy. The latter is not full
semantic entropy until sampled answers are clustered by meaning, but it is the
runnable sampling baseline that lets the cost comparison start.

## Deployment guardrail

The original experiments read one answer-onset snapshot. The sidecar now reads
the first `WORKSPACE_READ_TOKENS` generated token workspaces, default 3, and
routes on the highest-risk one. This handles the practical failure where a model
starts with filler text and only reaches the actual answer on token two or
three.

Keep `LOCAL_TERSE=1` for production-style use. The router was validated on
answer-leading generations; the prefix scan is a guardrail, not a license to
score long preambles as if they were the original experiment.

## Causal next step

Use:

```bash
python analysis/causal_hint_patch.py --n 24
```

This script runs a first causal bridge experiment: take noisy wrong questions,
construct a paired clean prompt by adding the correct answer as a hint, patch
the hinted residual delta into the original answer-onset run across workspace
layers, and measure whether the correct answer logit rises.

If this works, the next version should replace hint-delta patching with sparse
J-space coordinate swaps or tail-smear ablations. That is the path from "useful
correlate" to "workspace state is causally load-bearing for hallucination."

## Caveats to keep prominent

- Qwen is the miss. Its logprobs are already highly calibrated, and the
  workspace signal does not add value there.
- The emotion results are suggestive, not the lead claim: one prompt per
  condition is not enough.
- The sidecar's multi-token read changes deployment cost. Measure real overhead
  before claiming production economics.
- "Workspace" means the J-lens approximation of verbalizable workspace content,
  not a claim about subjective experience.
