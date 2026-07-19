<div align="center">

# jspace

**Do LLM internal signals actually detect errors?
Preregistered reliability research on open models.**

[![reproduce](https://github.com/solarkyle/jspace/actions/workflows/reproduce.yml/badge.svg)](https://github.com/solarkyle/jspace/actions/workflows/reproduce.yml)
[![release](https://img.shields.io/github/v/release/solarkyle/jspace)](https://github.com/solarkyle/jspace/releases)
[![license](https://img.shields.io/github/license/solarkyle/jspace)](LICENSE)
[![data](https://img.shields.io/badge/%F0%9F%A4%97%20data-24.5k%20graded%20traces-yellow)](https://huggingface.co/datasets/solarkyle/jspace-hallucination-campaign)
[![demo](https://img.shields.io/badge/demo-live-blue)](https://solarkyle.github.io/jspace/demo/)

[Try the demo](https://solarkyle.github.io/jspace/demo/) · [Reproduce in 90s](#reproduce-in-90-seconds-cpu-only) · [Reports](campaign/reports/) · [Data](https://huggingface.co/datasets/solarkyle/jspace-hallucination-campaign) · [Contributing](CONTRIBUTING.md)

</div>

**Finding:** a ~300 KB LightGBM classifier over Jacobian-lens workspace
readouts predicts Gemma-4-12B's wrong answers better than the model's own
output confidence, transfers across datasets within a task family, and fails a
pre-registered universal-transfer test in an informative way: transfer holds
on grounded and retrieval QA but breaks down on veracity-judgment tasks.

## Why it matters

Output logprobs are the standard cheap hallucination signal, but they are
dataset-specific: a confidence threshold tuned on one benchmark does not carry
to the next. An internal signal read from the residual stream through the
[Jacobian lens](https://github.com/anthropics/jacobian-lens) (Anthropic's
[global workspace paper](https://transformer-circuits.pub/2026/workspace/index.html),
July 2026) generalizes across datasets within a task family better than
output confidence does, at one extra forward-pass read. Just as important, the campaign's prospective
failure maps where such monitors stop working, and surfaces a confound that
can affect other published probe results.

## The campaign, in four pre-registered gates

25,340 prompts, 13 public benchmarks, 6 domains, Gemma-4-12B, greedy. Each
gate's threshold was registered before the data that scores it existed;
classifiers were frozen and SHA-256 hashed before the Stage 2 datasets were
generated.

| Gate | Registered test | Result |
|---|---|---|
| A | Combined (workspace+logprob) beats logprob-only under leave-one-dataset-out: mean AUROC increment >= +0.02, breadth >= 70% | **HIT**: +0.065, positive 6/7 datasets. Workspace-only 0.789 vs logprob-only 0.731 LODO |
| B | At a 20% routing budget, combined catches >= 5pp more errors, bootstrap CI > 0 | **Marginal HIT**: +5.1pp, CI [+3.2, +7.4]; concentration-driven (medhallu +20pp; +2.6pp without it) |
| C | 50%-prefix early warning | **Not tested** (prefix features exist in the traces; open) |
| D | Frozen classifiers, zero-shot on 5 never-seen datasets: mean increment >= +0.02, positive 4/5 | **MISS**: mean -0.016, 2/5. Transfers on grounded and retrieval QA (facts_grounding +0.063, nq_open +0.023); the increment goes negative on veracity and tool tasks (legal -0.107) |

Stage 1 numbers survived an independent adversarial audit and reproduction
(Codex / GPT-5.5, [campaign/PEER_REVIEW_CODEX.md](campaign/PEER_REVIEW_CODEX.md)):
the increment stayed at ~+0.06 under every control tried (judge-confidence
filtering, cross-dataset dedup, upstream-source splits).

Interpretation of the Gate D miss, kept separate from the numbers: by our own
registered test this is a per-task-family monitor, not a universal detector.
Why it breaks on veracity tasks is not settled, and two readings compete. One
is a genuine sign flip: in retrieval, internal fog accompanies error; in
veracity judgment ("is this legal case real?"), fog may instead accompany
correctly rejecting a fabrication, with confident fluency marking the errors
(same signal, opposite label). The other is the answer-readout confound
below: the legal inversion is concentrated on fixed-truth slices, where the
score reads answer identity rather than error, and the one varying-truth
legal slice is 0.582 in-domain — weakly aligned, not inverted. A powered
within-model follow-up (in preparation) looked for this sign flip on a
retrieve-vs-reason axis and did not find it — the signal collapses toward
chance off retrieval rather than inverting — while the veracity-style axis
was too degenerate to test. So the sign flip stays an open hypothesis; the
per-task-family scoping is the result.

## What failed, and the answer-readout confound

The most reusable methodological result came from auditing our own favorable
numbers. On any evaluation slice where the correct answer is constant (case
existence: always yes; unanswerable detection: abstain is always right), an
internal-features probe reaches AUC ~1.0 by reading which answer the model is
about to emit, not by detecting error: the label is a deterministic function
of the answer. Our in-domain probes hit 1.000/1.000/0.991 on three such
slices; honest varying-truth numbers on the same datasets are 0.55-0.78. If
you publish probe AUCs on fixed-truth slices, decompose them first. Details:
[STAGE2_REPORT](campaign/reports/STAGE2_REPORT.md).

## Reproduce in 90 seconds (CPU only)

```bash
pip install lightgbm numpy huggingface_hub
git clone https://github.com/solarkyle/jspace && cd jspace
python campaign/reproduce_mini.py
```

The script verifies the committed frozen classifiers against the SHA-256
hashes recorded in the pre-registration, downloads the Stage 2 feature table
from the HF dataset, rescores all 7,113 labeled rows zero-shot, and prints the
recomputed Gate D table next to the published numbers with a PASS/FAIL match
column (tolerance 0.002).

## Links

- Technical reports: [Stage 1](campaign/reports/STAGE1_REPORT.md),
  [Stage 2](campaign/reports/STAGE2_REPORT.md),
  [Stage 0](campaign/reports/STAGE0_REPORT.md)
- Pre-registrations: [PREREG_STAGE1.md](campaign/PREREG_STAGE1.md),
  [PREREG_STAGE2.md](campaign/PREREG_STAGE2.md)
- Independent adversarial audit and reproduction (Codex / GPT-5.5):
  [campaign/PEER_REVIEW_CODEX.md](campaign/PEER_REVIEW_CODEX.md)
- Fitted lenses, routers, frozen classifiers:
  [solarkyle/jspace-lenses](https://huggingface.co/solarkyle/jspace-lenses)
- All 24.5k graded traces, features, judge verdicts:
  [solarkyle/jspace-hallucination-campaign](https://huggingface.co/datasets/solarkyle/jspace-hallucination-campaign)
- Campaign code: [campaign/](campaign/)

## Earlier explorations

The campaign grew out of a same-day replication of the workspace paper on
open models plus a set of exploratory cross-model findings (covert-emotion
readouts, abliteration effects, the original TriviaQA router, an interactive
demo). Those results, at their varying evidence levels, are preserved in
[docs/EARLY_EXPLORATIONS.md](docs/EARLY_EXPLORATIONS.md), with the
[16GB-consumer-GPU fitting recipe](docs/GPU_NOTES.md) and the
[cross-model findings report](docs/FINDINGS.md). Interactive demo:
[solarkyle.github.io/jspace/demo](https://solarkyle.github.io/jspace/demo/).

## Credits

Paper and reference implementation: Anthropic
([transformer-circuits.pub/2026/workspace](https://transformer-circuits.pub/2026/workspace/index.html),
[anthropics/jacobian-lens](https://github.com/anthropics/jacobian-lens),
Apache 2.0). Models: Google (Gemma 4), Alibaba (Qwen3.6), huihui-ai
(abliteration). Scripts in this repo: MIT.

## Contributing

Contributions welcome — and most need **no GPU**: all traces, features, and
fitted lenses are public on HF, so analysis ideas run on a laptop. Start with
[CONTRIBUTING.md](CONTRIBUTING.md), the
[`analysis idea` issue template](https://github.com/solarkyle/jspace/issues/new?template=analysis-idea.yml),
or [Discussions](https://github.com/solarkyle/jspace/discussions). CI reruns
the full 90-second reproduction on every pull request.

By [@solarkyle](https://github.com/solarkyle). Contact: fintechkyle@gmail.com.
