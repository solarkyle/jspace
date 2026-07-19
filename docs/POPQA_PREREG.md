# PopQA run: registered predictions

Written and committed BEFORE any PopQA trace exists (see commit timestamp).
Protocol: 500 PopQA questions (seed-0 shuffle of akariasai/PopQA), same
uncertainty gauntlet as TriviaQA, 7 models (E4B, 12B, 12B-ablit, 26B-MoE,
Qwen-27B, Mistral-24B at bf16; 31B at NF4). Analysis: frozen TabFM config
(32|native|z) and logistic, feature sets logprob / workspace / combined,
5-fold CV per model. No new tuning of anything.

Lessons applied from the retracted calibration-gap law: predictions are for
ABSOLUTE AUCs and rates, not for differences of coupled quantities.

## P1 (the ceiling hypothesis)

On TriviaQA, combined detection saturated at 0.84 +/- 0.02 for every
standard-protocol model; we hypothesize that ceiling is TriviaQA's own
9-15 percent label-noise floor. PopQA labels are generated from Wikidata
triples and should be cleaner.

Prediction: mean combined AUC across the 7 models EXCEEDS 0.86, and at
least 4 of 7 models individually exceed their TriviaQA combined AUC.
Falsified if mean combined stays at or below 0.85.

## P2 (the miscalibration hypothesis, from docs/HALLUCINATION_PLAN.md
Phase 2, written 2026-07-07 and never yet tested on PopQA)

Long-tail entities are where output confidence is known to miscalibrate.

Prediction: logprob-only AUC DROPS on PopQA relative to TriviaQA for at
least 5 of 7 models (their TriviaQA logprob TabFM values: E4B .772,
12B .779, ablit .775, MoE .741, Qwen .849, 31B .720, Mistral .843), while
workspace-only AUC stays within 0.05 of its TriviaQA value for at least
5 of 7. Falsified if logprob holds and workspace drops instead.

## P3 (the anatomy hypothesis)

Models improvise more about rare entities than about common ones.

Prediction: the fabrication share of real errors (LLM-graded, same rubric
as before) RISES on PopQA vs TriviaQA for the graded models (TriviaQA:
E4B 19 percent, 12B 24 percent, Qwen 10 percent), and the workspace flag
rate on errors rises with it. Falsified if fabrication share stays flat
or falls.

## Scoring

Each prediction is scored HIT / MISS / PARTIAL in an update to this file,
whatever the outcome. Analysis script: analysis/analyze_popqa.py (to be committed
with the exact rule set before analysis runs).

## VERDICTS (2026-07-10, analysis/analyze_popqa.py on the committed traces)

- P1: HIT. Mean combined 0.920 (predicted >0.86); 7/7 beat their TriviaQA
  combined. The 0.84 saturation was TriviaQA's label-noise floor.
- P2: MISS on both clauses. Logprob AUC ROSE on 7/7 (+0.02 to +0.11);
  workspace rose too (+0.08 to +0.18), so "held within 0.05" also failed
  in the good direction. Interpretation: PopQA errors are dominated by
  unfamiliar entities, which models detect well (consistent with our own
  fake-entity result, which we should have weighted over the literature
  when registering). The logprob blind spot lives in familiar-but-wrong
  substitutions, which TriviaQA is rich in and PopQA is not.
- Unregistered observation, flagged as such: workspace-over-logprob
  increment is positive on ALL 7 models on PopQA, including Qwen (+0.02)
  and Mistral (+0.03); the earlier "edge only on Gemmas" conclusion was
  dataset-dependent. Cross-dataset AUC comparisons carry a base-rate
  caveat (PopQA accuracy 14-25% vs TriviaQA 43-67%).
- P3 pending LLM grading of wrong-answer samples.
