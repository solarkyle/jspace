# Stage 0 pilot report (2026-07-10)

496 prompts, google/gemma-4-12B-it bf16, one L40S. This run validates the
pipeline, measures cost, and surfaces what Stage 1 must fix. It was never sized
to pass Gate A.

## 1. Throughput and cost (the number that sizes Stage 1)

- 496 prompts in 1845.7 s (30.8 min), 0.269 prompts/s on one L40S.
- 88,536 prompt tokens + 11,697 gen tokens; mean answer 23.6 tokens.
- Cost: ~$1.00 for the pilot = **$2.02 / 1,000 prompts** blended.
- The rate is dominated by long-context grounded rows (SQuAD/HaluBench/MedHallu
  passages up to 2048 tokens); teacher-forced capture over ~1.5k positions x 24
  band layers is the bottleneck, not generation. Terse-QA-only rows are much
  cheaper.

**Stage 1 sizing:** $50 usable (holding ~$11 of the $61 in reserve) buys
~24,000 prompts at the blended pilot rate. Batching similar-length prompts (not
done in the pilot; one prompt at a time) is the obvious lever to push that
higher, and long-context shards should route to a separate queue. Target Stage 1
at ~18-20k prompts across >=12 datasets to leave headroom for retries.

## 2. Two-pass equivalence (re-confirmed at n=30)

Feature deltas teacher-forced vs autoregressive: ignition_frac 0.000,
band_agreement 0.000, ignition_depth 0.000, mean_log_rank_answer 0.030,
mean_entropy 0.024. Raw-logit max abs 0.81 (bf16 kernel noise). The consumed
features are stable; the two-pass optimization is trusted.

## 3. Deterministic grading

317 / 496 rows graded deterministically; 179 need a judge (all 40 ESConv, most
HaluBench/MedHallu). Exact-source accuracy 0.487 (n=300), consistent with the
known 12B TriviaQA rate. Per-source error rate among labeled rows: PopQA 0.89
(long-tail, as expected), TriviaQA 0.53, SQuAD 0.12 (mostly abstention grading).

## 4. Static bakeoff (pilot, exact-graded rows only)

|            | pooled grouped-CV | LODO mean |
|------------|------------------:|----------:|
| logistic logprob   | 0.873 | 0.678 |
| logistic workspace | 0.903 | 0.690 |
| logistic combined  | 0.909 | 0.683 |
| lightgbm logprob   | 0.873 | 0.558 |
| lightgbm workspace | 0.890 | 0.632 |
| lightgbm combined  | 0.907 | 0.628 |

- **Pooled workspace increment +0.036 (logistic) / +0.034 (lightgbm).** The
  within-distribution finding REPLICATES cleanly in the new pipeline.
- **LODO is noisy and underpowered here** (+0.005 logistic, +0.070 lightgbm):
  only 3 exact-graded datasets contribute test folds with both classes, and the
  HaluBench/MedHallu alias-prepass folds are degenerate (n=13/4, 0 errors). No
  Gate A verdict can or should be read from this. That is the pilot working as
  intended, not a failure.

## 5. The finding that justifies the whole campaign

Dataset-identity leakage: a classifier predicts the source dataset from the
deployable features at **0.94 accuracy vs a 0.315 majority baseline (margin
+0.625)**. Much of that is legitimate (PopQA logprobs and grounded-QA lengths
genuinely differ), but it confirms the pooled numbers are optimistic relative to
true cross-dataset deployment. Our historical 0.92 (PopQA) was within-dataset
5-fold CV; the honest metric is LODO, and LODO is exactly what this campaign was
built to measure. The pilot demonstrates the gap is real and large.

## 6. Fixes applied / carried into Stage 1

- FIXED: grouped_kfold had a shuffle/inverse-permutation bug that leaked 8
  groups across folds; replaced with StratifiedGroupKFold (verified group-pure).
- CARRY: the 179 llm-graded rows need blinded judging before medical/therapy
  contribute labels; harness (grade_claude.py) and frozen prompts are ready.
- CARRY: Stage 1 needs enough errors per dataset to avoid degenerate LODO folds;
  size each source for >=30 errors in expectation.
- CARRY: batch generation + separate long-context queue to lower $/1k.

## 7. Stage 0 acceptance checklist (handoff 19)

Pinned revisions (yes), existing 500 IDs excluded (yes), no dup prompts (yes),
idempotent response IDs (yes), teacher-forced == autoregressive at the checkpoint
(yes, section 2), no fold leakage (yes, after fix), deterministic grader unit
tests pass (13/13), trace storage compact (496 rows, scalar features only),
static baselines train end to end (yes). Judge-JSON-validity and the greedy
equivalence-to-old-runner checks move to the judge step and are pending.
