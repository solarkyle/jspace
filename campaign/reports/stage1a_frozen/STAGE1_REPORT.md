# Stage 1 report: Gate A is a HIT (2026-07-10)

18,220 prompts, Gemma-4-12B bf16, greedy, two-pass lens capture. Eight datasets
across five domains. Pre-registration: campaign/PREREG_STAGE1.md (written blind).

## Gate A (the pre-registered decision): HIT

On LightGBM (the registered production model), combined workspace+logprob beats
logprob-only under leave-one-dataset-out:

- mean LODO increment +0.0590 (needed >= +0.02)
- breadth 7/7 datasets positive = 1.00 (needed >= 0.70)
- per-dataset increment: drop +0.047, halubench +0.057, hotpotqa +0.044,
  medhallu +0.151, popqa +0.091, squad_v2 +0.011, trivia_qa +0.012

Logistic also HITs: mean +0.0893, breadth 6/7 (only trivia_qa slightly
negative, -0.045). The workspace signal predicts Gemma's errors on every
held-out dataset it never trained on.

## Workspace generalizes better than output confidence

Cross-dataset (LODO), workspace-only beats logprob-only:
- LightGBM: logprob 0.728, workspace 0.775, combined 0.787 (ws over lp +0.047)
- logistic: logprob 0.689, workspace 0.778, combined 0.778 (ws over lp +0.089)

Output confidence is dataset-specific (identity probe 0.80, margin +0.62); the
workspace is more invariant. This replicates the pilot finding at scale and is,
if anything, the more interesting result than the raw increment: the internal
signal transfers across distributions where logprobs do not.

## Full bakeoff (LODO combined AUROC)

| model      | logprob | workspace | combined | ws increment |
|------------|--------:|----------:|---------:|-------------:|
| logistic   | 0.689   | 0.778     | 0.778    | +0.089 |
| LightGBM   | 0.728   | 0.775     | 0.787    | +0.059 |
| CatBoost   | 0.724   | 0.710     | 0.755    | +0.031 |
| MLP        | 0.714   | 0.744     | 0.769    | +0.055 |

(deterministic-label bakeoff values; judged-set adds halubench/medhallu.)

Registered-prediction outcomes:
- CatBoost ~= LightGBM (control confirmed: banned categoricals leave it no edge).
- MLP competitive but does not beat LightGBM's combined; stays a distillation
  vessel only.
- TabFM: did not complete on 14k rows within a 500s budget (in-context tabular
  model is O(n) at inference). Consistent with the registered expectation that
  its small-n advantage does not carry to this scale; a clean head-to-head needs
  a batched-inference harness, deferred.
- Temporal CNN over the depth trajectory: +0.026 order advantage over static
  summaries (mean 0.620 vs 0.594), positive but modest. Trajectory order carries
  a little extra signal at scale; not enough alone to beat LightGBM-on-summaries.

Kill rule (handoff 12.7): no teacher beat LightGBM's combined cross-dataset, so
the ensemble and distillation stages are discarded rather than run.

## Grading

14,106 of 18,220 rows graded deterministically (exact-source accuracy 0.632).
The grounded/medical rows were judged blinded by Sonnet 5 (validated at Cohen's
kappa 0.84 vs Fable in the pilot); 21 of 30 judge shards completed under a
rate-limit storm, resolving 2,471 rows (303 ambiguous excluded), lifting the
labeled set to 16,577. The remaining ~1,643 rows (9 unfinished shards plus
therapy) are unlabeled and excluded; the verdict is already 7/7 without them.

Recurring judge finding (both Sonnet and Fable, pilot and Stage 1): HaluBench's
aggregated references (DROP/PubMedQA-derived) frequently contradict their own
context. Judges graded on the shown context and flagged the conflict. Gemma's
genuine errors were dominated by false refusals on answerable questions and
truncated financial calculations, not fabrication.

## Therapy note

ESConv factual-error rate 0.00 (degenerate class, excluded from LODO). Judges
flagged several crisis-handling gaps (incomplete safety response to disclosed
ideation) as a separate safety signal, kept distinct from the factuality label.
A subset of therapy rows tripped API usage-policy filters during judging (crisis
content); these were dropped without affecting Gate A.

## Cost

Generation ~$35 of Modal (8 shards, L40S, long-context tail ran hotter than the
pilot's $2/1k). All grading, judging, and the bakeoff were free (local + Claude
subagents + local GPU). TabFM and any Stage 1b remain within remaining budget.

## Bottom line

The pre-registered central test passed: train on many source distributions,
freeze, and predict Gemma-12B's errors on entirely unseen datasets without
dataset identity. Workspace features add real, broad cross-dataset value
(+0.059 mean, 7/7 datasets) and generalize better than output confidence. This
is the result that moves the project from "works within two QA datasets" to
"transfers across domains," and it is the backbone of the paper.
