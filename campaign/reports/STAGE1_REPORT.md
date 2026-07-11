# Stage 1 report, corrected (2026-07-10, post peer review)

18,220 prompts, Gemma-4-12B bf16, greedy, two-pass lens capture. Eight datasets
across five domains. Pre-registration: campaign/PREREG_STAGE1.md (written blind,
pre-Stage-0; caveat: the file is untracked by git, so prospective but not
immutable). This report supersedes the original Stage 1 report, which is frozen
verbatim with its artifacts in campaign/reports/stage1a_frozen/.

What changed between Stage 1a and this report:
1. An external adversarial review (campaign/PEER_REVIEW_CODEX.md) reproduced the
   headline and found two real bugs plus several overclaims. All conceded.
2. Bug fix: the judge confidence rule (frozen prompt: verdicts below 0.7 are
   ambiguous) was not enforced at ingestion; 399 low-confidence labels leaked
   into training. Now enforced; all numbers below are on the corrected labels.
3. Bug found (not yet re-run): the SQuAD v2 adapter emitted 3,000 all-answerable
   rows, zero unanswerable. The "SQuAD unanswerable" framing is invalid; that
   dataset is ordinary grounded QA here. Adapter fixed in code; regeneration on
   Modal is pending, so squad_v2 rows in this report are grounded QA only.

## Scope caveat, stated up front

The prereg targeted >= 12 sources. Stage 1 ran 8, of which 7 are evaluable
(ESConv therapy has a factual-error rate of 0.00, degenerate for this label).
So the honest claim is: Gate A passed on 7 evaluable dataset wrappers under a
reduced-source run, not "the complete preregistered campaign passed."

## Gate A (the pre-registered decision): HIT

On LightGBM (the registered production model), combined workspace+logprob beats
logprob-only under leave-one-dataset-out, on confidence-corrected labels:

- mean LODO increment +0.0651 (needed >= +0.02)
- breadth 6/7 datasets positive = 0.857 (needed >= 0.70)
- per-dataset increment: drop +0.062, halubench +0.098, hotpotqa +0.049,
  medhallu +0.149, popqa +0.087, squad_v2 -0.003, trivia_qa +0.015

The single non-positive dataset is squad_v2 at -0.003, which is also the
known-invalid row (all-answerable bug). Logistic also HITs: mean +0.0908,
breadth 6/7 (trivia_qa -0.050).

For the record, the pre-fix Stage 1a numbers were +0.0590 with 7/7 breadth;
enforcing the confidence rule strengthened the mean and moved squad_v2 from
+0.011 to -0.003. The gate holds under both labelings.

## Robustness (from the external review, on pre-fix labels)

The reviewer independently reproduced +0.0590 (7/7) and stress-tested it:
- excluding low-confidence judge labels: +0.065 (6/7)
- excluding 272 cross-dataset duplicate questions: +0.057 (7/7)
- both controls together: +0.0635 (7/7)
- splitting HaluBench into its true upstream sources: +0.0596 (9/10; only the
  small HaluBench/RAGTruth slice went ~-0.007)

The effect is ~+0.06 under every control tried so far.

## Workspace generalizes better than output confidence

Cross-dataset (LODO), workspace-only beats logprob-only on corrected labels:
- LightGBM: logprob 0.731, workspace 0.789, combined 0.796 (ws over lp +0.058)
- logistic: logprob 0.698, workspace 0.792, combined 0.789 (ws over lp +0.094)

Output confidence is dataset-specific; the workspace signature transfers. Note
the identity probe: a classifier predicts the source dataset from the features
at 0.811 accuracy vs 0.185 majority baseline (margin 0.625). Per the prereg,
this margin exceeds the 0.25 threshold, so pooled (non-LODO) numbers should not
be trusted as generalization evidence; LODO is the primary result throughout,
and it is exactly the design that survives this leakage.

## Gate B (operational routing value): marginal HIT, fragile

Registered: at a fixed 20 percent routing budget, combined catches >= 5pp more
wrong answers than logprob-only, mean over held-out datasets, bootstrap 95% CI
excluding zero. Scored by campaign/score_gate_b.py (LightGBM, LODO, cluster
bootstrap over split_groups, 2000 reps) on corrected labels:

| held-out   | catch lp | catch comb | delta |
|------------|---------:|-----------:|------:|
| drop       | 38.8%    | 41.1%      | +2.27 |
| halubench  | 43.2%    | 48.6%      | +5.45 |
| hotpotqa   | 48.1%    | 52.5%      | +4.42 |
| medhallu   | 56.0%    | 76.0%      | +20.00 |
| popqa      | 22.2%    | 23.1%      | +0.90 |
| squad_v2   | 62.3%    | 63.6%      | +1.32 |
| trivia_qa  | 42.9%    | 44.2%      | +1.35 |

Mean +5.10pp, CI [+3.15, +7.36]. That satisfies the registered criterion
(>= +5pp, CI > 0) by 0.10pp, so formally: HIT. Honest gloss: this is marginal
and concentration-driven. medhallu contributes +20pp; excluding it the mean is
+2.62pp. Excluding instead the invalid squad_v2 row it is +5.73pp. The reviewer
scored +4.19pp (MISS) on the pre-fix labels, so the verdict flips with the
label correction and sits on the boundary. Fair summary: routing value is real
(CI well above zero under every cut) but its magnitude is dataset-dependent and
largest where it matters most (medical); the registered +5pp bar is met only
marginally. Treat as borderline, not as a robust operational win.

## Gate C (early warning): NOT TESTED

The temporal CNN result (+0.026 for layer-depth order over static summaries)
tests trajectory-order information, not 50%-prefix forecasting. True prefix
Gate C, predicting the error before the answer is half-generated, remains open.

## Full bakeoff (LODO, corrected labels, all four models on all 7 datasets)

| model    | logprob | workspace | combined | ws increment |
|----------|--------:|----------:|---------:|-------------:|
| CatBoost | 0.728   | 0.787     | 0.800    | +0.072 |
| LightGBM | 0.731   | 0.789     | 0.796    | +0.065 |
| MLP      | 0.707   | 0.795     | 0.794    | +0.087 |
| logistic | 0.698   | 0.792     | 0.789    | +0.091 |

The top tier is effectively tied (no paired CIs computed; spread 0.796-0.800 is
within noise). LightGBM stays the production pick for simplicity and size, not
because it won. The earlier claim that no teacher beat it is corrected: CatBoost
edges it by +0.004; under the kill rule's spirit (no material teacher advantage)
ensemble/distillation stages remain discarded.

- TabFM: did not complete on 14k rows within a 500s budget (in-context model,
  O(n) inference). Consistent with the registered expectation; a clean
  head-to-head needs a batched harness, deferred.
- Temporal CNN: +0.026 order advantage (0.620 vs 0.594), positive but modest.

## Grading (corrected)

14,106 of 18,220 rows graded deterministically (exact-source accuracy 0.632).
Grounded/medical rows were judged blinded by Sonnet 5 (validated at Cohen's
kappa 0.84 vs Fable in the pilot); 21 of 30 judge shards completed under a
rate-limit storm. With the confidence rule enforced, 2,072 rows resolved via
judge and 702 excluded as ambiguous (the original ingestion had wrongly
accepted 399 of those). Labeled set: 16,178. The remaining ~1,643 rows (9
unfinished shards plus therapy) are unlabeled and excluded.

Recurring judge finding (both Sonnet and Fable, pilot and Stage 1): HaluBench's
aggregated references (DROP/PubMedQA-derived) frequently contradict their own
context. Judges graded on the shown context and flagged the conflict. Gemma's
genuine errors were dominated by false refusals on answerable questions and
truncated financial calculations, not fabrication.

## Therapy note

ESConv factual-error rate 0.00 (degenerate class, excluded from LODO). Judges
flagged several crisis-handling gaps as a separate safety signal, kept distinct
from the factuality label. A subset of therapy rows tripped API usage-policy
filters during judging (crisis content); dropped without affecting Gate A.

## Deployment status: NOT production-ready

Nothing is deployable yet: no exported/frozen model artifact, no calibration,
no frozen feature schema, no sidecar integration, no measured lens overhead.
The 31 features include onset/mid/end prefix features, so inference needs three
J-space reads per response, not one. The earlier "286 KB, microseconds" line
was a node-count estimate, not a benchmarked artifact. Per-dataset operating
points (catch@10%FP: medical 62%, trivia 56%, hotpot 54%, drop 43%, popqa 40%)
describe the classifier, not a shipped system.

## Cost

Generation ~$35 of Modal (8 shards, L40S, long-context tail ran hotter than the
pilot's $2/1k). Grading, judging, and all bakeoffs were free (local).

## Bottom line

The central pre-registered test passed on reduced scope: train on multiple
source distributions, freeze, and predict Gemma-12B's errors on held-out
datasets. The workspace increment is ~+0.06 and survived an adversarial
reproduction with dedup, confidence, and upstream-split controls. Workspace
features generalize across datasets better than output confidence, which is the
core conceptual claim. Gate B is a marginal, fragile HIT; Gate C is untested;
the models are tied; SQuAD needs regeneration; nothing is production-ready.
The strongest available next step is prospective validation: freeze one
classifier and score it zero-shot on genuinely new datasets it has never seen.
