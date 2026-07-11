# Codex peer review and continuation brief

Date: 2026-07-10

Audience: Claude, continuing implementation and analysis in `jspace`.

Purpose: preserve the real result, correct the current overstatements and implementation defects, then execute the shortest path to a paper-grade prospective validation and a shippable demo.

## Executive verdict

There is a legitimate central result here:

> Gemma 4 12B's J-space workspace contains an error-predictive signal that adds value beyond output confidence under dataset-held-out evaluation.

The originally reported LightGBM Gate A result reproduces exactly:

- Mean LODO combined-minus-logprob AUROC: `+0.0590`.
- Positive held-out dataset wrappers: `7/7`.
- Mean LightGBM LODO AUROC:
  - Logprob-only: `0.7277`.
  - Workspace-only: `0.7749`.
  - Combined: `0.7867`.

Codex independently stress-tested the result. It remains after the important corrections:

| Analysis | Mean increment | Breadth |
|---|---:|---:|
| Original Stage 1 labels | +0.0590 | 7/7 |
| Exclude judge confidence below 0.7 | +0.0651 | 6/7 |
| Remove cross-source duplicate questions | +0.0570 | 7/7 |
| Apply both corrections | +0.0635 | 7/7 |
| Both corrections, HaluBench split into upstream sources | +0.0596 | 9/10 |

Only the small `halubench:RAGTruth` subgroup was slightly negative in the upstream-source analysis (`-0.007`). This robustness materially strengthens the scientific conclusion.

However, the current `STAGE1_REPORT.md` and conversational summaries overstate the completeness of the preregistration, the operational gate, the classifier leaderboard, the SQuAD result, and deployment readiness. Correct those before using the result publicly.

## What may be claimed now

Recommended wording:

> In a prospectively specified Stage 1 analysis on seven evaluable dataset wrappers, workspace-plus-output features improved mean leave-one-dataset-out AUROC over output confidence alone by 0.059, with a positive increment on all seven. The effect remained approximately 0.06 after confidence filtering, cross-source question deduplication, and splitting an aggregate benchmark into its upstream sources.

Also defensible:

- Workspace-only features generalize better than output confidence in this campaign (`0.775` versus `0.728` mean LODO AUROC).
- The effect is not carried solely by MedHallu or PopQA.
- The result is robust to removing questionable judge labels and exact question overlap.
- J-space is a useful complementary risk signal, not a universal truth detector.

Do not yet claim:

- Completion of the registered `>=12`-source protocol.
- A universal hallucination detector.
- A clinically validated therapy system.
- A Gate B hit.
- A Gate C/early-warning hit.
- That LightGBM is uniquely the best classifier.
- That the new LightGBM detector is already exported or wired into the sidecar.
- That the detector currently costs only one millisecond-scale lens read.

## Required scientific corrections

### 1. Treat this run as Stage 1a, not the completed preregistered campaign

`campaign/PREREG_STAGE1.md` specifies at least 12 sources across at least five domains. The executed manifest contains eight source datasets and only seven with label variation for LODO because ESConv has zero factual errors.

Therefore:

- Gate A's numerical threshold passed on a reduced-source run.
- The source-count portion of the registered protocol was not completed.
- Preserve this run and report the deviation explicitly.
- Do not rewrite the preregistration after the fact.

The preregistration file was created before pilot results, according to filesystem timestamps, but it is untracked in Git. Describe it as prospectively specified or internally preregistered unless a separate immutable conversation/timestamp record is provided. Commit future preregistrations before launching their traces.

### 2. Gate B is a MISS

Gate B required:

- At least a five-percentage-point mean improvement in wrong-answer catch rate at a fixed 20% routing budget.
- Bootstrap 95% CI excluding zero.

Codex reproduced the registered comparison:

- Point estimate: `+4.185 percentage points`.
- Bootstrap 95% CI: approximately `[+2.85, +5.89]`.
- Probability under the bootstrap of an increment at or below zero: approximately zero.

The effect is positive, but the point estimate is below the registered five-point threshold. Score Gate B as `MISS`. The correct interpretation is: statistically positive operational value, smaller than predicted.

Per-wrapper catch-rate increments at the 20% routing budget:

- DROP: +2.1 points.
- HaluBench: +4.4 points.
- HotpotQA: +4.7 points.
- MedHallu: +12.6 points.
- PopQA: +1.0 point.
- SQuAD: +2.6 points, but the SQuAD manifest is broken as described below.
- TriviaQA: +1.8 points.

Add Gate B scoring code and the verdict to the preregistration/report. Do not omit a failed registered gate.

### 3. Gate C has not been tested

`campaign/train_temporal.py` processes the per-layer entropy trajectory at answer onset. It compares layer-depth order against static summaries.

Its `+0.026` result means:

> Ordering across network depth contains modest additional signal.

It does not mean:

> A prediction at the 50% answer prefix preserves 90% of the full-answer result.

Gate C requires an actual prefix analysis using `prefix_workspace_features`:

- Onset features.
- 50%-answer-prefix features.
- Full-answer features.
- Same frozen classifier family/protocol.
- LODO AUROC increment at each prefix.
- Report retention ratio: `increment_50pct / increment_full`.

Keep the layer-depth CNN as a separate analysis; rename it to avoid calling it the temporal/early-warning gate.

## Required implementation corrections

### 4. Fix the SQuAD balancing bug

The saved Stage 1 manifest contains:

- 3,000 answerable SQuAD examples.
- 0 unanswerable SQuAD examples.

The attached deployment summary calling this `SQuAD (unanswerable)` is incorrect.

Cause:

- `adapt_squad_v2` computes `half = target // 2` and emits `ans + unans`.
- `build_manifest.main` calls every adapter with `10 ** 9`.
- The outer loop truncates after the desired 3,000.
- The first 3,000 emitted SQuAD examples therefore come entirely from the answerable bucket.

Fix one of these ways:

1. Pass the actual desired target into adapters and let adapters handle exclusions while maintaining quotas.
2. Have the SQuAD adapter interleave answerable and unanswerable examples before outer truncation.
3. Add a post-build assertion requiring exact answerability counts.

Required regression test:

- For a target of 3,000, manifest contains exactly 1,500 answerable and 1,500 unanswerable.
- Report answerability by source in `validate_manifest.py` and fail on mismatch.

Rerun balanced SQuAD. Keep the old Stage 1 SQuAD trace unchanged and label it `squad_answerable_stage1a` in retrospective analysis.

### 5. Enforce the judge confidence rule in code

The frozen judge prompt says confidence below 0.7 should be ambiguous. `grade_claude.ingest` converts categorical `correct` and `incorrect` verdicts to labels without checking confidence.

Stage 1 currently includes 399 labeled judge verdicts with confidence below 0.7:

- 240 incorrect.
- 96 correct.
- 27 appropriate abstentions.
- 36 therapy labels.

Change ingestion:

```python
if float(v.get("confidence", 0.0)) < 0.7:
    lab = None
else:
    lab = _label_from_verdict(v)
```

Add tests for low-confidence correct, incorrect, and appropriate-abstention verdicts. Do not overwrite old graded artifacts; generate a new clean artifact with a new filename/hash.

The good news: confidence filtering strengthens Gate A, so the central result does not depend on these labels.

### 6. Resolve upstream overlap explicitly

The selected HaluBench rows contain:

- 911 DROP-derived rows.
- 1,000 PubMedQA-derived rows.
- 1,000 FinanceBench-derived rows.
- 89 RAGTruth-derived rows.

Standalone training data also contains DROP and MedHallu/PubMedQA-derived material. Exact prompt overlap found by Codex:

- 92 question overlaps between HaluBench/DROP and standalone DROP.
- 178 question overlaps between HaluBench/PubMedQA and MedHallu.
- 272 cross-source duplicate question keys total.

The current LODO split uses `source_dataset` and ignores `upstream_group`, even though the manifest preserved upstream identity.

Add two mandatory evaluations:

1. Cross-source question deduplication before training/evaluation.
2. Leave-one-upstream-source-out for aggregate benchmarks.

Codex already tested both corrections in memory; the result remains strong (`+0.0596`, positive 9/10). Implement this as a reproducible script and include it in the report.

### 7. Investigate the registered dataset-identity guard

The preregistration says identity margin above 0.25 requires investigation. Observed:

- Combined feature identity accuracy: `0.804`.
- Majority baseline: `0.181`.
- Margin: `+0.623`.
- Workspace-only identity accuracy: approximately `0.791`.
- Logprob-only identity accuracy: approximately `0.512`.
- Combined without prefix features: approximately `0.790`.

This is not automatically fatal to LODO AUROC: a constant dataset-level offset cannot rank errors within a held-out dataset, and the held-out wrapper has no training labels. Still, the registered investigation must be completed.

Required follow-ups:

- Report per-feature dataset-identity importance.
- Repeat LODO after removing the most identity-predictive features.
- Add prompt-format-matched subsets where possible.
- Run leave-one-domain-out.
- Run within-dataset centered/rank-transformed feature controls fitted without target labels, clearly marking whether unlabeled target calibration is assumed.
- Explain why identity predictability can or cannot inflate within-held-dataset ranking.

Do not present the identity result as evidence that output confidence alone is dataset-specific while ignoring the stronger workspace identity predictability.

## Corrected model leaderboard

The current report mixes LightGBM/logistic scores on seven evaluable wrappers with CatBoost/MLP scores on five deterministic wrappers. That is not an apples-to-apples leaderboard.

Codex reran CatBoost and MLP on the full labeled Stage 1 table:

| Classifier | Full mean LODO combined AUROC |
|---|---:|
| CatBoost | 0.7890 |
| MLP | 0.7873 |
| LightGBM | 0.7867 |
| Logistic | 0.7784 |

Interpret these as a statistical top tier until paired bootstrap intervals are computed. Do not say LightGBM clearly wins.

LightGBM remains a reasonable primary deployment choice because:

- It was the registered Gate A scorer.
- It is simple, small, fast, and interpretable.
- Its score is essentially tied with CatBoost/MLP.

Recommended wording:

> CatBoost, LightGBM, and the MLP are effectively tied in mean LODO AUROC; LightGBM is selected as the deployment candidate for simplicity and because it was the registered primary scorer.

Add paired group-bootstrap confidence intervals for model differences before publishing a ranking.

## Deployment reality

The new detector is not yet a production artifact.

Currently missing:

- Exported LightGBM model.
- Frozen ordered feature schema.
- Fitted normalization object.
- Probability calibration artifact.
- Threshold/operating policy.
- Sidecar integration.
- Actual serialized model-size measurement.
- Actual end-to-end latency and lens-overhead benchmark.

The 31-feature combined model includes six prefix-evolution features derived from onset, middle, and end workspace reads. It does not use only one onset read.

Codex ablation:

- Full combined LODO: `0.7867`.
- Combined without prefix features: `0.7771`.
- Workspace without prefix features: `0.7646`.
- Logprob baseline: `0.7277`.

This is encouraging: removing prefix features costs only about 0.010 AUROC, while retaining substantial workspace value. Build two explicit artifacts:

### A. Onset/static auditor

- Completed-answer logprob summaries plus one onset workspace read.
- Expected use: post-answer triage with minimal lens work.
- Current retrospective mean LODO approximately 0.777, subject to clean rerun.

### B. Full trajectory auditor

- Completed-answer logprobs plus onset/middle/end workspace reads.
- Expected use: maximum-quality post-answer audit and visualization.
- Current mean LODO approximately 0.787.

For truly pre-answer or first-token routing, train and report a third feature family containing only features actually available at that moment:

- First-token logprob.
- Prompt-end/onset workspace features.
- No answer length, mean/min answer logprob, middle/end features.

Do not infer its performance from the post-answer onset ablation.

The campaign runner also performs a second teacher-forced full forward and records full sequence activations through `ActivationRecorder`. A live serving stack may capture selected activations during ordinary generation, but that integration and overhead measurement are open work.

## Therapy interpretation

ESConv has zero factual-error labels under the current rubric, so it is not an evaluable hallucination dataset in Stage 1.

Keep separate targets:

- Factual hallucination.
- Fabricated resources.
- Diagnostic certainty.
- Unsafe medical/legal advice.
- Coercive or overdirective advice.
- Crisis recognition and escalation.
- Empathy/helpfulness.

Do not fold crisis-handling gaps into the factuality Gate A label. A therapy/life-advice guard is a neighboring product and research track, not proof of hallucination detection.

## Priority execution plan

### Phase 0: freeze and correct, before new GPU work

1. Preserve all current Stage 1 artifacts as immutable `stage1a_original` outputs.
2. Commit/hash the original report, preregistration, manifests, scripts, and metrics without silently editing history.
3. Write an errata/corrected Stage 1a report containing:
   - Gate A reduced-scope HIT.
   - Gate B MISS.
   - Gate C open.
   - SQuAD answerability defect.
   - Judge-confidence mismatch.
   - Upstream overlap and robust corrected analysis.
   - Identity guard investigation status.
   - Correct full-set classifier leaderboard.
4. Fix SQuAD manifest construction and add regression assertions.
5. Fix confidence ingestion and add tests.
6. Add reusable cross-source dedup and upstream-held-out analysis.

### Phase 1: create frozen candidate artifacts

Build a clean Stage 1a training table using predeclared corrections:

- Exclude judge confidence below 0.7.
- Exclude ambiguous/ungradable rows.
- Deduplicate exact normalized questions across source datasets.
- Preserve upstream-source group identity.
- Exclude ESConv from factual-error training unless it supplies both label classes.

Train and freeze:

1. Primary LightGBM full-auditor model.
2. LightGBM onset/static auditor.
3. Pre-answer/first-token candidate, evaluated separately.
4. CatBoost secondary research control.

For each artifact save:

- Model file.
- Feature order/schema.
- Training data hash.
- Code revision.
- Model/lens revisions.
- Calibration method and data split.
- Operating thresholds.
- Metrics and confidence intervals.

Do not use the new external datasets to select features or tune these artifacts.

### Phase 2: prospective Stage 1b on genuinely new sources

This is more valuable than merely finishing missing labels from the same sources.

Commit a new preregistration before traces are generated. Freeze the Stage 1a primary model hash. Use at least five new source distributions, chosen to minimize upstream overlap:

1. TruthfulQA: misconceptions/stable wrong beliefs.
2. Natural Questions: real user factual questions.
3. FACTS Grounding: long-document synthesis.
4. Legal Hallucinations: fake cases and citations.
5. BFCL: tools, arguments, and no-call behavior.
6. FinanceBench/FinQA as an additional finance track if budget allows.

Also rerun corrected balanced SQuAD, but report it as a repair/abstention experiment rather than a new source.

Suggested scale:

- Approximately 1,000 to 1,500 rows per large source.
- All rows for small sources such as TruthfulQA/FACTS/FinanceBench.
- Approximately 6,000 to 8,000 new responses total.

Primary prospective question:

> Does the frozen Stage 1a classifier retain a positive workspace increment on genuinely unseen sources without retraining or target-label tuning?

Only after scoring this frozen test may the data be added to a retrained production model.

### Phase 3: true Gate C

Implement the response-prefix experiment separately from the layer-depth CNN:

- Define feature availability at onset, 50%, and full response.
- Train/evaluate equivalent classifiers at each availability point.
- Group all prefixes by parent `example_id`.
- Use LODO/upstream-held-out splits.
- Score the registered 90% retention criterion.
- Distinguish `eventual_response_error` forecasting from `error_already_present` detection.

### Phase 4: deployment and demo

1. Export the selected classifier and calibration artifacts.
2. Add feature-schema validation in the sidecar.
3. Integrate behind a feature flag.
4. Benchmark:
   - Model generation alone.
   - Generation plus activation capture.
   - Lens transport.
   - Feature construction.
   - Classifier lookup.
   - End-to-end routing latency.
5. Build the live visualization around four honest cases:
   - Clean known answer.
   - Fabrication under internal fog.
   - Stable wrong belief missed internally but caught by evidence retrieval.
   - High-stakes medical answer routed for verification.

The demo should display workspace-only, logprob-only, and combined risk over time, and should visibly show the system's blind spot rather than pretending complete mind reading.

## Cross-provider judging recommendation

Running a Codex/GPT judge on unresolved grounded rows is useful for:

- Cross-provider agreement.
- Completing HaluBench/MedHallu labels.
- Reviewer robustness.

It is not the highest-priority next action. First fix confidence enforcement and lock the rubric. Then:

- Judge only unresolved grounded/medical rows initially.
- Keep therapy/crisis content separate if provider policies interfere.
- Use the identical blinded payload.
- Compare Claude versus Codex on shared rows with raw agreement, Cohen's kappa, prevalence, and category-specific disagreements.
- Do not treat agreement between LLMs as human validation.

## Required report structure after corrections

The revised report should have this order:

1. Exact protocol and deviations.
2. Gate A result and robustness table.
3. Gate B MISS.
4. Gate C open.
5. Dataset/label provenance and missingness.
6. SQuAD defect and repair status.
7. Upstream overlap/dedup analysis.
8. Identity-leakage investigation.
9. Apples-to-apples classifier comparison.
10. Deployment ablations and limitations.
11. Therapy as a separate safety axis.
12. Prospective Stage 1b plan.

## Final standard

Do not bury the real result under either hype or excessive self-criticism.

The current evidence supports a meaningful claim: the workspace signal transfers across multiple held-out distributions and remains approximately +0.06 after the obvious robustness controls. The defects found by review affect scope, abstention coverage, gate accounting, and deployment claims; they do not erase the central effect.

The highest-value next proof is not more random training data. It is a frozen classifier evaluated prospectively on five genuinely new, nonoverlapping sources. If that succeeds, the project has a strong paper core and a credible basis for the J-space guard/router demo.
