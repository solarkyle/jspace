# Stage 1 preregistration: cross-dataset workspace value

Written 2026-07-10, BEFORE Stage 0 pilot results were read and BEFORE any
Stage 1 trace exists. Same discipline as docs/POPQA_PREREG.md. The lesson from
the retracted calibration-gap law applies: these predictions are for ABSOLUTE
held-out quantities, scored HIT/MISS/PARTIAL whatever the outcome.

## The central test (handoff section 23)

Train the detector on many source distributions, freeze it, and predict Gemma
4 12B's errors on an entirely unseen dataset, WITHOUT using dataset identity,
domain labels, grader type, or formatting metadata as features, and WITHOUT
touching the held-out dataset's labels during model selection. Everything below
is measured under leave-one-dataset-out (LODO): train on all-but-one source,
test on the held-out one.

## Model + protocol

- Model: google/gemma-4-12B-it, bf16, greedy (generation_config greedy_v1).
- Lens: the committed 12B Jacobian lens (bf16 fit), read via the verified
  two-pass teacher-forced capture (feature-equivalence confirmed 2026-07-10:
  ignition/agreement deltas 0.000, rank delta 0.030).
- Feature families: logprob (4 output-confidence features) vs combined
  (logprob + workspace). Deployable features only; no identity columns.
- Classifiers: logistic and LightGBM (frozen configs registered in
  train_baselines.py) plus TabFM at its frozen config. Grouped folds by
  split_group; prefixes inherit parent example_id.
- Datasets: >= 12 sources spanning >= 5 domains (general, medical, therapy,
  legal/finance, reasoning). Exact counts set by Stage 0 throughput vs budget.

## Gate A (cross-dataset value) -- the gate that matters

Prediction: combined beats logprob-only on mean LODO AUROC by at least +0.02,
AND the increment is positive on at least 70 percent of held-out datasets.
- HIT if both clauses hold on LightGBM (the intended production model).
- PARTIAL if mean +0.02 holds but the 70 percent breadth clause fails.
- MISS if mean increment < +0.02. A MISS is reported, not re-metriced. It would
  mean the within-dataset PopQA signal did not survive distribution shift.

## Gate B (operational routing value)

Prediction: at a fixed 20 percent routing budget (escalate the riskiest 20
percent of queries), the combined detector catches at least 5 percentage points
more wrong answers than logprob-only, averaged over held-out datasets, with the
bootstrap 95 percent CI excluding zero.
- HIT if both hold; MISS otherwise. Report per-dataset catch rates regardless.

## Gate C (early warning), only if prefix traces prove reliable

Prediction: the risk score from the 50 percent answer-prefix preserves at least
90 percent of the full-answer LODO AUROC increment. Falsified if the 50 percent
prefix retains < 90 percent, which would mean the signal only exists once the
answer is essentially complete.

## Registered expectation on TabFM vs LightGBM (not a gate)

TabFM's edge over LightGBM shrinks as training rows grow. Prediction: at Stage 1
scale (>= 8k labeled rows), LightGBM's mean LODO AUROC is within 0.01 of TabFM,
or higher. A clear TabFM win (>0.01) at this scale is a reportable surprise and
gets its own analysis, not a silent adoption.

## Falsification honesty

Any gate that fails is written up as a bounded negative result (handoff 23):
the instrument helps only where it helps. Specifically pre-committed failure
interpretations:
- Gate A MISS -> workspace signal is dataset-specific, not a general monitor.
- Gate B MISS with Gate A HIT -> real but not operationally useful at this budget.
- Identity-leakage margin > 0.25 over majority baseline -> investigate which
  feature leaks dataset style before trusting any pooled number.

## Scoring

Appended as a VERDICTS section here after Stage 1, per gate, with the numbers.
Analysis scripts frozen before the run: campaign/train_baselines.py (logistic +
LightGBM), campaign/train_tabfm.py (TabFM), campaign/split_groups.py (folds +
leakage). No new tuning between prereg and scoring.

## VERDICTS (appended 2026-07-10, after Stage 1 + external review + label correction)

Scored on confidence-corrected labels (judge confidence rule enforced at
ingestion, 399 previously leaked low-confidence labels excluded). Pre-fix
Stage 1a numbers frozen in campaign/reports/stage1a_frozen/.

- Gate A: HIT (reduced scope: 8 sources ran vs >= 12 registered; 7 evaluable).
  LightGBM mean LODO increment +0.0651, breadth 6/7 (squad_v2 -0.003 is the
  known-invalid all-answerable row). Logistic +0.0908, 6/7. Pre-fix: +0.0590,
  7/7. External review reproduced the pre-fix number and it held at ~+0.06
  under dedup, confidence, and upstream-split controls.
- Gate B: MARGINAL HIT. +5.10pp mean catch-rate gain at 20% budget (needed
  >= +5pp), cluster-bootstrap 95% CI [+3.15, +7.36]. Fragile: medhallu alone
  contributes +20pp; excluding it the mean is +2.62pp. On pre-fix labels the
  same analysis gives +4.19pp = MISS. Treat as borderline.
- Gate C: NOT TESTED. The temporal CNN measured layer-order information, not
  50%-prefix forecasting. Open.
- Identity leakage: 0.811 vs 0.185 majority (margin 0.625 > 0.25 threshold).
  Per the registered interpretation, pooled numbers are not trusted; LODO is
  the primary analysis everywhere.
- TabFM: no clean head-to-head (timed out at 14k rows); registered tree-wins
  expectation neither confirmed nor refuted at scale.
- Model tier: CatBoost 0.800, LightGBM 0.796, MLP 0.794, logistic 0.789
  (combined LODO) - effectively tied; LightGBM kept for simplicity, not merit.
