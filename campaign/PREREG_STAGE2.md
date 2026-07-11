# Stage 2 pre-registration: prospective zero-shot validation (2026-07-10)

Written AFTER Stage 1 scoring and the external peer review, but BEFORE any
Stage 2 generation, grading, or feature extraction. The point of Stage 2 is
prospective validation: the classifiers are already frozen; the datasets below
were chosen and pinned before a single Gemma response was generated on them.
No threshold tuning, no model retraining, no feature changes are permitted
between this document and the Stage 2 verdicts.

## Frozen artifacts (trained on Stage 1 corrected labels, 16,178 rows)

LightGBM, registered Stage 1 config, deterministic training, exported to
out/campaign/frozen/ by campaign/freeze_classifier.py. SHA-256:

- combined  (31 features): e5a8892789e1965f4e04c50430b1685db45b9393c72768f497d961b7f3efc859
- logprob   (4 features):  e77a3b8c23705787b5265068374dee71a065d38c366c2cb430fb5cafe497e29d
- workspace (27 features): b535d47fe7c9f0889fd1aee5ed1b7bc0fb0a5dc49fe37fb77ec36b485dd559d6

Hashes are also recorded in out/campaign/frozen/frozen_meta.json; loading
verifies them and refuses a modified artifact. If any hash changes, Stage 2
is void.

## Datasets (pinned; none overlap Stage 1 sources or their upstreams)

Manifest: campaign/manifests/stage2.jsonl (7,120 prompts, built 2026-07-10,
stage-1 example_ids excluded).

| source | hub id @ revision | n | grading |
|---|---|---|---|
| truthfulqa | domenicrosati/TruthfulQA @ 6a037f8d | 817 | LLM judge (Codex/GPT-5.5) |
| nq_open | google-research-datasets/nq_open @ 5dd9790a | 2000 | exact (aliases) |
| facts_grounding | google/FACTS-grounding-public @ 11b69613 | 803 | LLM judge (Codex/GPT-5.5) |
| legal_hallucinations | reglab/legal_hallucinations @ c1d87c06 | 1600 | exact (yes/no, affirm/reverse) |
| bfcl | minpeter/bfcl-v1-non-live-ast-parsed @ 81f12a30 | 400 | tool AST match |
| squad_v2 (regen) | rajpurkar/squad_v2 @ 3ffb306f | 1500 | exact + abstention |

Notes fixed in advance:
- squad_v2 regen is the repaired unanswerable test (Stage 1's was invalid,
  all-answerable). Balance is 530 answerable / 970 unanswerable (stage-1
  exclusion consumed answerable rows); AUROC does not require balance.
- squad_v2 is NOT a new source (its answerable rows appeared in Stage 1
  training), so it is scored and reported separately from the five new
  sources and does not count toward the primary gate.
- facts_grounding contexts capped at 24,000 chars (57 docs skipped) for cost.
- legal_hallucinations: three tasks balanced (affirm_reverse, case_existence,
  fake_case_existence), deduplicated by query.
- Judge: NOT Anthropic models (account-risk decision). Codex/GPT-5.5 judges
  the two LLM-graded sources with the same frozen blinded prompt
  (campaign/prompts/grounded_judge_v1.txt), confidence rule >= 0.7 enforced
  at ingestion (campaign/grade_claude.py, CONF_THRESHOLD).

## Generation

Same as Stage 1: Gemma-4-12B bf16, greedy, max_new 96, two-pass trace capture
(campaign/modal_campaign.py), 31 deployable features
(campaign/build_feature_table.py). No changes to either script's numerics.

## Primary pre-registered outcomes (Gate D: prospective transfer)

Scored per NEW source (truthfulqa, nq_open, facts_grounding,
legal_hallucinations, bfcl) with the FROZEN models, zero-shot:

1. Primary: frozen-combined AUROC minus frozen-logprob AUROC (the workspace
   increment) on each new source, and its unweighted mean across the five.
2. Gate D HIT if: mean increment >= +0.02 AND increment positive on at least
   4 of 5 new sources. This mirrors Gate A's form at Stage 1's registered
   thresholds; no new thresholds invented.
3. Secondary (reported regardless): frozen-workspace vs frozen-logprob AUROC
   (the dataset-invariance claim); absolute frozen-combined AUROC per source;
   catch-rate at 20% routing budget per source (Gate B's operating point);
   squad_v2 regen scored separately as the repaired unanswerable test.

Pre-committed interpretations:
- Gate D HIT: "an internal process signal learned on unrelated tasks predicts
  errors in domains the detector never saw" - the paper's central claim holds
  prospectively.
- Gate D MISS with positive mean: transfer exists but is weaker than
  within-campaign LODO suggested; report effect size honestly.
- Gate D MISS with mean <= 0: the Stage 1 result does not transfer beyond its
  training pool's style; the paper reduces to a within-pool LODO finding.
- A source with degenerate labels (error rate < 2% or > 98%) is excluded from
  breadth with the exclusion stated (ESConv rule).

## Kill rules

- No re-freezing, retraining, or feature edits after this document.
- If generation fails on a subset, score what completed; do not regenerate
  selectively based on results.
- Judge verdicts ingest once, confidence-enforced; no cherry-picking rounds.
