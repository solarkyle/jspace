# Gemma 12B cross-domain campaign

Working scaffold for the campaign specified in `docs/GEMMA12B_CAMPAIGN_HANDOFF.md`.
That document is the source of truth for design; this file records the accepted
deviations, decided before Stage 0 ran.

## Accepted deviations from the handoff (decision record, 2026-07-10)

1. **Breadth over depth.** All domains stay (factual, unanswerable, RAG,
   reasoning, medical, legal, finance, tools, therapy/life-decisions).
   Leave-one-dataset-out power scales with the number of held-out datasets,
   not rows per dataset, so per-dataset targets shrink before any domain is
   dropped. Final counts are set by measured Stage 0 throughput against the
   fixed compute budget.
2. **GPU selection.** Default worker is L40S (12B bf16 fits comfortably);
   long-context shards (FACTS, long HaluBench/RAGTruth rows) route to
   A100-80GB. Stage 0 measures tokens/sec on both before sizing Stage 1.
3. **Natural Questions** enters only via a lightweight projection (question +
   short answer + reduced context), never the full raw dump.
4. **Temporal model input.** The temporal CNN is trained on real sequences:
   per-token feature traces for answers up to 128 tokens, and layerwise
   trajectories. The five fractional prefix checkpoints are treated as static
   columns (LightGBM handles those); a dilated CNN over a length-5 sequence
   tests nothing.
5. **CatBoost and the small MLP** run as cheap controls, not candidates. With
   dataset/domain categoricals banned from deployable features, CatBoost has
   no edge to express; expectation is LightGBM-equivalent within noise.
6. **Registered expectation on TabFM** (written before Stage 1): its edge over
   LightGBM shrinks as training rows grow; in-context tabular models win at
   small n. If TabFM still wins leave-one-dataset-out at Stage 1 scale, that
   is a reportable surprise.
7. **Distillation** exists only if TabFM (or the temporal model) beats direct
   LightGBM on held-out datasets. Otherwise discarded, per the handoff's own
   kill rule.
8. **Classifier bakeoff runs locally** on the committed feature tables; the
   cloud budget is spent exclusively on generation and trace capture.

## Stage gate

No Stage 1 launch until the Stage 0 checklist in the handoff (section 19)
passes, including the teacher-forced vs autoregressive activation equivalence
check at the production batch settings (padding and position ids included,
not batch=1).

## Layout

Follows handoff section 17. Files appear as they are implemented; nothing in
this directory rewrites or replaces the existing experiment scripts.
