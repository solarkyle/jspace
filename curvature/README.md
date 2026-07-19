# Contextual curvature × JSpace — do trajectory-geometry signals add to workspace uncertainty?

A cross-over between JSpace and the ICML-2026 reproduction of *Representational
Curvature Modulates Behavioral Uncertainty in LLMs* (arXiv:2604.23985).

**Contextual curvature** measures how sharply a token's residual-stream
trajectory bends over recent context (the angle between consecutive
token-to-token difference vectors, averaged over a 3-token window). The paper
links it to next-token entropy. JSpace measures a *vertical* signal (workspace
ignition up the layers at one token); curvature is a *horizontal* signal (the
path across tokens at one layer). This asks whether they carry overlapping or
complementary uncertainty.

## Experiment
On Gemma-4-E4B, for the 500 TriviaQA questions in
`../data/uncertainty_trivia_gemma-4-e4b-it.jsonl` (with existing right/wrong
labels + logprob + workspace features), we read contextual curvature at the
answer-onset position across the middle-layer band (L10–30) and test whether it
discriminates incorrect answers, alone and on top of the existing features.

## Finding
- Curvature shows **above-chance discrimination** of TriviaQA correctness on
  this sample (AUROC ≈ 0.606) — the paper's curvature↔uncertainty link is
  present in a modern architecture, not just GPT-2/Pythia.
- But it gives **negligible incremental AUROC** over the existing signals:
  logprob 0.716 → +curvature 0.719; logprob+JSpace 0.780 → +curvature 0.784.
- Read: curvature and the output/workspace features tap overlapping uncertainty;
  curvature tracks lexical/entropy uncertainty more than factual correctness per se.

See `gemma_hallucination_auroc.csv` and `fig_extension_auroc.png`.

## Reproduce
```
python gemma_curvature.py     # HF Jobs: extract curvature on the 500 questions
python analyze_extension.py   # nested-CV AUROC comparison
```
Numbers are a single 500-question sample; treat AUROC differences as indicative,
not intervals. Part of the full reproduction logbook:
https://huggingface.co/spaces/solarkyle/fufl3hBXMq

## Update: PopQA replication + per-layer analysis

Ran a second labeled set (PopQA, 500 Q) storing **per-layer** curvature, and found
the band-average was hiding real structure:

- **Layer-localized with a sign flip (~L22→L23).** Per-layer AUROC for flagging a
  wrong answer: L10–22 sit below 0.5, **L23–29 jump to 0.62–0.74** (L24 = 0.743,
  comparable to logprob's 0.68). The naive L10–30 **mean averages the opposing
  halves to 0.55** — standard mid-band averaging *destroys* the signal.
  Upper-band curvature (L23–26) alone = **0.731 AUROC**. See `fig_perlayer_auroc.png`.
- **Still redundant, though.** Nested CV: logprob+entropy 0.822 → +upper-band
  curvature 0.823 (+0.001). Even well-localized, curvature adds ~nothing over
  output-logprob + workspace entropy.
- The earlier confident-slice hint (+0.010 on trivia alone) did **not** replicate
  when pooled with PopQA (−0.026) — it was noise.

**Takeaway:** contextual curvature is a genuine, substantial uncertainty readout
*when read at the right layers* (a caution for band-averaging in this literature),
but it indexes the same latent uncertainty as logprob/JSpace rather than adding a
complementary factual-error signal. `analyze_extension_v2.py` reproduces this.
