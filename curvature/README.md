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
