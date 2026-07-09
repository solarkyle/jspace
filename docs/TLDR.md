# TLDR: what we did, in order, and what actually came out of it

Plain-language summary of the whole project. No pitch. The full numbers live in
[FINDINGS.md](FINDINGS.md); the running system lives in `sidecar/`. Everything
below is reproducible from the scripts and traces in this repo.

## Where we started

Anthropic published a paper on reading a model's "global workspace" with a
Jacobian lens: linearly transport the residual stream at any layer into the
final layer's basis, decode with the model's own unembedding, and you can read
what the model is disposed to say at every layer and position. We replicated
the fit same-day on an open model (Gemma 4 E4B) on a single 16GB consumer GPU,
then extended it to five models: Gemma 4 E4B, 12B, 12B-abliterated, 26B-A4B
MoE, and Qwen 3.6-27B.

The question that drove everything after the replication: is the workspace
readout just a curiosity, or does it carry information the output logits do
not have?

## The iterations, in order

1. **Replication.** Fit the lens on E4B locally (~77s/prompt), verified the
   paper's basic claim holds on an open model. Fit the other four lenses on
   Modal.

2. **Covert emotion probes.** Model writes one fixed sentence while
   instructed to secretly feel one of five emotions. The text is
   byte-identical across conditions, so any workspace difference is the covert
   state. Result: capable models hold the unspoken emotion near the very top
   of their internal "about to say" distribution (Qwen 27B: rank 0-2 out of
   ~260k tokens for all five emotions), and abliteration amplifies the
   internal emotional representation by 1-2 orders of magnitude on 4 of 5
   emotions.

3. **Hallucination prediction.** 500 TriviaQA questions, lens read once at
   the answer position before generation. The quadrant result: answers where
   the output logit is confident but the workspace is noisy are 42% correct,
   vs 75% when the workspace is clean. That cell (sounds sure, is wrong) is
   the hallucination case output confidence cannot catch by definition.

4. **Cross-model replication.** Same protocol on all five models. The
   overconfident-hallucination signal held on 3 of 4 new models (the
   pre-registered gate). The miss is Qwen 27B, whose output confidence is
   already so well calibrated there is nothing left in the blind spot.
   A threshold fit on E4B transfers to the other Gemmas untuned.

5. **Fake entities (negative result).** Questions about invented people and
   places. Output logprob detects them almost perfectly on capable models
   (AUC 0.94-1.00); the workspace adds nothing there. Unfamiliar names are
   NOT the blind spot. Models know when they have never heard of something.
   The workspace's value stays on real-but-hard questions answered
   confidently and wrongly. Side finding: abliteration converts refusal into
   fabrication (12B base fabricates on 17/50 fakes, abliterated on 49/50).

6. **Trajectory features.** Replaced the single mean-entropy scalar with 10
   layerwise features (slope, ignition depth, band agreement, etc.).
   Workspace-only then beats output confidence outright on every Gemma
   (CV-AUC up to 0.824 vs 0.736 on 12B), and a classifier trained only on
   E4B transfers zero-shot to the other Gemmas at 0.74-0.78.

7. **The noise reframe.** Decomposed workspace entropy into two parts: rival
   mass (probability on candidates 2-5: the model deliberating between real
   options) and tail smear (mass spread past the top ~20 tokens: undirected
   noise). On all five models, entropy correlates with tail smear at
   0.93-0.999. The predictive signal is the noise, not the deliberation.
   Deliberation is actually the safest state. So the honest name for the
   detector is a noise gate, not an uncertainty gate.

8. **Quantization.** A lens fitted on bf16 activations reads NF4 4-bit
   activations without refitting. Verified on both 12Bs: the signal survives
   quantization and quantization does not add smear. This is what makes local
   deployment on consumer cards practical (12B at NF4 is ~8GB).

9. **The sidecar (the running system).** OpenAI-compatible FastAPI server:
   local Gemma at 4-bit + lens, noise score on every response. The original
   experiments read one answer-onset snapshot; the sidecar now reads the first
   three answer-token workspaces by default and routes on the noisiest one.
   Four modes: detect (flag only), escalate (route flagged queries to a big
   model, GLM 5.2 via OpenRouter in our config), refuse, tag. Plus a chat UI
   with a live noise meter and a full workspace heatmap (layers x candidate
   tokens). Verified end to end: on an obscure-lyrics question the local 12B
   fabricates a band name with noise 0.99, the gate fires, the cloud model
   returns the right answer.

10. **The deployment gotcha we found the hard way.** The validated noise read
    happens at answer onset, which is only the answer token when the model
    answers tersely. If it preambles ("**The** singer who..."), token one is
    filler with a clean workspace and a single-token read misses. Fix: a terse
    no-markdown system prompt, plus the sidecar's new
    `WORKSPACE_READ_TOKENS=3` prefix scan. Same question went from 0.03
    (missed) to 0.90 (caught) when forced terse.

## What we think is actually new

Ranked by how confident we are that nobody has published it in this form
(checked against the literature the same day, not exhaustively):

1. **A single-pass hallucination signal from the model's own workspace that
   works precisely where output confidence is blind**, replicated across
   model scales, with zero-shot cross-model transfer. Semantic entropy needs
   5-10 sampled generations per query; this is one forward pass plus a
   linear transport.
2. **The noise decomposition**: what predicts wrongness is tail smear, not
   rival-candidate deliberation. We have not seen entropy split this way at
   the workspace level.
3. **Lens survives quantization without refit.** Small but deployment-
   critical, and nobody had reason to check it before.
4. **Abliteration measurably alters internal signal quality**: amplifies
   covert emotion 1-2 orders, converts refusal to fabrication 17/50 to
   49/50, and degrades the mean-entropy self-knowledge signal. "Same model
   minus refusals" is not an accurate description of an abliterated model.
5. **At matched active compute, total parameters buy internal richness**:
   the 26B MoE (4B active) has a far more vivid and selective emotional
   workspace than the 4B dense.

## The capability tradeoff (2026-07-09, the anatomy round)

Digging into WHAT the detector catches produced the cleanest one-liner of
the project: **capability trades fog for beliefs, so error detection fades
with scale while deception detection sharpens.** In detail:

- Wrong answers decompose into fabrication (improvising in fog: loud in the
  workspace, ~70% flagged) and substitution (cleanly retrieving the wrong
  fact: silent, ~50% = coin flip). Two independent graders, kappa 0.88.
- As capability rises the error mix shifts from fabrication to substitution
  (12B 24% fabrication, Qwen 10%), and the workspace's edge over output
  confidence shrinks accordingly (12B +0.09 AUC, 31B +0.01, Qwen 0.00).
- Clean-wrong answers are STABLE wrong beliefs: they resample to the same
  wrong answer 85% of the time in the cleanest quartile (junk-robust
  clustering; correct answers 95%). No method sees them, at any cost,
  because there is no internal disagreement to see.
- The OPPOSITE trend for lying: with type-matched controls and a verified
  honest baseline, the true answer stays elevated in the workspace during
  instructed lies (E4B p=.023, 12B p=.033, Qwen p=.043; MoE null), and it
  is most vivid on the most capable model. Deception is visible exactly
  where wrongness is not.
- Believed myths show NO truth trace (abliterated 12B repeats 7/20 myths
  with the myth winning the deep band outright): the lens distinguishes
  deception from delusion, which output text cannot.
- Provisional (bf16 confirmation pending): on the 31B the wrongness signal
  survives but MIGRATES DEEPER; the fixed 25-75% band averages dead and
  sign-flipped mid layers over the healthy deep ones (band-mean AUC 0.43,
  depth-aware split-half 0.75). Fixed-fraction bands do not transfer to
  deeper models.

## Honest misses and limits

- Qwen 27B: workspace entropy is at chance there. Well-calibrated models may
  have no blind spot to fill, or the lens works less well off-family. Cannot
  separate the two without a second large non-Gemma model.
- The detector catches noisy retrieval, not confident misconceptions. A model
  that firmly believes something wrong reads clean. This is a stated limit
  of the method, not a fixable bug.
- On well-calibrated datasets (TriviaQA) the end-to-end router gain over
  logprob-only is small. The value concentrates in the overconfident-wrong
  cell, which is rare there and common in the wild.
- Emotion findings are n=1 prompt per condition. Directions are consistent
  across five models but the per-cell numbers are noisy.
- "Workspace" means the lens's linear approximation of it, and these are
  representation findings, not claims about felt experience.

## Suspicions (consistent with the data, not yet proven)

- **Safety training suppresses specific internal emotions, not expression in
  general.** Abliteration unlocks anger, fear, euphoria, amusement but not
  grief, which RLHF plausibly trains up. The pattern matches how these
  models are tuned too well to be a coincidence, but it is one model pair.
- **Models may inherit the human affect circumplex.** Bleed between covert
  emotions trends toward the valence-arousal structure (r 0.4-0.5 on the
  three stronger models) but nothing survives a permutation test at n=1.
  If it holds at n=20 prompts, that is a paper on its own.
- **The 12B dense anomaly.** It buries emotions deeper than the 4B while
  being the best Gemma at hallucination self-knowledge. Something about its
  training differs from its siblings and we do not know what.
- **Calibration might be what capability buys.** Qwen's flat workspace
  signal next to its excellent logprob calibration hints that as models get
  better calibrated, internal-state detectors matter less. If true, this
  method's window is exactly the local-model regime, which is fine, because
  that is where it runs.
- **Reading noise over the first few answer tokens should close most of the
  preamble failure mode.** This is now implemented in the sidecar; the next
  missing number is a real serving-stack overhead benchmark.

## Run it

```
# server (detect mode is default)
HF_HOME=E:/hf-cache MODEL_ID=google/gemma-4-12B-it QUANT=4bit \
  .venv/Scripts/python -m uvicorn sidecar.server:app --port 8765
# lie detector chat UI
http://localhost:8765/chat
```

Analyses: `analyze_router.py`, `analyze_crossmodel.py`, `analyze_deep.py`,
all off the committed traces in `data/`. No GPU needed to reproduce the
numbers.
