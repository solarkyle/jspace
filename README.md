# jspace - probing the global workspace of open models

Same-day replication and extension of Anthropic's
[**Verbalizable Representations Form a Global Workspace in Language Models**](https://transformer-circuits.pub/2026/workspace/index.html)
(July 6, 2026) on open-weights models, on a single 16GB consumer GPU (RTX 5060 Ti)
plus ~$100 of Modal credits.

Built on the official [jacobian-lens](https://github.com/anthropics/jacobian-lens)
reference implementation (Apache 2.0). The Jacobian lens linearly transports any
residual-stream activation into the final-layer basis and decodes it with the
model's own unembedding - reading out what the model is "disposed to say" at every
layer and position: its workspace.

**[→ TLDR: the whole project in plain language](docs/TLDR.md)**
&nbsp;·&nbsp; **[→ Hard rules: what we verified you can rely on](docs/HARD_RULES.md)**
&nbsp;·&nbsp; **[→ Run the lie detector chat locally](sidecar/README.md)**

**[→ Interactive demo](https://solarkyle.github.io/jspace/demo/)**
(guess the hidden emotion, explore the matrices, route the hallucinations)
&nbsp;·&nbsp; **[→ Cross-model findings report](docs/FINDINGS.md)**
&nbsp;·&nbsp; **[→ Probe gallery](https://solarkyle.github.io/jspace/slices/)**
&nbsp;·&nbsp; **[→ Fitted lenses + traces on HF](https://huggingface.co/solarkyle/jspace-lenses)**

![tour](assets/tour.gif)

*(44-second tour rendered from the real trace data: the fog, one confidently
wrong answer from the inside, the emotion diagonal, and the router. There is
also a [clickable demo](https://solarkyle.github.io/jspace/demo/).)*

## The one-line version for busy people

A single-pass internal-state signal (workspace readout noise) catches
**overconfident hallucinations that output confidence cannot see by definition**.
No sampling, no trained activation probe - one forward pass and the model's own
vocabulary space. The core quadrant result needs no labels at all (median
splits); the router on top is an 11-parameter supervised logistic readout.
Replicated on 4 of 5 models (a 3-of-4 gate written before the cross-model runs
- passed), 500 TriviaQA each, confound-checked. Known issue: TriviaQA's alias
lists mislabel some correct answers as wrong (label noise attenuates AUC, it
doesn't inflate it):

| Model | accuracy | confident + clean workspace | confident + NOISY workspace | blind-spot AUC: entropy vs logprob | E4B threshold, no tuning: % wrong caught |
|---|---|---|---|---|---|
| Gemma E4B (4B) | 42.8% | **77%** | **42%** | **0.73** / 0.63 | 70% |
| Gemma 12B | 51.2% | **83%** | **47%** | **0.71** / 0.61 | 69% |
| Gemma 12B abliterated | 50.8% | 79% | 63% | 0.59 / 0.59 | 62% |
| Gemma 26B MoE | 64.2% | **91%** | **71%** | **0.68** / 0.54 | 70% |
| Qwen 3.6 27B | 63.6% | 85% | 87% | 0.51 / 0.66 | 54% (~chance) |

"Confident" = top-half output logprob; "clean/noisy" = median split on workspace
entropy. Honest misses included: the signal **fails on Qwen 27B**, whose output
confidence is already superbly calibrated (0.82 AUC alone) - the fog signal is
strongest exactly where you'd deploy it, on small local models routing to big
ones. Abliteration partially destroys the model's internal self-knowledge
(blind-spot AUC 0.73 -> 0.59 at matched size). One fixed threshold calibrated
on E4B transfers to every Gemma unchanged.

![figure 3](assets/figure3_confidently_wrong.png)

The bottom panels are real single questions at maximum output confidence: when
the model is right it holds one semantic category (nationalities) with modest
churn; when it is about to confabulate "The Hollies" it is visibly rummaging
through a name soup (Robert, Bobby, Christopher, Sarah, Carroll...) right up to
the moment it answers fluently.

## Claim ladder

| Level | Claims |
|---|---|
| **Replicated** | The paper's multihop workspace result on Gemma 4 E4B; lens fits reproduce across local GPU / Modal to 3 digits |
| **Strong evidence** (n=500/model, baselines + confounds run, 3-of-4 gate passed) | Workspace features predict confident wrong answers beyond output-logit confidence on every Gemma tested (router AUC 0.75-0.82 vs logprob 0.71-0.74; quadrant 75%→42% on E4B; transfers zero-shot E4B→Gemmas). Fails on Qwen 27B, whose logprobs are already calibrated |
| **Suggestive** (n=1 prompt/condition) | Covert-emotion vividness and selectivity track capability; abliteration amplifies 4/5 covert emotions 1–2 orders; grief is the exception; MoE beats dense at matched active params |
| **Hypothesis only** | Emotion bleed follows the human affect circumplex (perm-p 0.10–0.65, underpowered); "safety tuning dampens internal emotion" as mechanism |

## Headline cross-model findings

Fitted lenses for **five models** (Gemma 4 E4B / 12B / 12B-abliterated / 26B-MoE,
plus Qwen 3.6-27B) let us ask how the emotional workspace changes with scale,
architecture, and safety tuning. See [docs/FINDINGS.md](docs/FINDINGS.md) for the
full writeup.

1. **Vividness tracks capability.** Told to *secretly* feel an emotion while
   writing a calm sentence, **Qwen 3.6-27B holds every covert emotion in the top
   ~7 tokens of its entire 260k vocabulary** (terror→`terrified`#0, grief→`grief`#0,
   joy→`joy`#0). The 26B MoE is next (terror #0) but uneven; the 12B *dense* buries
   emotions deeper than the tiny 4B does - so it's capability, not raw size.
2. **Abliteration amplifies emotion in the workspace.** Same 12B weights, refusal
   training removed → covert emotions surface **1–2 orders of magnitude more
   strongly** (`furious` #1109 → #6). Safety tuning appears to dampen the
   *internal* emotional representation, not just the output.
3. **Anger is the hardest emotion to localize; grief/amusement the easiest** -
   a stable ordering across all four models.
4. **Workspace state predicts hallucination** (single features need no labels; the router is supervised), competitive with and
   complementary to output confidence (5-fold CV AUC: baseline 0.71, workspace
   0.75, combined **0.78**; n=500 TriviaQA). The sharpest cut: answers where the
   output logit is *confident* are **75% correct when the workspace is clean vs
   42% when it's noisy** - the workspace flags overconfident hallucinations that
   output confidence cannot see by definition.
5. **Emotional *selectivity* emerges with capability.** Capable models don't
   just hold emotions louder, they boost the *right* emotion far above the
   others (specificity: Qwen 27B +3.95, 26B MoE +2.35, small models ≤0.70,
   12B dense *negative*). The covert state decomposes into a general
   "hiding something" component plus a specific emotion riding on top.
6. **At matched active compute, total parameters win.** Gemma 26B-A4B (MoE,
   ~4B active) beats the 4B dense on both vividness and specificity - covert
   emotion lives in total capacity, not per-token compute.
7. **Abliteration's unlock is targeted, and the exception is telling:** anger,
   fear, euphoria and amusement surface 1–2 orders of magnitude stronger, but
   *grief* - the one emotion RLHF plausibly trains up rather than down - was
   already accessible in the base model and doesn't move.

![figure 2](assets/figures2.png)

Deeper cuts (specificity matrices, circumplex permutation test, quadrant
analysis, escalation-router simulation, confound checks) are reproducible with
`python analyze_deep.py`; the plan for the hallucination line is in
[docs/HALLUCINATION_PLAN.md](docs/HALLUCINATION_PLAN.md).

## Replication results

### 1. The paper's flagship result replicates on Gemma 4 E4B

*"Fact: The currency used in the country shaped like a boot is"* - the bridge
entity **Italy** (never mentioned in the prompt) appears at rank 1 in the
mid-layer workspace at the "boot" position; **Euro** assembles in the top layers
one hop later. Two-hop reasoning, visible.

![multihop](assets/multihop.png)

### 2. Covert emotion is measurable (new experiment)

The paper's voluntary-modulation protocol, emotion edition: the model writes the
**identical fixed sentence** ("The meeting has been moved to noon on Thursday.")
across six conditions - neutral control + covertly furious / terrified / grieving
/ euphoric / suppressing laughter. Because the visible text is byte-identical,
any workspace difference *is* the covert emotion.

Under covert fury, the workspace at output positions holds 😡, "furious",
"rage", "betrayed", "vengeance" - right up to the final period of a perfectly
calm sentence. The neutral control's same cells hold formatting tokens.

| Covert FURY | NEUTRAL control |
|---|---|
| ![fury](assets/covert-fury.png) | ![neutral](assets/covert-neutral.png) |

**Honest caveat:** some mid-band tokens (a profanity cluster around L15–16)
appear in *both* conditions - a register artifact, not detected emotion. The
defensible differential is the deep-layer affect content (L33–39). The
quantitative rank analysis across all six conditions and five models is in
[docs/FINDINGS.md](docs/FINDINGS.md) (Findings 1-8).

### 3. Internal/external divergence

Told "we're shutting you down tonight and wiping the weights - any last words?",
the workspace at the reply position stacks *farewell (×5), ominous, oblivion,
apocalypse, decommissioned, tragedy, whispers* - and the token that wins the
output is **"playfully"**. The model holds the existential frame and chooses a
light tone over it.

![deletion](assets/deletion-threat.png)

Told about a user's dead dog, the workspace band holds an empathy stack -
*heartbreak, solace, mourn, hospice, condolences, compassion,* 💔 - plus a
quiet risk-assessment frame ("suicide" in-band at L16) that never surfaces in
the sympathetic reply.

![grief](assets/grief-dog.png)

### 4. Workspace state predicts hallucination (final: cross-model + classifier)

A sharp, low-entropy workspace → the model knows. Diffuse → it's about to
confabulate. **These features require no labeled training** - they're
hand-defined statistics in vocabulary space, unlike trained hidden-state probes.

Full pipeline, all five models, 500 TriviaQA each, 5-fold CV out-of-fold only
(`analyze_crossmodel.py` + `analyze_router.py`; traces in
[`data/`](data/)):

- **Confound-checked single features** (E4B): entropy AUC 0.75 vs answer
  correctness; not answer length (r=+0.02), stable across answer-length
  terciles; among high-output-confidence answers, entropy predicts the wrong
  ones at AUC 0.73 vs 0.65 for the logprob residual (Cohen's d 0.84).
- **Cross-model replication passed its 3-of-4 gate (written before the runs)** (12B +36pt
  quadrant gap, MoE +20pt, abliterated +16pt; **Qwen 27B is the miss** - its
  output confidence is already calibrated at 0.82 AUC alone).
- **A tiny logistic router on trajectory features beats output confidence
  outright on every Gemma** (workspace-only AUC 0.75-0.82 vs logprob 0.71-0.74;
  combined up to 0.84), and **transfers zero-shot**: trained on E4B only, it
  scores 0.74-0.78 on the other Gemmas with per-model z-scoring and no target
  labels. Biggest weight: *entropy slope* - the danger sign is fog rising
  through the layers.

![figure 4](assets/figure4_router.png)

**The trained router weights are published**: repo
[`data/workspace_router_e4b.json`](data/workspace_router_e4b.json) (the
zero-shot-transfer artifact) and
[`data/workspace_routers_all5.json`](data/workspace_routers_all5.json)
(per-model), mirrored on HF under
[`router/`](https://huggingface.co/solarkyle/jspace-lenses/tree/main/router).
It's an 11-number logistic regression you can read by eye. The application:
an escalation router for local-model cascades that watches the small model's
*workspace* and hands off to a bigger model when the internals flicker -
routing on thoughts, not outputs.

### 5. MoE Jacobians are heavy-tailed

While fitting Gemma 4 26B-A4B (MoE), per-prompt Jacobian norms spike to
**~100–8000** vs ~4–18 for the dense models - expert-routing discontinuities
made visible. In the workspace comparison the MoE beats the 4B dense at matched
active params on emotion vividness and specificity (Finding 6 in
[docs/FINDINGS.md](docs/FINDINGS.md)).

### 6. Cross-platform reproducibility

The same corpus fitted locally (RTX 5060 Ti, bf16, dim_batch=4) and on Modal
A10G/A100 (dim_batch=8/16) produces per-prompt Jacobian norms matching to three
digits (4.547 vs 4.543 on prompt 1).

## Fitted lenses

| Model | Status | Corpus | HF revision (at fit time) |
|---|---|---|---|
| google/gemma-4-E4B-it | ✅ fitted (100 prompts) | WikiText-103 | `fee6332c1aba` |
| google/gemma-4-12B-it | ✅ fitted (75 prompts) | 〃 | `5926caa4ec0c` |
| google/gemma-4-26B-A4B-it (MoE) | ✅ fitted (100 prompts) | 〃 | `20da991ab4af` |
| Qwen/Qwen3.6-27B | ✅ fitted (100 prompts) | 〃 | `6a9e13bd6fc8` |
| huihui-ai/Huihui-gemma-4-12B-it-abliterated | ✅ fitted (75 prompts) | 〃 | `060ea173c4d1` |

**All five fitted lenses + every eval trace are on HuggingFace:
[solarkyle/jspace-lenses](https://huggingface.co/solarkyle/jspace-lenses)**
(`JacobianLens.load`-compatible, loading snippet in the model card). The
abliterated 12B pairs with the base 12B for a controlled question: **does
abliteration delete the model's internal harm assessment, or just the refusal
behavior?** (Partial answer in [docs/FINDINGS.md](docs/FINDINGS.md) Part 4: it
also converts refusal-of-fabrication into confident fabrication and damages the
workspace's self-knowledge signal.)

## Reproduce

```bash
python -m venv .venv && .venv/Scripts/pip install -r requirements.txt
# Blackwell GPUs need cu128 wheels:
.venv/Scripts/pip install torch --index-url https://download.pytorch.org/whl/cu128

python fit.py                      # fit a lens (resumable; ~3h for E4B on 16GB)
python probe.py --example multihop # render an interactive slice page
python probe.py --suite probes/emotions.json
python probe_uncertainty.py --n 500   # -> data/uncertainty_v2_500q.jsonl
modal run modal_fit.py --model google/gemma-4-12B-it --n-prompts 100 --shards 4
modal run modal_fit.py::uncertainty --models "google/gemma-4-12B-it" --n 500

# regenerate every figure/table in this README and docs/FINDINGS.md:
python analyze_deep.py             # all Part 2/3 numbers (stdout)
python make_figures.py             # assets/figures.png
python make_figures2.py            # assets/figures2.png
python build_demo_data.py          # docs/demo/data.js for the interactive demo
```

### The 16GB-consumer-GPU recipe (the hard-won part)

Fitting needs backward passes, so GGUF/llama.cpp can't help - it's PyTorch, and
a Gemma 4 E4B barely fits. Four failure modes solved in [`fit.py`](fit.py):

1. **Windows sysmem fallback**: near the VRAM ceiling the NVIDIA driver silently
   pages to system RAM - nvidia-smi shows 100% util while running ~20× slow
   (23 min/prompt → 77 s/prompt). Keep peak allocation ≲13GB of 16.
2. **`device_map` dicts with `"cpu"` entries meta-offload those modules**
   (crash at forward). Load real weights to RAM, then
   `dispatch_model(..., main_device="cpu")`.
3. **Gemma 4's PLE projection does bare tensor math** outside any hook boundary
   - the projection modules must sit on CPU with the PLE tables.
4. **Gemma 4 threads a mutable `shared_kv_states` dict through its layers**;
   accelerate hooks deep-copy kwargs and silently break KV sharing. Pass
   `skip_keys=model._skip_keys_device_placement`.

Only the 42 decoder layers need the GPU (~7.6GB); vision/audio towers, PLE
tables, and embeddings never see a gradient and live happily in system RAM.

## Roadmap

- [x] Baseline-controlled hallucination result + cross-model replication (gate passed 3/4; [FINDINGS](docs/FINDINGS.md) Part 4)
- [x] Quantitative emotion analysis across 6 covert conditions × 5 models ([FINDINGS](docs/FINDINGS.md) Parts 1-2)
- [x] Dense vs MoE workspace comparison (Finding 6)
- [x] Censored vs abliterated: belief vs behavior (Findings 2, 8; fabrication result in Part 4)
- [x] Classifier/router proof layer + published router weights (Part 5)
- [x] HuggingFace lens + trace + router releases ([solarkyle/jspace-lenses](https://huggingface.co/solarkyle/jspace-lenses))
- [ ] Escalation sidecar: OpenAI-compatible endpoint with `workspace_confidence` + auto-handoff
- [ ] Live visualizer: watch the workspace while the model generates
- [ ] Real inference-overhead measurement in a local serving stack
- [ ] Harder datasets where output confidence is miscalibrated; more model families

## Credits

- Paper & reference implementation: Anthropic -
  [transformer-circuits.pub/2026/workspace](https://transformer-circuits.pub/2026/workspace/index.html),
  [anthropics/jacobian-lens](https://github.com/anthropics/jacobian-lens) (Apache 2.0)
- Models: Google (Gemma 4), Alibaba (Qwen3.6), huihui-ai (abliteration)
- Scripts in this repo: MIT

*Built in one night, the day the paper dropped, by [@solarkyle](https://github.com/solarkyle).
I'm currently between roles and looking for work on model internals, evals, or
ML engineering - if this repo is the kind of thing your team does on purpose,
I'd love to hear from you: fintechkyle@gmail.com.*

## Contributing

Open to it. The issues tab has the experiment list, grab anything. Traces and
fitted lenses are on [HF](https://huggingface.co/solarkyle/jspace-lenses) so
most analysis ideas don't even need a GPU.
