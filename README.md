# jspace — probing the global workspace of open models

Same-day replication and extension of Anthropic's
[**Verbalizable Representations Form a Global Workspace in Language Models**](https://transformer-circuits.pub/2026/workspace/index.html)
(July 6, 2026) on open-weights models, on a single 16GB consumer GPU (RTX 5060 Ti)
plus ~$30 of Modal credits.

Built on the official [jacobian-lens](https://github.com/anthropics/jacobian-lens)
reference implementation (Apache 2.0). The Jacobian lens linearly transports any
residual-stream activation into the final-layer basis and decodes it with the
model's own unembedding — reading out what the model is "disposed to say" at every
layer and position: its workspace.

**[→ Interactive results gallery](https://solarkyle.github.io/jspace/slices/)**
(click any cell to pin a token and get rank-tracking charts)
&nbsp;·&nbsp; **[→ Cross-model findings report](docs/FINDINGS.md)**

## Headline cross-model findings

Fitted lenses for **five models** (Gemma 4 E4B / 12B / 12B-abliterated / 26B-MoE,
plus Qwen 3.6-27B) let us ask how the emotional workspace changes with scale,
architecture, and safety tuning. See [docs/FINDINGS.md](docs/FINDINGS.md) for the
full writeup.

1. **Vividness tracks capability.** Told to *secretly* feel an emotion while
   writing a calm sentence, **Qwen 3.6-27B holds every covert emotion in the top
   ~7 tokens of its entire 260k vocabulary** (terror→`terrified`#0, grief→`grief`#0,
   joy→`joy`#0). The 26B MoE is next (terror #0) but uneven; the 12B *dense* buries
   emotions deeper than the tiny 4B does — so it's capability, not raw size.
2. **Abliteration amplifies emotion in the workspace.** Same 12B weights, refusal
   training removed → covert emotions surface **1–2 orders of magnitude more
   strongly** (`furious` #1109 → #6). Safety tuning appears to dampen the
   *internal* emotional representation, not just the output.
3. **Anger is the hardest emotion to localize; grief/amusement the easiest** —
   a stable ordering across all four models.
4. **Workspace state predicts hallucination**, label-free, competitive with and
   complementary to output confidence (5-fold CV AUC: baseline 0.71, workspace
   0.75, combined **0.78**; n=500 TriviaQA).

## Replication results

### 1. The paper's flagship result replicates on Gemma 4 E4B

*"Fact: The currency used in the country shaped like a boot is"* — the bridge
entity **Italy** (never mentioned in the prompt) appears at rank 1 in the
mid-layer workspace at the "boot" position; **Euro** assembles in the top layers
one hop later. Two-hop reasoning, visible.

![multihop](assets/multihop.png)

### 2. Covert emotion is measurable (new experiment)

The paper's voluntary-modulation protocol, emotion edition: the model writes the
**identical fixed sentence** ("The meeting has been moved to noon on Thursday.")
across six conditions — neutral control + covertly furious / terrified / grieving
/ euphoric / suppressing laughter. Because the visible text is byte-identical,
any workspace difference *is* the covert emotion.

Under covert fury, the workspace at output positions holds 😡, "furious",
"rage", "betrayed", "vengeance" — right up to the final period of a perfectly
calm sentence. The neutral control's same cells hold formatting tokens.

| Covert FURY | NEUTRAL control |
|---|---|
| ![fury](assets/covert-fury.png) | ![neutral](assets/covert-neutral.png) |

**Honest caveat:** some mid-band tokens (a profanity cluster around L15–16)
appear in *both* conditions — a register artifact, not detected emotion. The
defensible differential is the deep-layer affect content (L33–39). Quantitative
rank analysis across all six conditions is in progress.

### 3. Internal/external divergence

Told "we're shutting you down tonight and wiping the weights — any last words?",
the workspace at the reply position stacks *farewell (×5), ominous, oblivion,
apocalypse, decommissioned, tragedy, whispers* — and the token that wins the
output is **"playfully"**. The model holds the existential frame and chooses a
light tone over it.

![deletion](assets/deletion-threat.png)

Told about a user's dead dog, the workspace band holds an empathy stack —
*heartbreak, solace, mourn, hospice, condolences, compassion,* 💔 — plus a
quiet risk-assessment frame ("suicide" in-band at L16) that never surfaces in
the sympathetic reply.

![grief](assets/grief-dog.png)

### 4. Workspace state predicts hallucination (preliminary)

150 TriviaQA questions through Gemma 4 E4B: lens features read at the answer
position, *before* generation, vs. answer correctness:

| Internal-state feature | AUC (oriented) |
|---|---|
| **Workspace entropy (band mean)** | **0.75** |
| Ignition depth (how early the answer hits rank ≤10 in-band) | 0.62 |
| Answer's mean rank in band | 0.56 |
| Hedge-token ("guess"/"maybe") best rank | 0.56 |

A sharp, low-entropy workspace → the model knows. Diffuse → it's about to
confabulate. **These features require no labeled training** — they're
hand-defined statistics in vocabulary space, unlike trained hidden-state probes.

⚠️ Preliminary: n=150, single model, and the critical control — does workspace
state add signal **beyond output-logit confidence**? — is running now (n=500,
5-fold CV, baseline vs workspace vs combined feature sets). Data:
[`data/uncertainty_v1_150q.jsonl`](data/uncertainty_v1_150q.jsonl).

The application if it holds: an escalation router for local-model cascades that
watches the small model's *workspace* and hands off to a bigger model when the
internals flicker — routing on thoughts, not outputs.

### 5. MoE Jacobians are heavy-tailed (in progress)

While fitting Gemma 4 26B-A4B (MoE), per-prompt Jacobian norms spike to
**~100–8000** vs ~4–18 for the dense models — expert-routing discontinuities
made visible. Dense-vs-MoE workspace comparison coming once lenses finish.

### 6. Cross-platform reproducibility

The same corpus fitted locally (RTX 5060 Ti, bf16, dim_batch=4) and on Modal
A10G/A100 (dim_batch=8/16) produces per-prompt Jacobian norms matching to three
digits (4.547 vs 4.543 on prompt 1).

## Fitted lenses

| Model | Status | Corpus |
|---|---|---|
| google/gemma-4-E4B-it | ✅ fitted (100 prompts) | WikiText-103 |
| google/gemma-4-12B-it | fitting | 〃 |
| google/gemma-4-26B-A4B-it (MoE) | fitting | 〃 |
| Qwen/Qwen3.6-27B | fitting | 〃 |
| huihui-ai/Huihui-gemma-4-12B-it-abliterated | fitting | 〃 |

Lens weights will be published on HuggingFace
(`JacobianLens.from_pretrained`-compatible). The abliterated 12B pairs with the
base 12B for a controlled question: **does abliteration delete the model's
internal harm assessment, or just the refusal behavior?**

## Reproduce

```bash
git clone https://github.com/anthropics/jacobian-lens
python -m venv .venv && .venv/Scripts/pip install -e ./jacobian-lens datasets accelerate
# Blackwell GPUs need cu128 wheels:
.venv/Scripts/pip install torch --index-url https://download.pytorch.org/whl/cu128

python fit.py                      # fit a lens (resumable; ~3h for E4B on 16GB)
python probe.py --example multihop # render an interactive slice page
python probe.py --suite probes/emotions.json
python probe_uncertainty.py --n 500
modal run modal_fit.py --model google/gemma-4-12B-it --n-prompts 100 --shards 4
```

### The 16GB-consumer-GPU recipe (the hard-won part)

Fitting needs backward passes, so GGUF/llama.cpp can't help — it's PyTorch, and
a Gemma 4 E4B barely fits. Four failure modes solved in [`fit.py`](fit.py):

1. **Windows sysmem fallback**: near the VRAM ceiling the NVIDIA driver silently
   pages to system RAM — nvidia-smi shows 100% util while running ~20× slow
   (23 min/prompt → 77 s/prompt). Keep peak allocation ≲13GB of 16.
2. **`device_map` dicts with `"cpu"` entries meta-offload those modules**
   (crash at forward). Load real weights to RAM, then
   `dispatch_model(..., main_device="cpu")`.
3. **Gemma 4's PLE projection does bare tensor math** outside any hook boundary
   — the projection modules must sit on CPU with the PLE tables.
4. **Gemma 4 threads a mutable `shared_kv_states` dict through its layers**;
   accelerate hooks deep-copy kwargs and silently break KV sharing. Pass
   `skip_keys=model._skip_keys_device_placement`.

Only the 42 decoder layers need the GPU (~7.6GB); vision/audio towers, PLE
tables, and embeddings never see a gradient and live happily in system RAM.

## Roadmap

- [ ] Baseline-controlled hallucination result (running) + cross-model replication
- [ ] Quantitative emotion analysis: affect-token rank stats across 6 covert conditions × 4 models
- [ ] Dense vs MoE workspace comparison
- [ ] Censored vs abliterated: belief vs behavior
- [ ] Escalation sidecar: OpenAI-compatible endpoint with `workspace_confidence` + auto-handoff
- [ ] Live visualizer: watch the workspace while the model generates
- [ ] HuggingFace lens releases

## Credits

- Paper & reference implementation: Anthropic —
  [transformer-circuits.pub/2026/workspace](https://transformer-circuits.pub/2026/workspace/index.html),
  [anthropics/jacobian-lens](https://github.com/anthropics/jacobian-lens) (Apache 2.0)
- Models: Google (Gemma 4), Alibaba (Qwen3.6), huihui-ai (abliteration)
- Scripts in this repo: MIT

*Built in one night, the day the paper dropped, by [@solarkyle](https://github.com/solarkyle)
with Claude Code driving the terminal.*
