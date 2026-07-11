# Hard rules: things we verified that you can rely on

Operational facts, each one tested in this repo. If you build on the sidecar
or the lenses, these are the constraints that matter. Receipts in
[FINDINGS.md](FINDINGS.md), [TLDR.md](TLDR.md), and the traces in `data/`.

## 1. One lens per model, any quantization

A lens fitted on bf16 activations reads NF4 4-bit activations without
refitting. Verified on both 12Bs (`out/uncertainty_shapeq4_*.jsonl`): the
noise signal survives quantization and quantization does not add smear.
Do not fit a separate "Q4 lens", it is wasted compute. Weight quantization
changes the activations far less than it changes the weights.

## 2. Sampling settings cannot affect the reading

Temperature, top_p, top_k act on the output logits AFTER the forward pass.
The lens reads hidden states BEFORE any of that. The noise score at answer
onset is identical at temperature 0 and 1.5 by construction: the snapshot is
a function of the prompt-conditioned forward pass, which the sampler never
touches. Corollary: you cannot
"fix" a noisy workspace by lowering temperature, and greedy decoding does
not make a fabrication less of a fabrication, it just makes it repeatable.
(Later tokens are conditioned on whatever was sampled, so scores past the
first token can differ across samples. The gate reads the first.)

## 3. The validated read is answer-onset; deployment now checks a short prefix

The published AUC/quadrant numbers read the workspace at answer onset. If the
model preambles ("**The** singer who had...") token one is filler with a clean
workspace and the old single-token read can miss. Measured: same question
scored 0.03 with preamble, 0.90 when forced terse.

The sidecar now scores the first `WORKSPACE_READ_TOKENS` generated tokens
(default 3) and routes on the highest-risk snapshot. Set
`WORKSPACE_READ_TOKENS=1` to reproduce the original experiments exactly. Keep
`LOCAL_TERSE=1` unless you are explicitly testing the prefix read: the router is
still calibrated on answer-leading generations, and extra preamble tokens cost
extra lens passes.

## 4. The danger signal is tail smear, not deliberation

Workspace entropy decomposes into rival mass (probability on candidates 2-5,
the model weighing real options) and tail smear (mass spread past the top
~20 tokens, undirected noise). Entropy correlates with tail smear at
0.93-0.999 on all five models. Deliberation is the SAFEST state we measured.
A gate that punishes a model for considering two good answers is
miscalibrated; gate on the smear.

## 5. Normalization must be frozen

Router features are z-scored against fixed stats shipped with the router
artifact. The all-model router has per-model `norm_stats`; the single-model
sidecar file `sidecar/norm_stats.json` is a legacy/default fallback. A rolling
window drifts with traffic mix and makes the same question score differently
depending on what was asked before it. Deterministic scores or no scores.

## 6. Thresholds transfer within a family, not across families

One rule fit on E4B (escalate when z-entropy > 0) catches 62-70% of wrong
answers on every Gemma with zero tuning. On Qwen it is chance. Same for the
trained router (zero-shot 0.74-0.78 across Gemmas, 0.65 on Qwen). Deploying
on a new Gemma needs no calibration set; deploying on a new family needs
validation before you trust a single number.

## 7. It detects noisy retrieval, not confident misconceptions

A model that firmly believes something wrong reads clean. This is a stated
limit of the method, not a bug. Do not sell it as a truth detector.

## 8. Fake entities are not the use case

Output logprob detects fabricated names near-perfectly on capable models
(AUC 0.94-1.00). The workspace adds nothing there. The value is on
real-but-hard questions answered confidently and wrongly, which is exactly
the cell output confidence cannot see (42% vs 75% correct at matched
confidence).

## 9. The current 4-bit path is CUDA-only

bitsandbytes NF4 does not run on CPU. CPU deployment means bf16/int8-dynamic
(slow) or waiting on the llama.cpp lens port (roadmap, not built).

## 10. Reasoning models as the escalation target need headroom

Reasoning models burn max_tokens on hidden reasoning and return empty
content if the budget is small. The sidecar forces max_tokens >= 1024 and
sends `reasoning: {enabled: false}` upstream. If you swap the escalation
target, keep both.

## 11. Pin model revisions

Each lens was fit against a specific HF revision (listed in the
[lens repo card](https://huggingface.co/solarkyle/jspace-lenses)). A wrong
architecture fails loudly at load; a silently retrained same-name checkpoint
would not. Pin revisions in anything you ship.

## 12. The snapshot is effectively free

One extra lens application at answer onset, a few milliseconds on GPU
(`snapshot_ms` in every response). No second forward pass, no sampling. The
cost argument against sampling-based methods (semantic entropy needs 5-10
full generations) is real and stays real at deployment.
