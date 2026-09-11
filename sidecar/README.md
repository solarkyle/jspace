# The lie detector sidecar

An OpenAI-compatible chat server that reads its own model's mind. Every
response includes a noise score from the model's internal workspace telling
you whether the answer came from clean retrieval or from static. By default
the sidecar reads the first three answer-token workspaces and routes on the
noisiest one, which patches the old "first token was just filler" failure
mode. Three interfaces are built in: the original heatmap chat, an interactive
2D semantic galaxy with answer-token and layer scrubbing, and an experiment
lab for causal branches, binding overload, and HF/GGUF conformance.

It detects noisy retrieval, not confident misconceptions. Read
[docs/HARD_RULES.md](../docs/HARD_RULES.md) before building on it.

## What you need

- A CUDA GPU with ~9GB free VRAM for the default E4B model at 4-bit.
- Model weights: downloaded automatically from HF on first run. Gemma is
  gated, so accept the license once at
  https://huggingface.co/google/gemma-4-E4B-it and run `hf auth login`.
- The fitted lens: downloaded automatically on first run from
  [solarkyle/jspace-lenses](https://huggingface.co/solarkyle/jspace-lenses)
  (or point `LENS_PATH` at a local `lens.pt`).

## Quickstart

```bash
git clone https://github.com/solarkyle/jspace && cd jspace
git clone https://github.com/anthropics/jacobian-lens   # the lens library (Apache 2.0)
python -m venv .venv && .venv/Scripts/activate   # or bin/activate on linux
pip install -e ./jacobian-lens
pip install torch transformers accelerate bitsandbytes fastapi uvicorn huggingface_hub

# run it (first run downloads the model + lens)
MODEL_ID=google/gemma-4-E4B-it QUANT=4bit python -m uvicorn sidecar.server:app --port 8765
```

Then open **http://localhost:8765/galaxy** for the semantic galaxy, or
**http://localhost:8765/lab** for the experiment workbench. The original
heatmap remains at **http://localhost:8765/chat**. Reliable
tripwires: obscure pop-culture trivia phrased tersely ("Which singer had a
big 60s No 1 with Roses Are Red?"). Things it should stay green on:
"capital of France".

The galaxy reuses the workspace snapshots already computed by the router.
Node activity is J-Lens top-k probability. Its 2D positions are a PCA of
those probability trajectories within the current trace, and its edges are
projection neighbors. The display does not label those edges as causal.

## The four modes

| mode | over threshold | use |
|---|---|---|
| `detect` (default) | flag it, answer locally anyway | the demo, exploring |
| `escalate` | forward to a bigger model, return its answer | actual routing |
| `refuse` | return "I am not confident enough to answer that one." | safety-first |
| `tag` | answer locally, mark `jspace.action="tagged"` | logging/eval |

Escalation config (any OpenAI-compatible endpoint):

```bash
ESCALATE_URL=https://openrouter.ai/api/v1/chat/completions \
ESCALATE_MODEL=z-ai/glm-5.2 \
OPENROUTER_API_KEY=sk-or-... \
RISK_THRESHOLD=0.6 \
MODEL_ID=google/gemma-4-E4B-it QUANT=4bit python -m uvicorn sidecar.server:app --port 8765
```

## Using the API

Standard chat completions, plus a `jspace` block on every response:

```bash
curl -s http://127.0.0.1:8765/v1/chat/completions -H "Content-Type: application/json" -d '{
  "messages": [{"role":"user","content":"Which singer had a big 60s No 1 with Roses Are Red?"}],
  "mode": "detect"
}'
```

```jsonc
  "jspace": {
  "noise": 0.90,            // 0..1, the score
  "action": "flagged",      // clean | flagged | escalated | refused | tagged
  "answered_by": "google/gemma-4-E4B-it",
  "snapshot_ms": 21,        // lens cost for this request
  "layer_entropies": [...], // per-band-layer entropy trajectory
  "band_tokens": [...],     // top tokens at 3 sampled layers
  "workspace_grid": {...},  // the full heatmap: layers x candidate tokens
  "galaxy_trace": {...},    // compact multi-token 2D trace payload
  "threshold": 0.6
}
```

`POST /escalate_one {"messages":[...]}` forces one cloud escalation without
re-running the local model (the "escalate this one" button in the UI).
`GET /health` reports model, quant, escalation target, threshold.

Experiment endpoints:

| endpoint | purpose |
|---|---|
| `POST /experiments/branch` | fork generation after a one-step concept boost or suppression |
| `POST /experiments/overload` | run a deterministic temporary-binding load sweep |
| `POST /experiments/conformance/capture` | export an HF/NF4 layer fingerprint and exact input token IDs |
| `POST /experiments/conformance/compare` | compare HF and GGUF fingerprints under explicit gates |

See [EXPERIMENTS.md](EXPERIMENTS.md) for evidence boundaries, capture schemas,
and the jlens-gguf exporter workflow.

## Knobs

| env | default | what |
|---|---|---|
| `MODEL_ID` | `google/gemma-4-E4B-it` | any model with a fitted lens |
| `MODEL_PATH` | off | local checkpoint dir to load weights from instead of downloading `MODEL_ID`. `MODEL_ID` still names the lens/router. See "Skipping the 16GB download" |
| `QUANT` | `4bit` | `4bit` (bitsandbytes NF4, CUDA only) or `bf16` |
| `LENS_PATH` | auto | local lens.pt; otherwise auto-download |
| `LENS_HUB_REPO` | `solarkyle/jspace-lenses` | where auto-download looks |
| `RISK_THRESHOLD` | `0.6` | noise level that trips the gate |
| `LOCAL_TERSE` | `1` | inject the terse system prompt. Do not turn this off casually: the read is still calibrated for answer-leading generations, even though the sidecar now checks the first few tokens (hard rule 3) |
| `WORKSPACE_READ_TOKENS` | `3` | score the first N generated answer-token workspaces and route on the highest-risk snapshot. Set `1` to reproduce the original answer-onset experiments |
| `LENS_DEVICE` | gpu | set `cpu` under VRAM pressure |
| `ESCALATE_URL` / `ESCALATE_MODEL` / `ESCALATE_KEY_ENV` | off | escalation target |

The released router files now carry frozen per-model feature normalization
stats. `sidecar/norm_stats.json` is kept as a legacy/default fallback. Do not
use rolling traffic stats for scoring (hard rule 5).

## Skipping the 16GB download

The default path downloads the BF16 checkpoint (~16GB) and quantizes it to
NF4 at load time. For deployment (or any second machine) do that once and
save the quantized result (~5GB):

```bash
python -m sidecar.save_nf4                      # writes <cache>/nf4/gemma-4-e4b-it
MODEL_PATH=/path/to/nf4/gemma-4-e4b-it python -m uvicorn sidecar.server:app --port 8765
```

The saved directory is plain safetensors with the quantization config
embedded, so it loads directly onto the GPU with no BF16 step. It can also be
pushed to a (private, Gemma-license-respecting) HF model repo and used as the
Space's download source. `AUTO_FALLBACK` does not apply to `MODEL_PATH`: a
broken explicit checkpoint fails loudly instead of silently re-downloading.

## Batch demo

`python sidecar/demo.py` fires 25 mixed questions (easy, obscure, fake) at a
running server and prints a live routing table + scoreboard.
