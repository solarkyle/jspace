# J-Space experiment lab

The lab at `http://localhost:8765/lab` contains three interoperable MVPs. They
all use the same fitted lens and the same answer-onset layer band.

## Causal branch debugger

1. Run a normal traced response.
2. Choose a fork token, fitted layer, observed token concept, boost/suppress,
   and intervention strength.
3. The backend computes the local gradient of that token's J-Lens log
   probability with respect to the selected residual, applies a one-step
   additive patch, and greedily generates the patched and unpatched futures
   from the same prefix.

Strength `1.0` applies a delta whose norm is 5 percent of the residual norm.
The response reports the exact relative norm and the largest changes in the
model's next-token distribution.

This establishes a causal effect of an activation patch on the computation.
It does not establish that a displayed token is a discrete internal thought,
nor that the local gradient is the unique direction representing that concept.

## Overload observatory

The exploratory sweep increases arbitrary name-to-value bindings and records:

- exact answer correctness;
- router risk;
- full-vocabulary token entropy;
- probability mass below the top 20 tokens;
- rival mass on candidates 2 through 5;
- family-corrected entropy;
- ignition depth.

Family-corrected entropy conservatively merges case, whitespace, tokenizer
marker, and Unicode-compatibility variants among the top 64 tokens. It
subtracts only the entropy removed by those observed merges. The rest of the
full-vocabulary entropy remains untouched, so rising tail smear cannot be
hidden by collapsing the tail into one bucket.

The UI sweep is a fast single-trial instrument check, not the preregistered
phase-transition experiment. The full study needs repeated prompt families,
token-count controls, randomized targets, and a registered changepoint versus
smooth-degradation comparison.

## HF to GGUF conformance

The HF capture stores, at the final prompt position:

- exact input token IDs;
- the post-block residual for every fitted layer;
- residual norm;
- J-Lens top-32 IDs and probabilities;
- final next-token distribution.

The comparison gate reports per-layer residual cosine, sparse-distribution
Jensen-Shannon divergence, top-k overlap, and final-token agreement.

The exporter integrates with
[`jlens-gguf`](https://github.com/igorbarshteyn/jlens-gguf), which captures
`l_out-<layer>` tensors from llama.cpp. It uses the HF reference's token IDs,
so chat-template differences cannot silently contaminate conformance.

```bash
# In jlens-gguf: convert the exact Anthropic-format lens once.
python -m jlens_gguf convert-pt /path/to/lens.pt /path/to/lens.gguf

# Start its native activation server on :8091, then export a candidate.
PYTHONPATH=/path/to/jlens-gguf python -m sidecar.export_gguf_capture \
  --reference jspace-hf-reference.json \
  --native-url http://127.0.0.1:8091 \
  --model-gguf /path/to/gemma.gguf \
  --lens-gguf /path/to/lens.gguf \
  --output jspace-gguf-candidate.json

# Compare offline or load the candidate in the lab.
python -m sidecar.conformance_cli \
  jspace-hf-reference.json jspace-gguf-candidate.json
```

The model checkpoint and tokenizer must correspond to the HF reference. A
quantized GGUF is expected to differ slightly; the harness measures that
difference rather than assuming it away.
