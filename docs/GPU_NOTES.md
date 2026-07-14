# GPU notes: the 16GB-consumer-GPU recipe (the hard-won part)

Fitting a Jacobian lens needs backward passes, so GGUF/llama.cpp can't help.
It's PyTorch, and a Gemma 4 E4B barely fits on a 16GB card (RTX 5060 Ti).
Four failure modes solved in [`fit.py`](../fit.py):

1. **Windows sysmem fallback**: near the VRAM ceiling the NVIDIA driver silently
   pages to system RAM. nvidia-smi shows 100% util while running ~20× slow
   (23 min/prompt → 77 s/prompt). Keep peak allocation ≲13GB of 16.
2. **`device_map` dicts with `"cpu"` entries meta-offload those modules**
   (crash at forward). Load real weights to RAM, then
   `dispatch_model(..., main_device="cpu")`.
3. **Gemma 4's PLE projection does bare tensor math** outside any hook boundary:
   the projection modules must sit on CPU with the PLE tables.
4. **Gemma 4 threads a mutable `shared_kv_states` dict through its layers**;
   accelerate hooks deep-copy kwargs and silently break KV sharing. Pass
   `skip_keys=model._skip_keys_device_placement`.

Only the 42 decoder layers need the GPU (~7.6GB); vision/audio towers, PLE
tables, and embeddings never see a gradient and live happily in system RAM.

## Cross-platform reproducibility

The same corpus fitted locally (RTX 5060 Ti, bf16, dim_batch=4) and on Modal
A10G/A100 (dim_batch=8/16) produces per-prompt Jacobian norms matching to three
digits (4.547 vs 4.543 on prompt 1).

## Blackwell wheels

```bash
python -m venv .venv && .venv/Scripts/pip install -r requirements.txt
# Blackwell GPUs need cu128 wheels:
.venv/Scripts/pip install torch --index-url https://download.pytorch.org/whl/cu128
```

The full GPU reproduction command list (lens fitting, probes, figures, Modal
runs) is in [EARLY_EXPLORATIONS.md](EARLY_EXPLORATIONS.md#reproduce-full-pipeline-gpu-required).
The campaign's generation scripts live in [`../campaign/`](../campaign/)
(`launch_stage1.sh`, `launch_stage2.sh`; Modal, L40S/A100).
