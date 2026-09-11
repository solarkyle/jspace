"""Shared infrastructure for the Self-Focus Lab: provenance, loading, capture, storage.

Everything here is local. There is no remote path, no paid judge and no cloud
fallback, and `assert_local_only()` is called before any model work so a misread
config cannot start one.

Design notes that are load-bearing:

* Trial IDs are derived from a stable hash of the whole configuration plus the
  task content, never from Python's `hash()`, which is salted per process.
* Every trial starts from fresh conversation state with no cache carried over.
  Reusing a KV cache between trials would let one condition see another's prefix.
* Capture hooks are removed in a `finally`, and an intervention hook is removed
  before the assistant emits its first report token so the report is generated
  from the live patched state rather than a replay.
"""

from __future__ import annotations

import hashlib
import io
import json
import os
import platform
import time
from pathlib import Path
from typing import Any

HERE = Path(__file__).resolve().parent
REPO = HERE.parent.parent


# --------------------------------------------------------------------------- #
# configuration and stable identity
# --------------------------------------------------------------------------- #

def load_json(path: Path) -> dict:
    with io.open(path, encoding="utf-8") as fh:
        return json.load(fh)


def config() -> dict:
    return load_json(HERE / "config.json")


def conditions() -> dict:
    return load_json(HERE / "conditions.json")


def stable_id(*parts: Any) -> str:
    """Deterministic short id. Never Python's salted hash()."""
    digest = hashlib.sha256()
    for part in parts:
        digest.update(repr(part).encode("utf-8"))
        digest.update(b"\x1f")
    return digest.hexdigest()[:20]


def file_sha256(path: str | Path, limit_bytes: int | None = None) -> dict:
    p = Path(path)
    if not p.exists():
        return {"path": str(p), "present": False}
    digest = hashlib.sha256()
    read = 0
    with open(p, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            digest.update(chunk)
            read += len(chunk)
            if limit_bytes is not None and read >= limit_bytes:
                break
    return {"path": str(p), "present": True, "sha256": digest.hexdigest(),
            "bytes": p.stat().st_size,
            "hashed_prefix_only": limit_bytes is not None and read >= limit_bytes}


def assert_local_only(cfg: dict) -> None:
    """Refuse to proceed if the config would permit spending."""
    if cfg.get("allow_remote"):
        raise RuntimeError("allow_remote is true; this experiment is authorized for local compute only")
    if cfg.get("allow_paid_judge"):
        raise RuntimeError("allow_paid_judge is true; no paid judge is authorized")
    if float(cfg.get("cloud_spend_authorization_usd", 0.0)) != 0.0:
        raise RuntimeError("cloud_spend_authorization_usd must be 0.0")


# --------------------------------------------------------------------------- #
# compute budget
# --------------------------------------------------------------------------- #

class Budget:
    """Wall-clock ceiling over model compute. A ceiling, not an expected runtime."""

    def __init__(self, minutes: float) -> None:
        self.limit_s = float(minutes) * 60.0
        self.used_s = 0.0
        self._mark: float | None = None

    def __enter__(self) -> "Budget":
        self._mark = time.perf_counter()
        return self

    def __exit__(self, *exc: Any) -> None:
        if self._mark is not None:
            self.used_s += time.perf_counter() - self._mark
            self._mark = None

    @property
    def remaining_s(self) -> float:
        return max(0.0, self.limit_s - self.used_s)

    def exhausted(self) -> bool:
        return self.remaining_s <= 0.0

    def summary(self) -> dict:
        return {"ceiling_minutes": self.limit_s / 60.0,
                "used_minutes": round(self.used_s / 60.0, 3),
                "remaining_minutes": round(self.remaining_s / 60.0, 3),
                "exhausted": self.exhausted()}


# --------------------------------------------------------------------------- #
# model
# --------------------------------------------------------------------------- #

class Model:
    """Local Gemma/Qwen wrapper: generation, teacher-forced capture, injection."""

    def __init__(self, cfg: dict) -> None:
        assert_local_only(cfg)
        import torch
        import transformers

        self.torch = torch
        self.cfg = cfg
        self.model_id = str(cfg["model_id"])
        local = str(cfg.get("local_model_path") or "").strip()
        self.source = local or self.model_id
        self.precision = str(cfg.get("precision", "nf4")).lower()

        if local and not Path(local).exists():
            raise RuntimeError(f"local_model_path does not exist: {local}")

        quantized = self._checkpoint_is_quantized(self.source)
        if quantized and not torch.cuda.is_available():
            raise RuntimeError("the configured checkpoint is 4-bit and needs CUDA")
        if quantized and self.precision != "nf4":
            raise RuntimeError(
                f"precision={self.precision!r} but the checkpoint at {self.source} is "
                f"pre-quantized; refusing to mislabel the run")

        revision = str(cfg.get("model_revision") or "").strip()
        kwargs: dict[str, Any] = {"device_map": "cuda" if torch.cuda.is_available() else "cpu"}
        if revision and not local:
            kwargs["revision"] = revision
        # A pre-quantized checkpoint carries its own quantization_config and it
        # wins; passing a fresh BitsAndBytesConfig here would silently conflict.
        self.hf = self._from_pretrained_any(self.source, transformers, **kwargs)
        tokenizer_src = self.source if (Path(self.source) / "tokenizer_config.json").exists() \
            else self.model_id
        self.tokenizer = transformers.AutoTokenizer.from_pretrained(tokenizer_src)
        self.tokenizer_source = tokenizer_src
        self.hf.eval()

        self.blocks = self._find_blocks()
        self.n_layers = len(self.blocks)
        self.device = next(self.hf.parameters()).device

    # -- loading helpers ---------------------------------------------------- #

    @staticmethod
    def _checkpoint_is_quantized(source: str) -> bool:
        cfg_file = Path(source) / "config.json"
        if not cfg_file.exists():
            return False
        try:
            return "quantization_config" in load_json(cfg_file)
        except Exception:
            return False

    @staticmethod
    def _from_pretrained_any(source: str, transformers: Any, **kwargs: Any) -> Any:
        try:
            return transformers.AutoModelForCausalLM.from_pretrained(source, **kwargs)
        except ValueError:
            return transformers.AutoModelForImageTextToText.from_pretrained(source, **kwargs)

    def _find_blocks(self):
        """Locate the decoder block list without assuming one architecture."""
        candidates = [
            "model.language_model.layers", "language_model.model.layers",
            "model.model.layers", "model.layers", "layers",
        ]
        for dotted in candidates:
            node: Any = self.hf
            for part in dotted.split("."):
                node = getattr(node, part, None)
                if node is None:
                    break
            if node is not None and hasattr(node, "__len__") and len(node) > 0:
                self.block_path = dotted
                return node
        raise RuntimeError("could not locate decoder blocks on this model")

    def module_name(self, layer: int) -> str:
        return f"{self.block_path}[{layer}]"

    # -- prompting ---------------------------------------------------------- #

    def render(self, user_text: str) -> str:
        """Render one fresh single-turn prompt. Verified, never silently fallen back."""
        messages = [{"role": "user", "content": user_text}]
        try:
            return self.tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True)
        except Exception as exc:
            raise RuntimeError(
                f"chat template rendering failed for {self.tokenizer_source}: {exc}. "
                f"Refusing to fall back to different formatting, which would make "
                f"conditions incomparable.") from exc

    def encode(self, rendered: str):
        ids = self.tokenizer(rendered, return_tensors="pt",
                             add_special_tokens=False)["input_ids"]
        limit = int(self.cfg["max_prompt_tokens"])
        if ids.shape[1] > limit:
            raise ValueError(
                f"prompt is {ids.shape[1]} tokens, over the {limit} limit. Rejecting "
                f"rather than truncating, which would remove grounding evidence.")
        return ids.to(self.device)

    # -- generation --------------------------------------------------------- #

    def generate(self, user_text: str, max_new_tokens: int) -> dict:
        """Greedy generation from fresh state. Returns text, ids and per-token logprobs."""
        torch = self.torch
        rendered = self.render(user_text)
        input_ids = self.encode(rendered)
        t0 = time.perf_counter()
        with torch.no_grad():
            out = self.hf.generate(
                input_ids=input_ids,
                max_new_tokens=int(max_new_tokens),
                do_sample=False,
                return_dict_in_generate=True,
                output_scores=True,
                use_cache=True,
                pad_token_id=self.tokenizer.eos_token_id,
            )
        elapsed = time.perf_counter() - t0
        seq = out.sequences[0]
        gen_ids = seq[input_ids.shape[1]:].tolist()
        stop = self._stop_ids()
        cut = len(gen_ids)
        for i, tok in enumerate(gen_ids):
            if tok in stop:
                cut = i
                break
        gen_ids = gen_ids[:cut]
        step_logprobs = []
        for i in range(len(gen_ids)):
            lp = out.scores[i][0].float().log_softmax(-1)
            step_logprobs.append(float(lp[gen_ids[i]]))
        return {
            "rendered_prompt": rendered,
            "prompt_token_ids": input_ids[0].tolist(),
            "n_prompt_tokens": int(input_ids.shape[1]),
            "text": self.tokenizer.decode(gen_ids, skip_special_tokens=True),
            "gen_token_ids": gen_ids,
            "step_logprobs": step_logprobs,
            "seconds": round(elapsed, 3),
        }

    def _stop_ids(self) -> set:
        out = set()
        eos = getattr(self.tokenizer, "eos_token_id", None)
        if isinstance(eos, int):
            out.add(eos)
        for token in ("<end_of_turn>", "<|im_end|>", "<turn|>"):
            tid = self.tokenizer.convert_tokens_to_ids(token)
            if isinstance(tid, int) and tid >= 0:
                out.add(tid)
        return out

    # -- capture ------------------------------------------------------------ #

    def capture(self, user_text: str, layers: list[int], *,
                forced_continuation_ids: list[int] | None = None,
                positions: list[int] | None = None,
                patch: dict | None = None) -> dict:
        """One forward pass, recording residuals at `layers`.

        `forced_continuation_ids` teacher-forces the same continuation across
        conditions, so the emitted text is identical and only the instruction
        differs. `patch` optionally perturbs given prefill positions.
        """
        torch = self.torch
        rendered = self.render(user_text)
        input_ids = self.encode(rendered)
        if forced_continuation_ids:
            tail = torch.tensor([list(forced_continuation_ids)],
                                device=input_ids.device, dtype=input_ids.dtype)
            full = torch.cat([input_ids, tail], dim=1)
        else:
            full = input_ids

        recorded: dict[int, Any] = {}
        handles = []

        def make_recorder(layer: int):
            def hook(_module, _inputs, output):
                hidden = output if torch.is_tensor(output) else output[0]
                recorded[layer] = hidden.detach()[0].float().cpu()
                return output
            return hook

        def make_patcher(layer: int, direction, alpha: float, at_positions: list[int]):
            def hook(_module, _inputs, output):
                hidden = output if torch.is_tensor(output) else output[0]
                unit = direction.to(device=hidden.device, dtype=torch.float32)
                unit = unit / unit.norm().clamp_min(1e-12)
                changed = hidden.clone()
                for pos in at_positions:
                    if pos >= hidden.shape[1]:
                        continue
                    base = hidden[:, pos:pos + 1].float()
                    delta = unit * (float(alpha) * base.norm().clamp_min(1e-12))
                    changed[:, pos:pos + 1] = (base + delta).to(dtype=hidden.dtype)
                if torch.is_tensor(output):
                    return changed
                return (changed,) + tuple(output[1:])
            return hook

        t0 = time.perf_counter()
        try:
            # ORDER MATTERS. Forward hooks fire in registration order, so the
            # patcher must be registered FIRST: otherwise a recorder on the same
            # module captures the pre-patch output and the patched layer appears
            # unaffected by its own patch. Note also that a patch at layer L and
            # positions P changes layer L's output only at P; other positions at
            # that layer are unchanged until the next layer mixes them, so an
            # intervention is best observed downstream of L.
            if patch:
                handles.append(self.blocks[int(patch["layer"])].register_forward_hook(
                    make_patcher(int(patch["layer"]), patch["direction"],
                                 float(patch["alpha"]), list(patch["positions"]))))
            for layer in layers:
                handles.append(self.blocks[layer].register_forward_hook(make_recorder(layer)))
            with torch.no_grad():
                self.hf(full)
        finally:
            # removed even on exception, so a failed trial cannot leak a hook
            for handle in handles:
                handle.remove()
        elapsed = time.perf_counter() - t0

        n_prompt = int(input_ids.shape[1])
        total = int(full.shape[1])
        if positions is None:
            limit = int(self.cfg["max_recorded_positions"])
            span = list(range(n_prompt - 1, total))
            positions = span[:limit] if len(span) > limit else span
        bad = [p for p in positions if p < 0 or p >= total]
        if bad:
            raise ValueError(
                f"requested positions {bad} are outside the {total}-token sequence. "
                f"Rejecting rather than clipping: a silently shortened position list "
                f"would make conditions incomparable.")
        out_layers = {}
        for layer, tensor in recorded.items():
            out_layers[layer] = {
                "module": self.module_name(layer),
                "positions": positions,
                "residual_norms": [float(tensor[p].norm()) for p in positions],
                "residuals": [tensor[p].tolist() for p in positions],
            }
        return {
            "rendered_prompt": rendered,
            "prompt_token_ids": input_ids[0].tolist(),
            "n_prompt_tokens": n_prompt,
            "forced_continuation_ids": list(forced_continuation_ids or []),
            "total_tokens": total,
            "layers": out_layers,
            "seconds": round(elapsed, 3),
        }

    # -- provenance --------------------------------------------------------- #

    def provenance(self) -> dict:
        torch = self.torch
        import transformers
        gpu = {}
        if torch.cuda.is_available():
            props = torch.cuda.get_device_properties(0)
            free, total = torch.cuda.mem_get_info()
            gpu = {"name": props.name, "total_gb": round(props.total_memory / 1e9, 2),
                   "free_gb": round(free / 1e9, 2),
                   "peak_allocated_gb": round(torch.cuda.max_memory_allocated() / 1e9, 3)}
        weight_files = sorted(
            [str(p.name) for p in Path(self.source).glob("*.safetensors")]
        ) if Path(self.source).is_dir() else []
        hashes = {}
        if Path(self.source).is_dir():
            for name in ("config.json", "tokenizer_config.json", "chat_template.jinja"):
                candidate = Path(self.source) / name
                if candidate.exists():
                    hashes[name] = file_sha256(candidate)
            for name in weight_files:
                # prefix hash only: full hashes of multi-GB shards are not worth
                # the minutes, and the prefix still detects a swapped file
                hashes[name] = file_sha256(Path(self.source) / name, limit_bytes=1 << 24)
        return {
            "model_id": self.model_id,
            "loaded_from": self.source,
            "tokenizer_source": self.tokenizer_source,
            "precision": self.precision,
            "model_revision": str(self.cfg.get("model_revision") or "") or "(unpinned; local snapshot)",
            "architecture": type(self.hf).__name__,
            "block_path": self.block_path,
            "n_layers": self.n_layers,
            "device": str(self.device),
            "device_map_requested": "cuda" if torch.cuda.is_available() else "cpu",
            "weight_files": weight_files,
            "file_hashes": hashes,
            "packages": {
                "torch": torch.__version__,
                "transformers": transformers.__version__,
            },
            "python": platform.python_version(),
            "platform": platform.platform(),
            "gpu": gpu,
        }


# --------------------------------------------------------------------------- #
# durable trial storage
# --------------------------------------------------------------------------- #

class TrialStore:
    """One durable JSON line per trial, resumable, provenance-guarded."""

    def __init__(self, path: Path, provenance_fingerprint: str) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.fingerprint = provenance_fingerprint
        self.done: set[str] = set()
        if self.path.exists():
            with io.open(self.path, encoding="utf-8") as fh:
                for line in fh:
                    if not line.strip():
                        continue
                    row = json.loads(line)
                    stored = row.get("provenance_fingerprint")
                    if stored != self.fingerprint:
                        raise RuntimeError(
                            f"{self.path} was written under provenance {stored!r} but this "
                            f"run is {self.fingerprint!r}. Refusing to resume into a run "
                            f"with different model, prompts or configuration.")
                    self.done.add(row["trial_id"])
        self.handle = io.open(self.path, "a", encoding="utf-8")

    def has(self, trial_id: str) -> bool:
        return trial_id in self.done

    def write(self, row: dict) -> None:
        if row["trial_id"] in self.done:
            raise RuntimeError(f"duplicate trial_id {row['trial_id']}")
        row["provenance_fingerprint"] = self.fingerprint
        self.handle.write(json.dumps(row, ensure_ascii=False) + "\n")
        self.handle.flush()
        os.fsync(self.handle.fileno())
        self.done.add(row["trial_id"])

    def close(self) -> None:
        try:
            self.handle.close()
        except Exception:
            pass


def read_trials(path: Path) -> list[dict]:
    rows = []
    p = Path(path)
    if not p.exists():
        return rows
    with io.open(p, encoding="utf-8") as fh:
        for line in fh:
            if line.strip():
                rows.append(json.loads(line))
    return rows


# --------------------------------------------------------------------------- #
# prompt assembly
# --------------------------------------------------------------------------- #

def build_prompt(cond_name: str, task_block: str, conds: dict) -> str:
    """Same outer instructions and task for every condition.

    The manipulation lives only in the delimited instruction block, and the
    condition ID is never shown to the model.
    """
    spec = conds["conditions"][cond_name]
    parts = [conds["shared_outer"], ""]
    if spec["uses_jspace_explanation"]:
        parts += [conds["jspace_explanation"], ""]
    parts += ["--- instruction ---", spec["instruction"], "--- end instruction ---", ""]
    parts += [task_block]
    return "\n".join(parts)


def select_layers(n_layers: int, cfg: dict, lens_band: list[int] | None) -> dict:
    """At most four layers, evenly spaced, from the lens band when available."""
    k = int(cfg["max_recorded_layers"])
    if lens_band:
        pool = sorted(lens_band)
        basis = "lens fitted band"
    else:
        lo, hi = int(n_layers * 0.25), int(n_layers * 0.75)
        pool = list(range(lo, hi))
        basis = "predeclared evenly spaced intermediate layers (no lens)"
    if not pool:
        raise RuntimeError("no candidate layers available")
    if len(pool) <= k:
        chosen = pool
    else:
        step = (len(pool) - 1) / (k - 1) if k > 1 else 0
        chosen = [pool[int(round(i * step))] for i in range(k)]
    return {"layers": sorted(set(chosen)), "basis": basis,
            "pool_size": len(pool), "zero_based": True,
            "note": "Not called a validated workspace band for this model unless "
                    "separately established."}


def lens_band_from(lens_path: str, n_layers: int) -> tuple[list[int] | None, dict]:
    """Fitted layers inside the 25-75 percent depth band, if a lens is present."""
    info: dict[str, Any] = {"lens_path": str(lens_path)}
    p = Path(lens_path)
    if not p.exists():
        info.update({"available": False, "reason": "file not found"})
        return None, info
    try:
        import jlens
        lens = jlens.JacobianLens.load(str(p))
        fitted = sorted(int(k) for k in lens.jacobians)
        band = [l for l in fitted if int(n_layers * 0.25) <= l < int(n_layers * 0.75)]
        info.update({"available": True, "fitted_layers": fitted, "band": band,
                     "sha256": file_sha256(p, limit_bytes=1 << 24).get("sha256")})
        return band or None, info
    except Exception as exc:
        info.update({"available": False, "reason": f"{type(exc).__name__}: {exc}"})
        return None, info
