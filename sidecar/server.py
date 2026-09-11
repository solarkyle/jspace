from __future__ import annotations

import json
import math
import os
import re
import time
import uuid
import urllib.error
import urllib.request
from collections import deque
from dataclasses import dataclass
from pathlib import Path
from threading import Lock
from typing import Any

# Keep new downloads off C: when this repo is run on the author's Windows box.
if os.path.isdir("E:/hf-cache"):
    os.environ.setdefault("HF_HOME", "E:/hf-cache")

import numpy as np
import torch
import transformers
from fastapi import Body, FastAPI, HTTPException
from fastapi.responses import HTMLResponse

import jlens
from jlens.hooks import ActivationRecorder

from sidecar.conformance import compare_captures
from sidecar.experiments import (
    aggregate_top_token_families,
    answer_matches_expected,
    binding_prompt,
    entropy_from_masses,
)
from sidecar.galaxy import trajectory_layout


ROOT = Path(__file__).resolve().parents[1]
SIDECAR = Path(__file__).resolve().parent
BAND_LO = 0.25
BAND_HI = 0.75
LENS_HUB_REPO = os.environ.get("LENS_HUB_REPO", "solarkyle/jspace-lenses")
VRAM_BUDGET_BYTES = int(13 * 1024**3)
HEDGE_WORDS = [
    " guess",
    " maybe",
    " unsure",
    " unknown",
    " perhaps",
    " possibly",
    " unclear",
    " uncertain",
    "?",
    " hmm",
    " Hmm",
    " probably",
]


@dataclass
class Config:
    model_id: str
    model_path: str
    quant: str
    lens_path: Path
    lens_path_explicit: bool
    router_path: Path
    escalate_url: str
    escalate_model: str
    escalate_key_env: str
    risk_threshold: float
    max_prompt_tokens: int
    max_new_tokens: int
    escalate_timeout_s: float
    lens_device: str
    auto_fallback: bool
    workspace_read_tokens: int


@dataclass
class LocalAnswer:
    answer: str
    gen_ids: list[int]
    prompt_tokens: int
    finish_reason: str
    features: dict[str, Any]
    layer_entropies: list[float]
    band_tokens: list[dict[str, Any]]
    workspace_grid: dict[str, Any]
    galaxy_trace: dict[str, Any]
    risk: float
    snapshot_ms: float


@dataclass
class WorkspaceSnapshot:
    step: int
    token_id: int
    token_text: str
    token_logprob: float
    lens_logits: dict[int, torch.Tensor]
    risk: float
    features: dict[str, Any]
    galaxy_rows: list[dict[int, float]]
    # Same snapshot scored with output features truncated to tokens 0..step.
    # Honest about its inputs, but uncalibrated -- see the note in the scoring loop.
    online_risk: float = 0.0


def replace_block_hidden(output: Any, hidden: torch.Tensor) -> Any:
    """Replace a residual block's hidden tensor without dropping side outputs."""
    if torch.is_tensor(output):
        return hidden
    if isinstance(output, tuple):
        return (hidden, *output[1:])
    if isinstance(output, list):
        return [hidden, *output[1:]]
    raise TypeError(f"unsupported residual block output type: {type(output).__name__}")


class RollingRouter:
    def __init__(
        self,
        feature_names: list[str],
        weights: list[float],
        bias: float,
        norm_stats: dict[str, list[float]] | None = None,
    ):
        self.feature_names = feature_names
        self.weights = np.array(weights, dtype=np.float64)
        self.bias = float(bias)
        self.norm_stats = norm_stats
        self.history: deque[np.ndarray] = deque(maxlen=200)

    FROZEN = None  # {name: [mean, std]} loaded from sidecar/norm_stats.json

    def score(self, raw_features: dict[str, float], *, record: bool = True) -> float:
        x = np.array(
            [float(raw_features.get(name, 0.0)) for name in self.feature_names],
            dtype=np.float64,
        )
        frozen = self.norm_stats or RollingRouter.FROZEN
        if frozen:
            mu = np.array([frozen.get(n, [0.0, 1.0])[0] for n in self.feature_names])
            sd = np.array([frozen.get(n, [0.0, 1.0])[1] for n in self.feature_names])
            z = (x - mu) / np.where(sd < 1e-6, 1.0, sd)
            if record:
                self.history.append(x)
            logit = float(z @ self.weights + self.bias)
            if logit >= 0:
                return float(1.0 / (1.0 + math.exp(-logit)))
            exp_logit = math.exp(logit)
            return float(exp_logit / (1.0 + exp_logit))
        samples = np.vstack([*self.history, x]) if self.history else x[None, :]
        if len(samples) < 2:
            z = np.zeros_like(x)
        else:
            mu = samples.mean(axis=0)
            sd = samples.std(axis=0)
            sd = np.where(sd < 1e-6, 1.0, sd)
            z = (x - mu) / sd
        if record:
            self.history.append(x)
        logit = float(z @ self.weights + self.bias)
        if logit >= 0:
            return float(1.0 / (1.0 + math.exp(-logit)))
        exp_logit = math.exp(logit)
        return float(exp_logit / (1.0 + exp_logit))


class Runtime:
    def __init__(
        self,
        cfg: Config,
        model_id: str,
        quant: str,
        tokenizer: Any,
        model: Any,
        lens: Any,
        band: list[int],
        router: RollingRouter,
        fallback_reason: str | None,
    ) -> None:
        self.cfg = cfg
        self.model_id = model_id
        self.quant = quant
        self.tokenizer = tokenizer
        self.model = model
        self.lens = lens
        self.band = band
        self.router = router
        self.fallback_reason = fallback_reason
        self.stop_ids = stop_ids(tokenizer)
        self.hedge_ids = hedge_ids(tokenizer)

    def answer(self, body: dict[str, Any]) -> dict[str, Any]:
        if body.get("stream"):
            raise HTTPException(status_code=400, detail="stream=true is not supported")
        messages = body.get("messages")
        if not isinstance(messages, list) or not messages:
            raise HTTPException(status_code=400, detail="messages must be a non-empty list")
        mode = str(body.get("mode") or "detect").lower()
        if mode not in {"detect", "escalate", "refuse", "tag"}:
            mode = "detect"

        max_new = int(
            body.get("max_tokens")
            or body.get("max_completion_tokens")
            or self.cfg.max_new_tokens
        )
        max_new = max(1, min(max_new, 512))
        prompt = build_prompt(self.tokenizer, messages)
        local = self.local_completion(prompt, max_new)
        over_threshold = local.risk > self.cfg.risk_threshold
        should_escalate = (
            mode == "escalate" and over_threshold and bool(self.cfg.escalate_url)
        )

        jspace = {
            "noise": round(local.risk, 6),
            "escalated": False,
            "answered_by": self.model_id,
            "snapshot_ms": round(local.snapshot_ms, 3),
            "features": local.features,
            "layer_entropies": local.layer_entropies,
            "band_tokens": local.band_tokens,
            "workspace_grid": local.workspace_grid,
            "galaxy_trace": local.galaxy_trace,
            "threshold": self.cfg.risk_threshold,
            "action": "local",
        }
        if self.fallback_reason:
            jspace["load_fallback"] = self.fallback_reason
        if body.get("jspace_return_local"):
            jspace["local_answer"] = local.answer
            jspace["local_model"] = self.model_id

        if mode == "refuse" and over_threshold:
            jspace["action"] = "refused"
            return openai_response(
                model=self.model_id,
                answer="I am not confident enough to answer that one.",
                prompt_tokens=local.prompt_tokens,
                completion_tokens=0,
                finish_reason="stop",
                jspace=jspace,
            )

        if mode == "tag" and over_threshold:
            jspace["action"] = "tagged"

        if mode == "detect":
            jspace["action"] = "flagged" if over_threshold else "clean"

        if should_escalate:
            upstream = self.try_escalate(body, jspace)
            if upstream is not None:
                return upstream

        return openai_response(
            model=self.model_id,
            answer=local.answer,
            prompt_tokens=local.prompt_tokens,
            completion_tokens=len(local.gen_ids),
            finish_reason=local.finish_reason,
            jspace=jspace,
        )

    def local_completion(self, prompt: str, max_new: int) -> LocalAnswer:
        input_ids = self.model.encode(prompt, max_length=self.cfg.max_prompt_tokens)
        ids = input_ids
        gen_ids: list[int] = []
        step_logprobs: list[float] = []
        snapshots: list[WorkspaceSnapshot] = []
        snapshot_ms = 0.0
        finish_reason = "length"
        read_tokens = max(1, int(self.cfg.workspace_read_tokens))

        for step in range(max_new):
            lens_logits: dict[int, torch.Tensor] | None = None
            if step < read_tokens:
                snap_start = time.perf_counter()
                lens_logits, model_logits = self.lens_snapshot_from_ids(ids)
                snapshot_ms += (time.perf_counter() - snap_start) * 1000.0
                logits = model_logits
            else:
                logits = self.next_logits(ids)
            logprobs = logits.float().log_softmax(-1)
            nxt = int(logits.argmax(dim=-1).item())
            if nxt in self.stop_ids:
                finish_reason = "stop"
                break
            gen_ids.append(nxt)
            token_logprob = float(logprobs[0, nxt].item())
            step_logprobs.append(token_logprob)
            if lens_logits is not None:
                snapshots.append(
                    WorkspaceSnapshot(
                        step=step,
                        token_id=nxt,
                        token_text=sanitize_band_token(self.tokenizer.decode([nxt])),
                        token_logprob=token_logprob,
                        lens_logits=lens_logits,
                        risk=0.0,
                        features={},
                        galaxy_rows=[],
                    )
                )
            token = torch.tensor([[nxt]], device=ids.device, dtype=ids.dtype)
            ids = torch.cat([ids, token], dim=1)
        else:
            finish_reason = "length"

        answer = strip_gemma_spillover(
            self.tokenizer.decode(gen_ids, skip_special_tokens=True),
            self.model_id,
        )
        if gen_ids:
            first_answer_id = gen_ids[0]
        else:
            first_answer_id = snapshots[0].token_id if snapshots else 0
        if not snapshots:
            snap_start = time.perf_counter()
            lens_logits, model_logits = self.lens_snapshot_from_ids(input_ids)
            snapshot_ms += (time.perf_counter() - snap_start) * 1000.0
            fallback_token = int(model_logits.argmax(dim=-1).item())
            snapshots.append(
                WorkspaceSnapshot(
                    step=0,
                    token_id=fallback_token,
                    token_text=sanitize_band_token(self.tokenizer.decode([fallback_token])),
                    token_logprob=0.0,
                    lens_logits=lens_logits,
                    risk=0.0,
                    features={},
                    galaxy_rows=[],
                )
            )

        # NOTE ON LEAKAGE. This loop runs AFTER generation completes, and the
        # bundled router consumes bl_mean_logprob and bl_answer_len. Passing the
        # full step_logprobs and the final len(gen_ids) to a snapshot taken at
        # step 0 hands that snapshot information from tokens it had not emitted
        # yet, so the resulting curve is RETROSPECTIVE, not an early warning.
        # It stays available because scoring a completed answer is what the
        # router was trained for, but it is now labelled, and an online-only
        # curve is computed beside it.
        #
        # The online curve truncates every output feature to tokens 0..step. It
        # is honest about its inputs but the router is still trained on
        # whole-answer aggregates, so these scores are out-of-distribution and
        # are NOT calibrated. A properly calibrated prefix router needs features
        # the first campaign never saved; that is what Gate C collects.
        for snap in snapshots:
            snap.features, snap.galaxy_rows = self.features_from_snapshot(
                lens_logits=snap.lens_logits,
                answer_token_id=snap.token_id,
                first_answer_logprob=step_logprobs[0] if step_logprobs else 0.0,
                step_logprobs=step_logprobs,
                answer_len=len(gen_ids),
                read_step=snap.step,
                read_token=snap.token_text,
            )
            snap.risk = self.router.score(
                {k: v for k, v in snap.features.items() if isinstance(v, float)},
                record=False,
            )
            window = step_logprobs[: snap.step + 1] or step_logprobs[:1]
            online_features, _ = self.features_from_snapshot(
                lens_logits=snap.lens_logits,
                answer_token_id=snap.token_id,
                first_answer_logprob=step_logprobs[0] if step_logprobs else 0.0,
                step_logprobs=window,
                answer_len=len(window),
                read_step=snap.step,
                read_token=snap.token_text,
            )
            snap.online_risk = self.router.score(
                {k: v for k, v in online_features.items() if isinstance(v, float)},
                record=False,
            )
        selected = max(snapshots, key=lambda snap: snap.risk)
        risk = self.router.score(
            {k: v for k, v in selected.features.items() if isinstance(v, float)}
        )
        features = dict(selected.features)
        features["ws_selected_risk"] = float(risk)
        features["ws_read_tokens"] = float(len(snapshots))
        features["ws_configured_read_tokens"] = float(read_tokens)
        features["ws_per_token_risk_is_retrospective"] = True
        features["ws_per_token_risk_online_uncalibrated"] = [
            {
                "step": int(snap.step),
                "token": snap.token_text,
                "risk": round(float(snap.online_risk), 6),
            }
            for snap in snapshots
        ]
        features["ws_per_token_risk"] = [
            {
                "step": int(snap.step),
                "token": snap.token_text,
                "risk": round(float(snap.risk), 6),
            }
            for snap in snapshots
        ]
        layer_entropies = features.get("layer_entropies", [])
        if not isinstance(layer_entropies, list):
            layer_entropies = []
        return LocalAnswer(
            answer=answer,
            gen_ids=gen_ids,
            prompt_tokens=int(input_ids.shape[1]),
            finish_reason=finish_reason,
            features=features,
            layer_entropies=[float(v) for v in layer_entropies],
            band_tokens=self.band_tokens_from_snapshot(selected.lens_logits),
            workspace_grid=self.workspace_grid_from_snapshot(
                selected.lens_logits,
                selected.token_id,
            ),
            galaxy_trace=self.galaxy_trace_from_snapshots(
                snapshots,
                gen_ids,
                selected.step,
            ),
            risk=risk,
            snapshot_ms=snapshot_ms,
        )

    @torch.no_grad()
    def lens_snapshot_from_ids(
        self, input_ids: torch.Tensor
    ) -> tuple[dict[int, torch.Tensor], torch.Tensor]:
        final_layer = self.model.n_layers - 1
        record_at = sorted({*self.band, final_layer})
        with ActivationRecorder(self.model.layers, at=record_at) as recorder:
            self.model.forward(input_ids)
            activations = {i: recorder.activations[i].detach() for i in record_at}

        def select(layer: int) -> torch.Tensor:
            return activations[layer][0, -1:].float()

        lens_logits: dict[int, torch.Tensor] = {}
        for layer in self.band:
            residual = self.lens.transport(select(layer), layer)
            lens_logits[layer] = self.model.unembed(residual).float().cpu()

        model_logits = self.model.unembed(select(final_layer)).float().cpu()
        return lens_logits, model_logits

    def intervention_direction(
        self,
        input_ids: torch.Tensor,
        layer: int,
        token_id: int,
    ) -> tuple[torch.Tensor, float]:
        """Gradient direction that increases one J-Lens token at one layer."""
        if layer not in self.band:
            raise HTTPException(status_code=400, detail=f"layer {layer} is outside the fitted band")
        with torch.no_grad():
            with ActivationRecorder(self.model.layers, at=[layer]) as recorder:
                self.model.forward(input_ids)
            source = recorder.activations[layer][0, -1:].detach().float()
        source.requires_grad_(True)
        with torch.enable_grad():
            transported = self.lens.transport(source, layer)
            logits = self.model.unembed(transported).float()
            if not 0 <= token_id < logits.shape[-1]:
                raise HTTPException(status_code=400, detail="concept token id is outside the vocabulary")
            log_probability = logits.log_softmax(-1)[0, token_id]
            gradient = torch.autograd.grad(log_probability, source)[0]
        norm = float(gradient.norm().item())
        if not math.isfinite(norm) or norm <= 1e-12:
            raise HTTPException(status_code=422, detail="selected concept has no usable local direction")
        probability = float(log_probability.detach().exp().item())
        return gradient.detach(), probability

    @torch.no_grad()
    def patched_snapshot_from_ids(
        self,
        input_ids: torch.Tensor,
        *,
        layer: int,
        direction: torch.Tensor,
        signed_strength: float,
    ) -> tuple[dict[int, torch.Tensor], torch.Tensor, dict[str, float]]:
        applied: dict[str, float] = {}

        def patch(_module: Any, _inputs: Any, output: Any) -> Any:
            hidden = output if torch.is_tensor(output) else output[0]
            unit = direction.to(device=hidden.device, dtype=torch.float32)
            unit = unit / unit.norm().clamp_min(1e-12)
            base = hidden[:, -1:].float()
            base_norm = base.norm().clamp_min(1e-12)
            delta = unit * (float(signed_strength) * 0.05 * base_norm)
            changed = hidden.clone()
            changed[:, -1:] = (base + delta).to(dtype=hidden.dtype)
            applied["base_norm"] = float(base_norm.item())
            applied["delta_norm"] = float(delta.norm().item())
            applied["relative_norm"] = float(delta.norm().item() / base_norm.item())
            return replace_block_hidden(output, changed)

        handle = self.model.layers[layer].register_forward_hook(patch)
        try:
            lens_logits, model_logits = self.lens_snapshot_from_ids(input_ids)
        finally:
            handle.remove()
        return lens_logits, model_logits, applied

    @torch.no_grad()
    def greedy_from_ids(
        self,
        start_ids: torch.Tensor,
        first_logits: torch.Tensor,
        max_new: int,
    ) -> dict[str, Any]:
        ids = start_ids.clone()
        generated: list[int] = []
        finish_reason = "length"
        logits = first_logits
        for step in range(max_new):
            if step:
                logits = self.next_logits(ids)
            nxt = int(logits.argmax(dim=-1).item())
            if nxt in self.stop_ids:
                finish_reason = "stop"
                break
            generated.append(nxt)
            token = torch.tensor([[nxt]], device=ids.device, dtype=ids.dtype)
            ids = torch.cat([ids, token], dim=1)
        return {
            "text": strip_gemma_spillover(
                self.tokenizer.decode(generated, skip_special_tokens=True), self.model_id
            ),
            "token_ids": generated,
            "finish_reason": finish_reason,
        }

    def branch_experiment(self, body: dict[str, Any]) -> dict[str, Any]:
        messages = body.get("messages")
        if not isinstance(messages, list) or not messages:
            raise HTTPException(status_code=400, detail="messages must be a non-empty list")
        prompt = build_prompt(self.tokenizer, messages)
        input_ids = self.model.encode(prompt, max_length=self.cfg.max_prompt_tokens)
        prefix_ids = body.get("prefix_token_ids") or []
        if not isinstance(prefix_ids, list) or len(prefix_ids) > 128:
            raise HTTPException(status_code=400, detail="prefix_token_ids must be a list of at most 128 ids")
        if prefix_ids:
            prefix = torch.tensor(
                [[int(value) for value in prefix_ids]],
                device=input_ids.device,
                dtype=input_ids.dtype,
            )
            input_ids = torch.cat([input_ids, prefix], dim=1)

        layer = int(body.get("layer", self.band[len(self.band) // 2]))
        token_id = int(body.get("concept_token_id", -1))
        mode = str(body.get("intervention") or "boost").lower()
        if mode not in {"boost", "suppress"}:
            raise HTTPException(status_code=400, detail="intervention must be boost or suppress")
        strength = max(0.0, min(float(body.get("strength", 1.0)), 4.0))
        signed_strength = strength if mode == "boost" else -strength
        max_new = max(1, min(int(body.get("max_tokens", 24)), 96))

        direction, before_direction_probability = self.intervention_direction(
            input_ids, layer, token_id
        )
        baseline_lens, baseline_logits = self.lens_snapshot_from_ids(input_ids)
        patched_lens, patched_logits, patch_stats = self.patched_snapshot_from_ids(
            input_ids,
            layer=layer,
            direction=direction,
            signed_strength=signed_strength,
        )
        baseline = self.greedy_from_ids(input_ids, baseline_logits, max_new)
        intervention = self.greedy_from_ids(input_ids, patched_logits, max_new)

        baseline_probs = baseline_logits[0].softmax(-1)
        patched_probs = patched_logits[0].softmax(-1)
        delta = patched_probs - baseline_probs
        effect_ids = delta.abs().topk(min(12, delta.numel())).indices.tolist()
        effects = [
            {
                "token_id": int(effect_id),
                "token": sanitize_band_token(self.tokenizer.decode([int(effect_id)])),
                "baseline_probability": round(float(baseline_probs[effect_id].item()), 7),
                "intervention_probability": round(float(patched_probs[effect_id].item()), 7),
                "delta": round(float(delta[effect_id].item()), 7),
            }
            for effect_id in effect_ids
        ]
        before = float(baseline_lens[layer][0].softmax(-1)[token_id].item())
        after = float(patched_lens[layer][0].softmax(-1)[token_id].item())
        return {
            "schema_version": 1,
            "experiment": "causal_branch",
            "model": self.model_id,
            "quant": self.quant,
            "layer": layer,
            "concept_token_id": token_id,
            "concept": sanitize_band_token(self.tokenizer.decode([token_id])),
            "intervention": mode,
            "strength": strength,
            "prefix_token_ids": [int(value) for value in prefix_ids],
            "baseline": baseline,
            "branch": intervention,
            "concept_probability": {
                "direction_probe": round(before_direction_probability, 7),
                "before": round(before, 7),
                "after": round(after, 7),
                "delta": round(after - before, 7),
            },
            "patch": patch_stats,
            "effects": effects,
            "evidence": {
                "causal": True,
                "scope": "single residual-stream intervention at the fork point",
                "direction": "local gradient of selected J-Lens token log-probability",
            },
        }

    def overload_experiment(self, body: dict[str, Any]) -> dict[str, Any]:
        raw_levels = body.get("levels") or [1, 2, 4, 6, 8, 10]
        if not isinstance(raw_levels, list):
            raise HTTPException(status_code=400, detail="levels must be a list")
        levels = sorted({max(1, min(int(value), 16)) for value in raw_levels})[:10]
        max_new = max(1, min(int(body.get("max_tokens", 6)), 16))
        rows: list[dict[str, Any]] = []
        started = time.perf_counter()
        for count in levels:
            raw_prompt, target, expected = binding_prompt(count)
            prompt = build_prompt(self.tokenizer, [{"role": "user", "content": raw_prompt}])
            row_started = time.perf_counter()
            local = self.local_completion(prompt, max_new)
            first_token = (
                sanitize_band_token(self.tokenizer.decode([local.gen_ids[0]]))
                if local.gen_ids
                else ""
            )
            rows.append(
                {
                    "bindings": count,
                    "target": target,
                    "expected": expected,
                    "answer": local.answer,
                    "first_token": first_token,
                    "correct": answer_matches_expected(local.answer, expected),
                    "risk": round(float(local.risk), 6),
                    "mean_entropy": round(float(local.features.get("ws_mean_entropy", 0.0)), 6),
                    "token_tail_mass": round(float(local.features.get("ws_mean_tail_mass", 0.0)), 6),
                    "rival_mass": round(float(local.features.get("ws_mean_rival_mass", 0.0)), 6),
                    "family_entropy": round(float(local.features.get("ws_mean_family_entropy", 0.0)), 6),
                    "ignition_depth": round(float(local.features.get("ws_ignition_depth", 1.0)), 6),
                    "latency_ms": round((time.perf_counter() - row_started) * 1000.0, 2),
                }
            )
        return {
            "schema_version": 1,
            "experiment": "binding_overload",
            "model": self.model_id,
            "quant": self.quant,
            "rows": rows,
            "elapsed_ms": round((time.perf_counter() - started) * 1000.0, 2),
            "metric_note": "family_entropy subtracts only the entropy removed by deterministic merges among the top 64 token surfaces; the full-vocabulary tail remains intact",
        }

    @torch.no_grad()
    def conformance_capture(self, body: dict[str, Any]) -> dict[str, Any]:
        messages = body.get("messages")
        if not isinstance(messages, list) or not messages:
            raise HTTPException(status_code=400, detail="messages must be a non-empty list")
        prompt = build_prompt(self.tokenizer, messages)
        input_ids = self.model.encode(prompt, max_length=self.cfg.max_prompt_tokens)
        final_layer = self.model.n_layers - 1
        record_at = sorted({*self.band, final_layer})
        with ActivationRecorder(self.model.layers, at=record_at) as recorder:
            self.model.forward(input_ids)
        activations = {layer: recorder.activations[layer].detach() for layer in record_at}

        rows: list[dict[str, Any]] = []
        for layer in self.band:
            residual = activations[layer][0, -1].float()
            lens_logits = self.model.unembed(
                self.lens.transport(residual[None, :], layer)
            ).float().cpu()[0]
            probs = lens_logits.softmax(-1)
            top_probs, top_ids = probs.topk(min(32, probs.numel()))
            rows.append(
                {
                    "layer": int(layer),
                    "residual": [round(float(value), 7) for value in residual.cpu().tolist()],
                    "residual_norm": round(float(residual.norm().item()), 7),
                    "top_ids": [int(value) for value in top_ids.tolist()],
                    "top_probs": [round(float(value), 8) for value in top_probs.tolist()],
                }
            )

        final_residual = activations[final_layer][0, -1:].float()
        final_logits = self.model.unembed(final_residual).float().cpu()[0]
        final_probs = final_logits.softmax(-1)
        final_top_probs, final_top_ids = final_probs.topk(min(32, final_probs.numel()))
        next_token_id = int(final_top_ids[0].item())
        return {
            "schema_version": 1,
            "capture_type": "jspace_conformance",
            "backend": "transformers",
            "model": self.model_id,
            "quant": self.quant,
            "prompt_tokens": int(input_ids.shape[1]),
            "input_token_ids": [int(value) for value in input_ids[0].tolist()],
            "residual_point": "post-block, final prompt position",
            "layers": rows,
            "final": {
                "next_token_id": next_token_id,
                "next_token": sanitize_band_token(self.tokenizer.decode([next_token_id])),
                "top_ids": [int(value) for value in final_top_ids.tolist()],
                "top_probs": [round(float(value), 8) for value in final_top_probs.tolist()],
            },
        }

    @torch.no_grad()
    def next_logits(self, ids: torch.Tensor) -> torch.Tensor:
        hidden = self.model.forward(ids).last_hidden_state[:, -1]
        head = self.model._lm_head
        logits = head(hidden.to(head.weight.dtype).to(head.weight.device))
        softcap = getattr(self.model, "_logit_softcap", None)
        if softcap is not None:
            logits = softcap * torch.tanh(logits / softcap)
        return logits

    def features_from_snapshot(
        self,
        lens_logits: dict[int, torch.Tensor],
        answer_token_id: int,
        first_answer_logprob: float,
        step_logprobs: list[float],
        answer_len: int,
        read_step: int,
        read_token: str,
    ) -> tuple[dict[str, Any], list[dict[int, float]]]:
        ranks_ans: list[int] = []
        ranks_hedge: list[int] = []
        entropies: list[float] = []
        top1s: list[int] = []
        rival_masses: list[float] = []
        tail_masses: list[float] = []
        family_entropies: list[float] = []
        galaxy_rows: list[dict[int, float]] = []

        for layer in self.band:
            logits = lens_logits[layer][0].float()
            order = logits.argsort(descending=True)
            rank_of = torch.empty_like(order)
            rank_of[order] = torch.arange(len(order), device=order.device)
            ranks_ans.append(int(rank_of[answer_token_id].item()))
            if self.hedge_ids:
                ranks_hedge.append(int(min(rank_of[t].item() for t in self.hedge_ids)))
            probs = logits.softmax(-1)
            token_entropy = float(-(probs * probs.clamp_min(1e-12).log()).sum().item())
            entropies.append(token_entropy)
            metric_take = min(64, int(probs.numel()))
            metric_probs, metric_ids = torch.topk(probs, metric_take)
            rival_masses.append(float(metric_probs[1:5].sum().item()))
            tail_masses.append(float(max(0.0, 1.0 - metric_probs[:20].sum().item())))
            families = aggregate_top_token_families(
                metric_ids.tolist(),
                metric_probs.tolist(),
                self.tokenizer,
            )
            top_token_entropy = entropy_from_masses(metric_probs.tolist())
            top_family_entropy = entropy_from_masses(families.values())
            family_entropies.append(
                max(0.0, token_entropy - (top_token_entropy - top_family_entropy))
            )
            top1s.append(int(order[0].item()))
            take = min(12, int(probs.numel()))
            top_probs, top_ids = torch.topk(probs, take)
            galaxy_row = {
                int(token_id): float(prob)
                for token_id, prob in zip(top_ids.tolist(), top_probs.tolist())
            }
            galaxy_row[int(answer_token_id)] = float(probs[answer_token_id].item())
            galaxy_rows.append(galaxy_row)

        e = np.array(entropies, dtype=np.float64)
        n = len(e)
        x = np.arange(n, dtype=np.float64)
        slope = float(np.polyfit(x, e, 1)[0]) if n > 1 else 0.0
        ranks = np.array(ranks_ans, dtype=np.float64)
        ignited = np.nonzero(ranks <= 10)[0]
        if not step_logprobs:
            step_logprobs = [0.0]
        hedge_rank = min(ranks_hedge) if ranks_hedge else 0

        features = {
            "bl_first_token_logprob": float(first_answer_logprob),
            "bl_mean_logprob": float(np.mean(step_logprobs)),
            "bl_min_logprob": float(np.min(step_logprobs)),
            "bl_answer_len": float(answer_len),
            "ws_read_step": float(read_step),
            "ws_mean_entropy": float(e.mean()),
            "ws_max_entropy": float(e.max()),
            "ws_late_entropy": float(e[2 * n // 3 :].mean()),
            "ws_entropy_slope": slope,
            "ws_entropy_std": float(e.std()),
            "ws_mean_rival_mass": float(np.mean(rival_masses)),
            "ws_mean_tail_mass": float(np.mean(tail_masses)),
            "ws_mean_family_entropy": float(np.mean(family_entropies)),
            "ws_ignition_frac": float((ranks <= 10).mean()),
            "ws_ignition_depth": float(ignited[0] / n) if len(ignited) else 1.0,
            "ws_mean_log_rank": float(np.log1p(ranks).mean()),
            "ws_band_agreement": float(np.mean(np.array(top1s) == answer_token_id)),
            "ws_hedge_rank": float(np.log1p(hedge_rank)),
            "ws_read_token": read_token,
            "layer_entropies": [round(float(v), 4) for v in entropies],
        }
        return features, galaxy_rows

    def band_tokens_from_snapshot(
        self, lens_logits: dict[int, torch.Tensor]
    ) -> list[dict[str, Any]]:
        sample_layers = [self.band[0], self.band[len(self.band) // 2], self.band[-1]]
        rows: list[dict[str, Any]] = []
        for layer in sample_layers:
            logits = lens_logits[layer][0].float()
            token_ids = logits.argsort(descending=True)[:8].tolist()
            tokens = [
                sanitize_band_token(self.tokenizer.decode([int(token_id)]))
                for token_id in token_ids
            ]
            rows.append({"layer_index": int(layer), "tokens": tokens})
        return rows

    def workspace_grid_from_snapshot(
        self,
        lens_logits: dict[int, torch.Tensor],
        first_answer_id: int,
    ) -> dict[str, Any]:
        rows_by_id: list[dict[int, float]] = []
        best_prob_by_id: dict[int, float] = {}

        for layer in self.band:
            logits = lens_logits[layer][0].float()
            probs = logits.softmax(-1)
            take = min(10, int(probs.numel()))
            top_probs, top_ids = torch.topk(probs, take)
            row: dict[int, float] = {}
            for token_id, prob in zip(top_ids.tolist(), top_probs.tolist()):
                tid = int(token_id)
                p = float(prob)
                row[tid] = p
                best_prob_by_id[tid] = max(best_prob_by_id.get(tid, 0.0), p)
            rows_by_id.append(row)

        column_ids = [
            token_id
            for token_id, _prob in sorted(
                best_prob_by_id.items(),
                key=lambda item: item[1],
                reverse=True,
            )[:24]
        ]
        columns = [
            sanitize_band_token(self.tokenizer.decode([int(token_id)]))
            for token_id in column_ids
        ]
        values = [
            [round(float(row.get(token_id, 0.0)), 6) for token_id in column_ids]
            for row in rows_by_id
        ]
        try:
            answer_col = column_ids.index(int(first_answer_id))
        except ValueError:
            answer_col = -1

        return {
            "layers": [int(layer) for layer in self.band],
            "column_ids": [int(token_id) for token_id in column_ids],
            "columns": columns,
            "values": values,
            "answer_col": answer_col,
        }

    def galaxy_trace_from_snapshots(
        self,
        snapshots: list[WorkspaceSnapshot],
        gen_ids: list[int],
        selected_step: int,
        *,
        top_k: int = 12,
        max_concepts: int = 64,
    ) -> dict[str, Any]:
        """Build a compact, model-native 2D view from existing lens reads.

        This deliberately reuses the logits already produced for routing. The
        layout is a PCA of each candidate token's layer-by-layer J-Lens
        probability trajectory, not a claim about literal residual geometry.
        """
        if not snapshots:
            return {}

        layers = [int(layer) for layer in self.band]
        traced_answer_ids = {int(snap.token_id) for snap in snapshots}
        best_prob: dict[int, float] = {token_id: 0.0 for token_id in traced_answer_ids}
        sparse_rows: list[list[dict[int, float]]] = []

        for snap in snapshots:
            step_rows = snap.galaxy_rows
            if not step_rows:
                # Compatibility for snapshots constructed outside local_completion.
                step_rows = []
                for layer in self.band:
                    probs = snap.lens_logits[layer][0].float().softmax(-1)
                    take = min(top_k, int(probs.numel()))
                    top_probs, top_ids = torch.topk(probs, take)
                    step_rows.append(
                        {
                            int(token_id): float(prob)
                            for token_id, prob in zip(top_ids.tolist(), top_probs.tolist())
                        }
                    )
            for row in step_rows:
                for token_id, prob in row.items():
                    best_prob[token_id] = max(best_prob.get(token_id, 0.0), prob)
            sparse_rows.append(step_rows)

        concept_ids = [
            token_id
            for token_id, _prob in sorted(
                best_prob.items(), key=lambda item: (-item[1], item[0])
            )[:max_concepts]
        ]
        # A traced answer is always inspectable, even if it never reaches top-k.
        for token_id in sorted(traced_answer_ids):
            if token_id not in concept_ids:
                if len(concept_ids) >= max_concepts:
                    concept_ids[-1] = token_id
                else:
                    concept_ids.append(token_id)
        concept_ids = list(dict.fromkeys(concept_ids))

        score_vectors: list[list[float]] = []
        score_matrices: dict[int, list[list[float | None]]] = {}
        for token_id in concept_ids:
            matrix: list[list[float | None]] = []
            vector: list[float] = []
            for step_rows in sparse_rows:
                row_values: list[float | None] = []
                for row in step_rows:
                    value = row.get(token_id)
                    rounded = round(float(value), 7) if value is not None else None
                    row_values.append(rounded)
                    vector.append(float(value) if value is not None else 0.0)
                matrix.append(row_values)
            score_matrices[token_id] = matrix
            score_vectors.append(vector)

        nodes, edges = trajectory_layout(
            concept_ids,
            np.asarray(score_vectors, dtype=np.float64),
            neighbors=2,
        )
        node_by_id = {int(node["id"]): node for node in nodes}
        concepts: list[dict[str, Any]] = []
        for token_id in concept_ids:
            matrix = score_matrices[token_id]
            observed = [value for row in matrix for value in row if value is not None]
            label = sanitize_band_token(self.tokenizer.decode([int(token_id)]))
            concepts.append(
                {
                    **node_by_id[token_id],
                    "token": label,
                    "scores": matrix,
                    "peak": round(max(observed, default=0.0), 7),
                    "answer_steps": [
                        int(snap.step) for snap in snapshots if snap.token_id == token_id
                    ],
                }
            )

        steps = []
        for snap in snapshots:
            entropies = snap.features.get("layer_entropies", [])
            if not isinstance(entropies, list):
                entropies = []
            steps.append(
                {
                    "step": int(snap.step),
                    "token_id": int(snap.token_id),
                    "token": snap.token_text,
                    "logprob": round(float(snap.token_logprob), 6),
                    "risk": round(float(snap.risk), 6),
                    "entropies": [round(float(value), 4) for value in entropies],
                }
            )

        completion = [
            {
                "step": index,
                "token_id": int(token_id),
                "token": sanitize_band_token(self.tokenizer.decode([int(token_id)])),
                "traced": index < len(snapshots),
            }
            for index, token_id in enumerate(gen_ids)
        ]
        sector_labels: dict[int, list[str]] = {}
        for concept in sorted(concepts, key=lambda item: item["peak"], reverse=True):
            labels = sector_labels.setdefault(int(concept["sector"]), [])
            if concept["token"] and concept["token"] not in labels and len(labels) < 2:
                labels.append(concept["token"])

        return {
            "schema_version": 1,
            "model": self.model_id,
            "layers": layers,
            "selected_step": int(selected_step),
            "steps": steps,
            "completion": completion,
            "concepts": concepts,
            "edges": edges,
            "sector_labels": {str(key): value for key, value in sector_labels.items()},
            "evidence": {
                "activity": "J-Lens top-k probability",
                "layout": "PCA of J-Lens trajectories in this trace",
                "edges": "nearest neighbors in the 2D projection",
                "causal": False,
            },
        }

    def escalate_one(self, body: dict[str, Any]) -> dict[str, Any]:
        messages = body.get("messages")
        if not isinstance(messages, list) or not messages:
            raise HTTPException(status_code=400, detail="messages must be a non-empty list")
        if not self.cfg.escalate_url:
            raise HTTPException(status_code=503, detail="no upstream escalation URL configured")

        jspace = {
            "escalated": True,
            "answered_by": self.cfg.escalate_model or "upstream",
            "action": "escalated",
            "forced": True,
        }
        upstream = self.try_escalate(body, jspace)
        if upstream is None:
            detail = str(jspace.get("escalation_error") or "upstream escalation failed")
            raise HTTPException(status_code=502, detail=detail)
        return upstream

    def try_escalate(
        self, body: dict[str, Any], jspace: dict[str, Any]
    ) -> dict[str, Any] | None:
        payload = {
            k: v
            for k, v in body.items()
            if k not in {"stream", "jspace_return_local", "mode"}
        }
        payload["stream"] = False
        payload["messages"] = add_no_think_prefix(payload.get("messages"))
        if self.cfg.escalate_model:
            payload["model"] = self.cfg.escalate_model
        payload["max_tokens"] = max(1024, int(payload.get("max_tokens") or 0))
        headers = {"Content-Type": "application/json"}
        key = os.environ.get(self.cfg.escalate_key_env, "") if self.cfg.escalate_key_env else ""
        if key and not self.cfg.escalate_url.startswith("http://localhost"):
            headers["Authorization"] = "Bearer " + key
            payload["reasoning"] = {"enabled": False}
        data = json.dumps(payload).encode("utf-8")
        req = urllib.request.Request(
            self.cfg.escalate_url,
            data=data,
            headers=headers,
            method="POST",
        )
        try:
            with urllib.request.urlopen(req, timeout=self.cfg.escalate_timeout_s) as resp:
                upstream = json.loads(resp.read().decode("utf-8"))
        except (OSError, urllib.error.URLError, json.JSONDecodeError) as exc:
            jspace["escalation_error"] = str(exc)
            return None

        answered_by = str(upstream.get("model") or "escalated")
        strip_escalation_response(upstream)
        for ch in upstream.get("choices") or []:
            msg = ch.get("message") or {}
            if msg.get("content") is None:
                msg["content"] = ""
        jspace["escalated"] = True
        jspace["answered_by"] = answered_by
        jspace["action"] = "escalated"
        upstream["jspace"] = jspace
        upstream.setdefault("model", answered_by)
        return upstream


class RuntimeHolder:
    def __init__(self) -> None:
        self.runtime: Runtime | None = None
        self.lock = Lock()


STATE = RuntimeHolder()
app = FastAPI(title="jspace sidecar")


def model_slug(model_id: str) -> str:
    return model_id.split("/")[-1].lower()


def resolve_path(value: str | Path) -> Path:
    path = Path(value)
    return path if path.is_absolute() else ROOT / path


def load_config() -> Config:
    defaults: dict[str, Any] = {
        "MODEL_ID": "google/gemma-4-E4B-it",
        "MODEL_PATH": "",
        "QUANT": "4bit",
        "LENS_PATH": "",
        "ROUTER_PATH": "data/workspace_routers_all5.json",
        "ESCALATE_URL": "http://localhost:8080/v1/chat/completions",
        "ESCALATE_MODEL": "",
        "ESCALATE_KEY_ENV": "OPENROUTER_API_KEY",
        "RISK_THRESHOLD": 0.6,
        "MAX_PROMPT_TOKENS": 1536,
        "MAX_NEW_TOKENS": 64,
        "ESCALATE_TIMEOUT_S": 600,
        "LENS_DEVICE": "auto",
        "AUTO_FALLBACK": "1",
        "WORKSPACE_READ_TOKENS": 3,
    }
    cfg_path = SIDECAR / "config.json"
    if cfg_path.exists():
        with cfg_path.open(encoding="utf-8") as f:
            defaults.update(json.load(f))
    for key in list(defaults):
        if key in os.environ:
            defaults[key] = os.environ[key]

    model_id = str(defaults["MODEL_ID"])
    lens_path_explicit = bool(str(defaults.get("LENS_PATH") or "").strip())
    lens_value = defaults["LENS_PATH"] or f"out/{model_slug(model_id)}/lens.pt"
    return Config(
        model_id=model_id,
        model_path=str(defaults.get("MODEL_PATH") or "").strip(),
        quant=str(defaults["QUANT"]).lower(),
        lens_path=resolve_path(lens_value),
        lens_path_explicit=lens_path_explicit,
        router_path=resolve_path(defaults["ROUTER_PATH"]),
        escalate_url=str(defaults["ESCALATE_URL"]),
        escalate_model=str(defaults.get("ESCALATE_MODEL") or ""),
        escalate_key_env=str(defaults.get("ESCALATE_KEY_ENV") or ""),
        risk_threshold=float(defaults["RISK_THRESHOLD"]),
        max_prompt_tokens=int(defaults["MAX_PROMPT_TOKENS"]),
        max_new_tokens=int(defaults["MAX_NEW_TOKENS"]),
        escalate_timeout_s=float(defaults["ESCALATE_TIMEOUT_S"]),
        lens_device=str(defaults["LENS_DEVICE"]).lower(),
        auto_fallback=str(defaults["AUTO_FALLBACK"]).lower() not in {"0", "false", "no"},
        workspace_read_tokens=max(1, int(defaults["WORKSPACE_READ_TOKENS"])),
    )


def load_runtime() -> Runtime:
    cfg = load_config()
    _ns = SIDECAR / "norm_stats.json"
    if _ns.exists():
        with _ns.open(encoding="utf-8") as f:
            RollingRouter.FROZEN = json.load(f)
    hf_model, runtime_model_id, runtime_quant, fallback_reason = load_hf_with_fallback(cfg)
    tokenizer_src = runtime_model_id
    if cfg.model_path and (Path(cfg.model_path) / "tokenizer_config.json").exists():
        tokenizer_src = cfg.model_path
    tokenizer = transformers.AutoTokenizer.from_pretrained(tokenizer_src)
    model = jlens.from_hf(hf_model, tokenizer)

    lens_path = cfg.lens_path
    if runtime_model_id != cfg.model_id and not cfg.lens_path_explicit:
        lens_path = ROOT / "out" / model_slug(runtime_model_id) / "lens.pt"
    if not Path(lens_path).exists():
        # fetch the fitted lens from the public HF repo on first run
        slug = model_slug(runtime_model_id)
        print(f"[sidecar] lens not found at {lens_path}, downloading {slug}/lens.pt from {LENS_HUB_REPO} ...")
        from huggingface_hub import hf_hub_download
        lens_path = hf_hub_download(LENS_HUB_REPO, f"{slug}/lens.pt")
    lens = jlens.JacobianLens.load(str(lens_path))
    band = [
        layer
        for layer in range(int(model.n_layers * BAND_LO), int(model.n_layers * BAND_HI))
        if layer in lens.jacobians
    ]
    if not band:
        raise RuntimeError(f"no fitted lens layers found in band for {lens_path}")
    lens = prepare_lens(lens, band, cfg, runtime_quant)
    router = load_router(cfg.router_path, runtime_model_id)
    return Runtime(
        cfg=cfg,
        model_id=runtime_model_id,
        quant=runtime_quant,
        tokenizer=tokenizer,
        model=model,
        lens=lens,
        band=band,
        router=router,
        fallback_reason=fallback_reason,
    )


def load_hf_with_fallback(cfg: Config) -> tuple[Any, str, str, str | None]:
    if cfg.model_path:
        # Pre-quantized local checkpoint (see sidecar/save_nf4.py). Its embedded
        # quantization_config wins, so never pass a fresh BitsAndBytesConfig.
        # Errors propagate: an explicit MODEL_PATH must fail loudly, not fall back.
        if checkpoint_is_quantized(cfg.model_path):
            if not torch.cuda.is_available():
                raise RuntimeError(f"MODEL_PATH={cfg.model_path!r} is a 4bit checkpoint and needs CUDA")
            model = from_pretrained_any(cfg.model_path, device_map="cuda")
            return model, cfg.model_id, "4bit", None
        return load_hf(cfg.model_path, cfg.quant), cfg.model_id, cfg.quant, None
    try:
        return load_hf(cfg.model_id, cfg.quant), cfg.model_id, cfg.quant, None
    except Exception as exc:
        if cfg.quant != "4bit" or not cfg.auto_fallback:
            raise
        fallback_id = "google/gemma-4-E4B-it"
        fallback = load_bf16_with_fit_device_map(fallback_id)
        reason = f"4bit load failed, fell back to {fallback_id} bf16: {exc}"
        return fallback, fallback_id, "bf16", reason


def checkpoint_is_quantized(path: str) -> bool:
    cfg_file = Path(path) / "config.json"
    if not cfg_file.exists():
        raise RuntimeError(f"MODEL_PATH={path!r} has no config.json")
    with cfg_file.open(encoding="utf-8") as f:
        return "quantization_config" in json.load(f)


def load_hf(model_id: str, quant: str) -> Any:
    if quant == "4bit":
        if not torch.cuda.is_available():
            raise RuntimeError("QUANT=4bit needs CUDA")
        if not has_package("bitsandbytes"):
            raise RuntimeError("bitsandbytes is not installed")
        kwargs = {
            "quantization_config": transformers.BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_compute_dtype=torch.bfloat16,
            ),
            "device_map": "cuda",
        }
        return from_pretrained_any(model_id, **kwargs)
    if quant == "bf16":
        return load_bf16_with_fit_device_map(model_id)
    raise ValueError(f"unsupported QUANT={quant!r}")


def load_bf16_with_fit_device_map(model_id: str) -> Any:
    from fit import load_model

    return load_model(model_id)


def from_pretrained_any(model_id: str, **kwargs: Any) -> Any:
    try:
        return transformers.AutoModelForCausalLM.from_pretrained(model_id, **kwargs)
    except ValueError:
        return transformers.AutoModelForImageTextToText.from_pretrained(model_id, **kwargs)


def has_package(name: str) -> bool:
    import importlib.util

    return importlib.util.find_spec(name) is not None


def prepare_lens(lens: Any, band: list[int], cfg: Config, quant: str) -> Any:
    lens.jacobians = {layer: lens.jacobians[layer] for layer in band}
    lens.source_layers = band
    target = choose_lens_device(lens, cfg, quant)
    try:
        if target == "cuda":
            lens.jacobians = {
                layer: tensor.to(device="cuda", dtype=torch.float16)
                for layer, tensor in lens.jacobians.items()
            }
        else:
            lens.jacobians = {
                layer: tensor.to(device="cpu", dtype=torch.float32)
                for layer, tensor in lens.jacobians.items()
            }
    except RuntimeError:
        torch.cuda.empty_cache()
        target = "cpu"
        lens.jacobians = {
            layer: tensor.to(device="cpu", dtype=torch.float32)
            for layer, tensor in lens.jacobians.items()
        }

    lens._jspace_lens_device = target

    def transport(self: Any, residual: torch.Tensor, layer: int) -> torch.Tensor:
        J = self.jacobians[layer]
        if self._jspace_lens_device == "cpu":
            residual = residual.to(device="cpu", dtype=J.dtype)
        else:
            residual = residual.to(device=J.device, dtype=J.dtype)
        return residual @ J.T

    import types

    lens.transport = types.MethodType(transport, lens)
    return lens


def choose_lens_device(lens: Any, cfg: Config, quant: str) -> str:
    if cfg.lens_device in {"cpu", "cuda"}:
        return cfg.lens_device
    if not torch.cuda.is_available() or quant != "4bit":
        return "cpu"
    lens_bytes = sum(t.numel() * 2 for t in lens.jacobians.values())
    allocated = torch.cuda.memory_allocated()
    if allocated + lens_bytes > VRAM_BUDGET_BYTES:
        return "cpu"
    return "cuda"


def load_router(path: Path, model_id: str) -> RollingRouter:
    with path.open(encoding="utf-8") as f:
        data = json.load(f)
    routers = data["routers"]
    slug = model_slug(model_id)
    if slug not in routers:
        raise RuntimeError(f"router weights for {slug!r} not found in {path}")
    combined = routers[slug]["combined"]
    norm_stats = (
        combined.get("norm_stats")
        or routers[slug].get("norm_stats")
        or (data.get("norm_stats") or {}).get(slug)
    )
    return RollingRouter(
        feature_names=list(combined["features"]),
        weights=list(combined["weights"]),
        bias=float(combined["bias"]),
        norm_stats=norm_stats,
    )


def stop_ids(tokenizer: Any) -> set[int]:
    out: set[int] = set()
    eos = getattr(tokenizer, "eos_token_id", None)
    if isinstance(eos, int):
        out.add(eos)
    for token in ("<end_of_turn>", "<|im_end|>"):
        tid = tokenizer.convert_tokens_to_ids(token)
        if isinstance(tid, int) and tid >= 0:
            out.add(tid)
    return out


def hedge_ids(tokenizer: Any) -> list[int]:
    ids: set[int] = set()
    for word in HEDGE_WORDS:
        encoded = tokenizer(word, add_special_tokens=False).input_ids
        if encoded:
            ids.add(int(encoded[0]))
    return sorted(ids)


SPECIAL_TOKEN_RE = re.compile(r"^<\|?([^<>]*?)\|?>$")


def sanitize_band_token(token: str) -> str:
    token = re.sub(r"\s+", " ", str(token)).strip()
    if any(ord(ch) >= 0x0500 for ch in token):
        return "^"
    special = SPECIAL_TOKEN_RE.fullmatch(token)
    if special:
        label = special.group(1).strip().strip("|")
        label = re.sub(r"[^A-Za-z0-9_:-]+", "", label) or "special"
        if len(label) > 18:
            label = label[:17] + "."
        return f"<{label}>"
    return token


# The research signal was validated at answer onset. The sidecar now scans the
# first WORKSPACE_READ_TOKENS answer-token workspaces, but terse answers still
# keep the read aligned with the calibrated setting and avoid paying for filler.
# Set LOCAL_TERSE=0 to disable.
TERSE_SYSTEM = ("Answer directly and concisely in plain text. Lead with the answer itself. "
                "No markdown, no bold, no asterisks, no preamble, no restating the question.")

def build_prompt(tokenizer: Any, messages: list[dict[str, Any]]) -> str:
    clean_messages = [
        {"role": str(msg.get("role", "user")), "content": message_content(msg)}
        for msg in messages
    ]
    if os.environ.get("LOCAL_TERSE", "1") not in {"0", "false", "no"}:
        if not clean_messages or clean_messages[0].get("role") != "system":
            clean_messages = [{"role": "system", "content": TERSE_SYSTEM}] + clean_messages
    try:
        return tokenizer.apply_chat_template(
            clean_messages,
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=False,
        )
    except TypeError:
        try:
            return tokenizer.apply_chat_template(
                clean_messages,
                tokenize=False,
                add_generation_prompt=True,
            )
        except Exception:
            pass
    except Exception:
        pass
    lines = [f"{m['role']}: {m['content']}" for m in clean_messages]
    return "\n".join(lines) + "\nassistant:"


def strip_gemma_spillover(answer: str, model_id: str) -> str:
    answer = answer.strip()
    if "gemma" not in model_id.lower():
        return answer
    return answer.split("thought", 1)[0].strip()


def add_no_think_prefix(messages: Any) -> Any:
    if not isinstance(messages, list):
        return messages
    clean_messages = [dict(msg) if isinstance(msg, dict) else msg for msg in messages]
    for i in range(len(clean_messages) - 1, -1, -1):
        msg = clean_messages[i]
        if not isinstance(msg, dict) or str(msg.get("role", "")).lower() != "user":
            continue
        content = msg.get("content", "")
        if isinstance(content, str):
            msg["content"] = "/no_think " + content
        elif isinstance(content, list):
            msg["content"] = prefix_text_content(content, "/no_think ")
        else:
            msg["content"] = "/no_think " + str(content)
        break
    return clean_messages


def prefix_text_content(content: list[Any], prefix: str) -> list[Any]:
    prefixed = False
    out: list[Any] = []
    for item in content:
        if isinstance(item, dict):
            copied = dict(item)
            if not prefixed and copied.get("type") == "text":
                copied["text"] = prefix + str(copied.get("text", ""))
                prefixed = True
            out.append(copied)
        else:
            out.append(item)
    if not prefixed:
        out.insert(0, {"type": "text", "text": prefix})
    return out


THINK_BLOCK_RE = re.compile(r"<think>.*?</think>", re.IGNORECASE | re.DOTALL)
REASONING_HEADER_RE = re.compile(r"\s*(?:\d+\.|\*\*|Thinking Process\s*:)", re.IGNORECASE)


def strip_escalation_response(upstream: dict[str, Any]) -> None:
    choices = upstream.get("choices")
    if not isinstance(choices, list):
        return
    for choice in choices:
        if not isinstance(choice, dict):
            continue
        message = choice.get("message")
        if isinstance(message, dict) and isinstance(message.get("content"), str):
            message["content"] = strip_escalation_reasoning(message["content"])
        elif isinstance(choice.get("text"), str):
            choice["text"] = strip_escalation_reasoning(choice["text"])


def strip_escalation_reasoning(text: str) -> str:
    text = THINK_BLOCK_RE.sub("", text).strip()
    paragraphs = re.split(r"\n\s*\n", text)
    if len(paragraphs) < 2:
        return text

    prior = "\n\n".join(paragraphs[:-1])
    final = paragraphs[-1].strip()
    final_lines = [line.strip() for line in final.splitlines() if line.strip()]
    has_reasoning_header = any(
        REASONING_HEADER_RE.match(line) for line in prior.splitlines()
    )
    if has_reasoning_header and len(final_lines) == 1 and len(final_lines[0]) <= 240:
        return final_lines[0]
    return text


def message_content(msg: dict[str, Any]) -> str:
    content = msg.get("content", "")
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts = []
        for item in content:
            if isinstance(item, dict) and item.get("type") == "text":
                parts.append(str(item.get("text", "")))
        return "\n".join(part for part in parts if part)
    return str(content)


def openai_response(
    *,
    model: str,
    answer: str,
    prompt_tokens: int,
    completion_tokens: int,
    finish_reason: str,
    jspace: dict[str, Any],
) -> dict[str, Any]:
    return {
        "id": f"chatcmpl-{uuid.uuid4().hex}",
        "object": "chat.completion",
        "created": int(time.time()),
        "model": model,
        "choices": [
            {
                "index": 0,
                "message": {"role": "assistant", "content": answer},
                "finish_reason": finish_reason,
            }
        ],
        "usage": {
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens,
            "total_tokens": prompt_tokens + completion_tokens,
        },
        "jspace": jspace,
    }


@app.on_event("startup")
def startup() -> None:
    STATE.runtime = load_runtime()


@app.get("/health")
def health() -> dict[str, Any]:
    runtime = STATE.runtime
    if runtime is None:
        return {"ok": False}
    return {
        "ok": True,
        "model": runtime.model_id,
        "model_path": runtime.cfg.model_path or None,
        "quant": runtime.quant,
        "escalate_model": runtime.cfg.escalate_model or None,
        "threshold": runtime.cfg.risk_threshold,
        "workspace_read_tokens": runtime.cfg.workspace_read_tokens,
        "band": [runtime.band[0], runtime.band[-1]],
        "fallback": runtime.fallback_reason,
    }


@app.get("/chat", response_class=HTMLResponse)
def chat_page() -> HTMLResponse:
    path = SIDECAR / "chat.html"
    if not path.exists():
        raise HTTPException(status_code=404, detail="chat.html not found")
    return HTMLResponse(path.read_text(encoding="utf-8"))


@app.get("/galaxy", response_class=HTMLResponse)
def galaxy_page() -> HTMLResponse:
    path = SIDECAR / "galaxy.html"
    if not path.exists():
        raise HTTPException(status_code=404, detail="galaxy.html not found")
    return HTMLResponse(path.read_text(encoding="utf-8"))


@app.get("/lab", response_class=HTMLResponse)
def lab_page() -> HTMLResponse:
    path = SIDECAR / "lab.html"
    if not path.exists():
        raise HTTPException(status_code=404, detail="lab.html not found")
    return HTMLResponse(path.read_text(encoding="utf-8"))


@app.post("/v1/chat/completions")
def chat_completions(body: dict[str, Any] = Body(...)) -> dict[str, Any]:
    if STATE.runtime is None:
        STATE.runtime = load_runtime()
    with STATE.lock:
        return STATE.runtime.answer(body)


@app.post("/experiments/branch")
def branch_experiment(body: dict[str, Any] = Body(...)) -> dict[str, Any]:
    if STATE.runtime is None:
        STATE.runtime = load_runtime()
    with STATE.lock:
        return STATE.runtime.branch_experiment(body)


@app.post("/experiments/overload")
def overload_experiment(body: dict[str, Any] = Body(...)) -> dict[str, Any]:
    if STATE.runtime is None:
        STATE.runtime = load_runtime()
    with STATE.lock:
        return STATE.runtime.overload_experiment(body)


@app.post("/experiments/conformance/capture")
def conformance_capture(body: dict[str, Any] = Body(...)) -> dict[str, Any]:
    if STATE.runtime is None:
        STATE.runtime = load_runtime()
    with STATE.lock:
        return STATE.runtime.conformance_capture(body)


@app.post("/experiments/conformance/compare")
def conformance_compare(body: dict[str, Any] = Body(...)) -> dict[str, Any]:
    reference = body.get("reference")
    candidate = body.get("candidate")
    if not isinstance(reference, dict) or not isinstance(candidate, dict):
        raise HTTPException(status_code=400, detail="reference and candidate captures are required")
    thresholds = body.get("thresholds")
    if thresholds is not None and not isinstance(thresholds, dict):
        raise HTTPException(status_code=400, detail="thresholds must be an object")
    return compare_captures(reference, candidate, thresholds=thresholds)


@app.post("/escalate_one")
def escalate_one(body: dict[str, Any] = Body(...)) -> dict[str, Any]:
    if STATE.runtime is None:
        with STATE.lock:
            if STATE.runtime is None:
                STATE.runtime = load_runtime()
    return STATE.runtime.escalate_one(body)
