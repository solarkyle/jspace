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


@dataclass
class LocalAnswer:
    answer: str
    gen_ids: list[int]
    prompt_tokens: int
    finish_reason: str
    features: dict[str, float | list[float]]
    layer_entropies: list[float]
    band_tokens: list[dict[str, Any]]
    workspace_grid: dict[str, Any]
    risk: float
    snapshot_ms: float


class RollingRouter:
    def __init__(self, feature_names: list[str], weights: list[float], bias: float):
        self.feature_names = feature_names
        self.weights = np.array(weights, dtype=np.float64)
        self.bias = float(bias)
        self.history: deque[np.ndarray] = deque(maxlen=200)

    FROZEN = None  # {name: [mean, std]} loaded from sidecar/norm_stats.json

    def score(self, raw_features: dict[str, float]) -> float:
        x = np.array(
            [float(raw_features.get(name, 0.0)) for name in self.feature_names],
            dtype=np.float64,
        )
        if RollingRouter.FROZEN:
            mu = np.array([RollingRouter.FROZEN.get(n, [0.0, 1.0])[0] for n in self.feature_names])
            sd = np.array([RollingRouter.FROZEN.get(n, [0.0, 1.0])[1] for n in self.feature_names])
            z = (x - mu) / np.where(sd < 1e-6, 1.0, sd)
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
        start = time.perf_counter()
        lens_logits, first_logits, input_ids = self.lens.apply(
            self.model,
            prompt,
            layers=self.band,
            positions=[-1],
            max_seq_len=self.cfg.max_prompt_tokens,
        )
        snapshot_ms = (time.perf_counter() - start) * 1000.0

        ids = input_ids
        logits = first_logits[0].unsqueeze(0)
        gen_ids: list[int] = []
        step_logprobs: list[float] = []
        finish_reason = "length"

        for step in range(max_new):
            if step > 0:
                logits = self.next_logits(ids)
            logprobs = logits.float().log_softmax(-1)
            nxt = int(logits.argmax(dim=-1).item())
            if nxt in self.stop_ids:
                finish_reason = "stop"
                break
            gen_ids.append(nxt)
            step_logprobs.append(float(logprobs[0, nxt].item()))
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
            first_answer_id = int(first_logits.argmax(dim=-1).item())
        features = self.features_from_snapshot(
            lens_logits=lens_logits,
            first_answer_id=first_answer_id,
            step_logprobs=step_logprobs,
            answer_len=len(gen_ids),
        )
        layer_entropies = features.get("layer_entropies", [])
        if not isinstance(layer_entropies, list):
            layer_entropies = []
        risk = self.router.score({k: v for k, v in features.items() if isinstance(v, float)})
        return LocalAnswer(
            answer=answer,
            gen_ids=gen_ids,
            prompt_tokens=int(input_ids.shape[1]),
            finish_reason=finish_reason,
            features=features,
            layer_entropies=[float(v) for v in layer_entropies],
            band_tokens=self.band_tokens_from_snapshot(lens_logits),
            workspace_grid=self.workspace_grid_from_snapshot(
                lens_logits,
                first_answer_id,
            ),
            risk=risk,
            snapshot_ms=snapshot_ms,
        )

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
        first_answer_id: int,
        step_logprobs: list[float],
        answer_len: int,
    ) -> dict[str, float | list[float]]:
        ranks_ans: list[int] = []
        ranks_hedge: list[int] = []
        entropies: list[float] = []
        top1s: list[int] = []

        for layer in self.band:
            logits = lens_logits[layer][0].float()
            order = logits.argsort(descending=True)
            rank_of = torch.empty_like(order)
            rank_of[order] = torch.arange(len(order), device=order.device)
            ranks_ans.append(int(rank_of[first_answer_id].item()))
            if self.hedge_ids:
                ranks_hedge.append(int(min(rank_of[t].item() for t in self.hedge_ids)))
            probs = logits.softmax(-1)
            entropies.append(float(-(probs * probs.clamp_min(1e-12).log()).sum().item()))
            top1s.append(int(order[0].item()))

        e = np.array(entropies, dtype=np.float64)
        n = len(e)
        x = np.arange(n, dtype=np.float64)
        slope = float(np.polyfit(x, e, 1)[0]) if n > 1 else 0.0
        ranks = np.array(ranks_ans, dtype=np.float64)
        ignited = np.nonzero(ranks <= 10)[0]
        if not step_logprobs:
            step_logprobs = [0.0]
        hedge_rank = min(ranks_hedge) if ranks_hedge else 0

        return {
            "bl_first_token_logprob": float(step_logprobs[0]),
            "bl_mean_logprob": float(np.mean(step_logprobs)),
            "bl_min_logprob": float(np.min(step_logprobs)),
            "bl_answer_len": float(answer_len),
            "ws_mean_entropy": float(e.mean()),
            "ws_max_entropy": float(e.max()),
            "ws_late_entropy": float(e[2 * n // 3 :].mean()),
            "ws_entropy_slope": slope,
            "ws_entropy_std": float(e.std()),
            "ws_ignition_frac": float((ranks <= 10).mean()),
            "ws_ignition_depth": float(ignited[0] / n) if len(ignited) else 1.0,
            "ws_mean_log_rank": float(np.log1p(ranks).mean()),
            "ws_band_agreement": float(np.mean(np.array(top1s) == first_answer_id)),
            "ws_hedge_rank": float(np.log1p(hedge_rank)),
            "layer_entropies": [round(float(v), 4) for v in entropies],
        }

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
            "columns": columns,
            "values": values,
            "answer_col": answer_col,
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
        "MODEL_ID": "google/gemma-4-12B-it",
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
    )


def load_runtime() -> Runtime:
    cfg = load_config()
    _ns = SIDECAR / "norm_stats.json"
    if _ns.exists():
        with _ns.open(encoding="utf-8") as f:
            RollingRouter.FROZEN = json.load(f)
    hf_model, runtime_model_id, runtime_quant, fallback_reason = load_hf_with_fallback(cfg)
    tokenizer = transformers.AutoTokenizer.from_pretrained(runtime_model_id)
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
    try:
        return load_hf(cfg.model_id, cfg.quant), cfg.model_id, cfg.quant, None
    except Exception as exc:
        if cfg.quant != "4bit" or not cfg.auto_fallback:
            raise
        fallback_id = "google/gemma-4-E4B-it"
        fallback = load_bf16_with_fit_device_map(fallback_id)
        reason = f"4bit load failed, fell back to {fallback_id} bf16: {exc}"
        return fallback, fallback_id, "bf16", reason


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
    return RollingRouter(
        feature_names=list(combined["features"]),
        weights=list(combined["weights"]),
        bias=float(combined["bias"]),
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


# The noise signal is defined at answer-onset. When the model preambles
# ("The singer who had...") the first generated token is filler with a clean
# workspace, so the detector reads that instead of the answer. Forcing a terse
# answer makes token-1 == the answer token, which is the setting the signal was
# validated in. Set LOCAL_TERSE=0 to disable.
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
        "quant": runtime.quant,
        "escalate_model": runtime.cfg.escalate_model or None,
        "threshold": runtime.cfg.risk_threshold,
        "band": [runtime.band[0], runtime.band[-1]],
        "fallback": runtime.fallback_reason,
    }


@app.get("/chat", response_class=HTMLResponse)
def chat_page() -> HTMLResponse:
    path = SIDECAR / "chat.html"
    if not path.exists():
        raise HTTPException(status_code=404, detail="chat.html not found")
    return HTMLResponse(path.read_text(encoding="utf-8"))


@app.post("/v1/chat/completions")
def chat_completions(body: dict[str, Any] = Body(...)) -> dict[str, Any]:
    if STATE.runtime is None:
        STATE.runtime = load_runtime()
    with STATE.lock:
        return STATE.runtime.answer(body)


@app.post("/escalate_one")
def escalate_one(body: dict[str, Any] = Body(...)) -> dict[str, Any]:
    if STATE.runtime is None:
        with STATE.lock:
            if STATE.runtime is None:
                STATE.runtime = load_runtime()
    return STATE.runtime.escalate_one(body)
