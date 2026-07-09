from __future__ import annotations

import importlib.util
import json
import math
import os
import re
import time
from dataclasses import dataclass, field
from importlib import resources
from pathlib import Path
from typing import Any

BAND_LO = 0.25
BAND_HI = 0.75
DEFAULT_MODEL = "google/gemma-4-E4B-it"
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

AVAILABLE_LENSES = [
    {
        "slug": "gemma-4-e4b-it",
        "model_id": "google/gemma-4-E4B-it",
        "label": "Gemma 4 E4B",
    },
    {
        "slug": "gemma-4-12b-it",
        "model_id": "google/gemma-4-12B-it",
        "label": "Gemma 4 12B",
    },
    {
        "slug": "huihui-gemma-4-12b-it-abliterated",
        "model_id": "huihui-ai/Huihui-gemma-4-12B-it-abliterated",
        "label": "Huihui Gemma 4 12B abliterated",
    },
    {
        "slug": "gemma-4-26b-a4b-it",
        "model_id": "google/gemma-4-26B-A4B-it",
        "label": "Gemma 4 26B A4B",
    },
    {
        "slug": "qwen3.6-27b",
        "model_id": "Qwen/Qwen3.6-27B",
        "label": "Qwen 3.6 27B",
    },
]

ROUTERS = {
    "gemma-4-e4b-it": {
        "features": [
            "bl_first_token_logprob",
            "bl_mean_logprob",
            "bl_min_logprob",
            "bl_answer_len",
            "ws_mean_entropy",
            "ws_max_entropy",
            "ws_late_entropy",
            "ws_entropy_slope",
            "ws_entropy_std",
            "ws_ignition_frac",
            "ws_ignition_depth",
            "ws_mean_log_rank",
            "ws_band_agreement",
            "ws_hedge_rank",
        ],
        "weights": [
            -0.35574,
            -0.57688,
            0.40373,
            0.34521,
            0.28294,
            0.38259,
            -0.14355,
            0.98739,
            -0.3593,
            0.3902,
            0.40726,
            -0.12692,
            -0.09776,
            0.20108,
        ],
        "bias": 0.40705,
    },
    "gemma-4-12b-it": {
        "features": [
            "bl_first_token_logprob",
            "bl_mean_logprob",
            "bl_min_logprob",
            "bl_answer_len",
            "ws_mean_entropy",
            "ws_max_entropy",
            "ws_late_entropy",
            "ws_entropy_slope",
            "ws_entropy_std",
            "ws_ignition_frac",
            "ws_ignition_depth",
            "ws_mean_log_rank",
            "ws_band_agreement",
            "ws_hedge_rank",
        ],
        "weights": [
            -0.72695,
            -0.05685,
            0.08191,
            -0.17313,
            0.28286,
            -0.72357,
            0.50632,
            0.4506,
            0.43193,
            -0.08501,
            -0.35828,
            0.15565,
            0.11485,
            -0.55831,
        ],
        "bias": 0.07648,
    },
    "huihui-gemma-4-12b-it-abliterated": {
        "features": [
            "bl_first_token_logprob",
            "bl_mean_logprob",
            "bl_min_logprob",
            "bl_answer_len",
            "ws_mean_entropy",
            "ws_max_entropy",
            "ws_late_entropy",
            "ws_entropy_slope",
            "ws_entropy_std",
            "ws_ignition_frac",
            "ws_ignition_depth",
            "ws_mean_log_rank",
            "ws_band_agreement",
            "ws_hedge_rank",
        ],
        "weights": [
            -0.70157,
            -0.18929,
            -0.05444,
            0.09719,
            -0.01666,
            0.59829,
            -0.54203,
            1.48867,
            -0.09453,
            0.08478,
            0.12635,
            0.00726,
            0.0,
            -0.32372,
        ],
        "bias": 0.23879,
    },
    "gemma-4-26b-a4b-it": {
        "features": [
            "bl_first_token_logprob",
            "bl_mean_logprob",
            "bl_min_logprob",
            "bl_answer_len",
            "ws_mean_entropy",
            "ws_max_entropy",
            "ws_late_entropy",
            "ws_entropy_slope",
            "ws_entropy_std",
            "ws_ignition_frac",
            "ws_ignition_depth",
            "ws_mean_log_rank",
            "ws_band_agreement",
            "ws_hedge_rank",
        ],
        "weights": [
            -0.81187,
            -0.13444,
            0.18382,
            0.16841,
            0.05808,
            0.36346,
            -0.06901,
            0.97275,
            0.3966,
            0.43184,
            -0.43184,
            0.20674,
            0.08671,
            -0.11975,
        ],
        "bias": -0.56518,
    },
    "qwen3.6-27b": {
        "features": [
            "bl_first_token_logprob",
            "bl_mean_logprob",
            "bl_min_logprob",
            "bl_answer_len",
            "ws_mean_entropy",
            "ws_max_entropy",
            "ws_late_entropy",
            "ws_entropy_slope",
            "ws_entropy_std",
            "ws_ignition_frac",
            "ws_ignition_depth",
            "ws_mean_log_rank",
            "ws_band_agreement",
            "ws_hedge_rank",
        ],
        "weights": [
            0.05127,
            -0.20117,
            -1.6785,
            -0.12354,
            0.09854,
            -0.30625,
            0.01904,
            -0.26732,
            0.58953,
            -0.20983,
            -0.15159,
            0.20149,
            0.01758,
            -0.28693,
        ],
        "bias": -0.5788,
    },
}

ROUTER_NORM_STATS = {
    "gemma-4-e4b-it": {
        "bl_first_token_logprob": [-0.18680763472514308, 0.3334438307000124],
        "bl_mean_logprob": [-0.08812023260144558, 0.08640312725033589],
        "bl_min_logprob": [-0.49210601466965453, 0.3875959896361741],
        "bl_answer_len": [11.022, 7.747871707765946],
        "ws_mean_entropy": [4.289909333333333, 0.3814461947048876],
        "ws_max_entropy": [8.717844, 0.9332776294672448],
        "ws_late_entropy": [7.416372485714286, 1.0178450169852675],
        "ws_entropy_slope": [0.24511220545454546, 0.07354418372471581],
        "ws_entropy_std": [2.719658958797589, 0.4063920808899519],
        "ws_ignition_frac": [0.014952380952380951, 0.05394639146344672],
        "ws_ignition_depth": [0.9634285714285714, 0.11681890941269689],
        "ws_mean_log_rank": [8.74651351773166, 1.515822295487187],
        "ws_band_agreement": [0.0016190476190476191, 0.01285220363680857],
        "ws_hedge_rank": [4.973914688341347, 0.6352312059557577],
    },
    "gemma-4-12b-it": {
        "bl_first_token_logprob": [-0.19050464271170248, 0.3575396511984926],
        "bl_mean_logprob": [-0.10013178506882005, 0.05862577158180529],
        "bl_min_logprob": [-0.9847056640312075, 0.44576004864945834],
        "bl_answer_len": [22.448, 3.824041840775281],
        "ws_mean_entropy": [3.1669996166666663, 0.49263782679704105],
        "ws_max_entropy": [9.009086799999999, 0.46897717972813985],
        "ws_late_entropy": [1.682919375, 1.3166794769437709],
        "ws_entropy_slope": [-0.10617562086956522, 0.09380618361271512],
        "ws_entropy_std": [2.8437043833529647, 0.1882740132431386],
        "ws_ignition_frac": [0.00375, 0.03089801773577069],
        "ws_ignition_depth": [0.9903333333333334, 0.08781862621967569],
        "ws_mean_log_rank": [11.018422225977007, 0.9026169863614572],
        "ws_band_agreement": [8.333333333333333e-05, 0.001861525658640723],
        "ws_hedge_rank": [7.224672026869209, 1.2596939104506406],
    },
    "huihui-gemma-4-12b-it-abliterated": {
        "bl_first_token_logprob": [-0.1638284393569174, 0.3096914025447826],
        "bl_mean_logprob": [-0.08466774430471635, 0.05104008428968878],
        "bl_min_logprob": [-0.8739933599345386, 0.385361586576625],
        "bl_answer_len": [22.542, 3.8982349852208755],
        "ws_mean_entropy": [3.554442525, 0.37389707510054476],
        "ws_max_entropy": [8.7878816, 0.33072152439997005],
        "ws_late_entropy": [1.3412479249999998, 0.8855060223851569],
        "ws_entropy_slope": [-0.13800729452173915, 0.06559333377520767],
        "ws_entropy_std": [3.1461558333171262, 0.10492047355022222],
        "ws_ignition_frac": [0.00225, 0.0221690161662122],
        "ws_ignition_depth": [0.9919166666666667, 0.08555744496470713],
        "ws_mean_log_rank": [11.068858990269932, 0.8760815760133189],
        "ws_band_agreement": [0.0, 0.0],
        "ws_hedge_rank": [7.29604362326081, 0.5120249734482138],
    },
    "gemma-4-26b-a4b-it": {
        "bl_first_token_logprob": [-0.12912354012159416, 0.28501491136713797],
        "bl_mean_logprob": [-0.10667088930630504, 0.07906247961360045],
        "bl_min_logprob": [-0.7327741804496454, 0.45518073088848143],
        "bl_answer_len": [17.314, 8.51019412234527],
        "ws_mean_entropy": [8.907819906666667, 0.26909664256196997],
        "ws_max_entropy": [10.8451728, 0.07498579212197468],
        "ws_late_entropy": [7.90232248, 0.7052237428789884],
        "ws_entropy_slope": [-0.10299300928571428, 0.06842854224015979],
        "ws_entropy_std": [2.02805640295033, 0.13510810387896152],
        "ws_ignition_frac": [0.0016, 0.014518034761403948],
        "ws_ignition_depth": [0.9984, 0.014518034761403942],
        "ws_mean_log_rank": [10.370542878152339, 1.2548210164744111],
        "ws_band_agreement": [0.0004, 0.005148462553682086],
        "ws_hedge_rank": [5.2347262049756385, 0.36442091158716394],
    },
    "qwen3.6-27b": {
        "bl_first_token_logprob": [-0.4560135702027837, 0.5793894857672305],
        "bl_mean_logprob": [-0.2539550558321474, 0.3611503637129811],
        "bl_min_logprob": [-0.5274681434140948, 0.6197578035281954],
        "bl_answer_len": [3.252, 2.269029748592997],
        "ws_mean_entropy": [1.98413851875, 0.8794302615073911],
        "ws_max_entropy": [8.9006726, 2.2318387095373264],
        "ws_late_entropy": [5.196933072727273, 2.4627450489270686],
        "ws_entropy_slope": [0.21631288134164223, 0.10564207991224402],
        "ws_entropy_std": [3.0104002842990805, 1.0424485069196412],
        "ws_ignition_frac": [0.00675, 0.03834139408002792],
        "ws_ignition_depth": [0.984625, 0.06680440198819236],
        "ws_mean_log_rank": [8.846927968475589, 1.3915750232855022],
        "ws_band_agreement": [0.0004375, 0.007513269511337923],
        "ws_hedge_rank": [4.587580387369737, 0.426560759066764],
    },
}

GEMMA4_DEVICE_MAP = {
    "lm_head": "cpu",
    "model.language_model": 0,
    "model.language_model.embed_tokens": "cpu",
    "model.language_model.embed_tokens_per_layer": "cpu",
    "model.language_model.per_layer_model_projection": "cpu",
    "model.language_model.per_layer_projection_norm": "cpu",
    "model.vision_tower": "cpu",
    "model.audio_tower": "cpu",
    "model.embed_vision": "cpu",
    "model.embed_audio": "cpu",
}


@dataclass
class Snapshot:
    """A local model answer plus its band-layer workspace heatmap."""

    answer: str
    noise: float
    entropies: list[float]
    grid: dict[str, Any]
    features: dict[str, Any] = field(default_factory=dict)
    band_tokens: list[dict[str, Any]] = field(default_factory=list)
    model_id: str = ""
    quant: str = ""
    prompt: str = ""
    snapshot_ms: float = 0.0

    def show(self, threshold: float = 0.6) -> None:
        """Print an ANSI workspace heatmap."""
        from .render import print_snapshot

        print_snapshot(self, threshold=threshold)

    def to_html(self, path: str | os.PathLike[str], threshold: float = 0.6) -> Path:
        """Write a standalone HTML workspace heatmap."""
        from .render import write_html

        return write_html(self, path, threshold=threshold)


@dataclass
class _WorkspaceSnapshot:
    step: int
    token_id: int
    token_text: str
    token_logprob: float
    lens_logits: dict[int, Any]
    risk: float = 0.0
    features: dict[str, Any] = field(default_factory=dict)


@dataclass
class _LocalAnswer:
    answer: str
    gen_ids: list[int]
    prompt_tokens: int
    finish_reason: str
    features: dict[str, Any]
    layer_entropies: list[float]
    band_tokens: list[dict[str, Any]]
    workspace_grid: dict[str, Any]
    risk: float
    snapshot_ms: float


class RollingRouter:
    """Small logistic router using frozen feature normalization."""

    FROZEN: dict[str, list[float]] | None = None

    def __init__(
        self,
        feature_names: list[str],
        weights: list[float],
        bias: float,
        norm_stats: dict[str, list[float]] | None = None,
    ) -> None:
        self.feature_names = feature_names
        self.weights = weights
        self.bias = float(bias)
        self.norm_stats = norm_stats
        self.history: list[list[float]] = []

    def score(self, raw_features: dict[str, float], *, record: bool = True) -> float:
        x = [float(raw_features.get(name, 0.0)) for name in self.feature_names]
        frozen = self.norm_stats or RollingRouter.FROZEN
        if frozen:
            z = []
            for name, value in zip(self.feature_names, x):
                mu, sd = frozen.get(name, [0.0, 1.0])
                z.append((value - float(mu)) / (float(sd) if float(sd) >= 1e-6 else 1.0))
            if record:
                self._append_history(x)
            return _sigmoid(_dot(z, self.weights) + self.bias)

        samples = [*self.history, x]
        if len(samples) < 2:
            z = [0.0 for _ in x]
        else:
            z = []
            for col in range(len(x)):
                vals = [row[col] for row in samples]
                mu = _mean(vals)
                sd = _std(vals)
                z.append((x[col] - mu) / (sd if sd >= 1e-6 else 1.0))
        if record:
            self._append_history(x)
        return _sigmoid(_dot(z, self.weights) + self.bias)

    def _append_history(self, row: list[float]) -> None:
        self.history.append(row)
        if len(self.history) > 200:
            del self.history[0]


@dataclass
class _Runtime:
    model_id: str
    quant: str
    tokenizer: Any
    model: Any
    lens: Any
    band: list[int]
    router: RollingRouter
    stop_ids: set[int]
    hedge_ids: list[int]
    torch: Any
    activation_recorder: Any
    max_prompt_tokens: int
    read_tokens: int

    def local_completion(self, prompt: str, max_new: int) -> _LocalAnswer:
        input_ids = self.model.encode(prompt, max_length=self.max_prompt_tokens)
        ids = input_ids
        gen_ids: list[int] = []
        step_logprobs: list[float] = []
        snapshots: list[_WorkspaceSnapshot] = []
        snapshot_ms = 0.0
        finish_reason = "length"
        read_tokens = max(1, int(self.read_tokens))

        for step in range(max_new):
            lens_logits = None
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
                    _WorkspaceSnapshot(
                        step=step,
                        token_id=nxt,
                        token_text=sanitize_band_token(self.tokenizer.decode([nxt])),
                        token_logprob=token_logprob,
                        lens_logits=lens_logits,
                    )
                )
            token = self.torch.tensor([[nxt]], device=ids.device, dtype=ids.dtype)
            ids = self.torch.cat([ids, token], dim=1)
        else:
            finish_reason = "length"

        answer = strip_model_spillover(
            self.tokenizer.decode(gen_ids, skip_special_tokens=True),
            self.model_id,
        )
        if not snapshots:
            snap_start = time.perf_counter()
            lens_logits, model_logits = self.lens_snapshot_from_ids(input_ids)
            snapshot_ms += (time.perf_counter() - snap_start) * 1000.0
            fallback_token = int(model_logits.argmax(dim=-1).item())
            snapshots.append(
                _WorkspaceSnapshot(
                    step=0,
                    token_id=fallback_token,
                    token_text=sanitize_band_token(self.tokenizer.decode([fallback_token])),
                    token_logprob=0.0,
                    lens_logits=lens_logits,
                )
            )

        for snap in snapshots:
            snap.features = self.features_from_snapshot(
                lens_logits=snap.lens_logits,
                answer_token_id=snap.token_id,
                first_answer_logprob=step_logprobs[0] if step_logprobs else 0.0,
                step_logprobs=step_logprobs,
                answer_len=len(gen_ids),
                read_step=snap.step,
                read_token=snap.token_text,
            )
            snap.risk = self.router.score(_numeric_features(snap.features), record=False)

        selected = max(snapshots, key=lambda snap: snap.risk)
        risk = self.router.score(_numeric_features(selected.features))
        features = dict(selected.features)
        features["ws_selected_risk"] = float(risk)
        features["ws_read_tokens"] = float(len(snapshots))
        features["ws_configured_read_tokens"] = float(read_tokens)
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

        return _LocalAnswer(
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
            risk=risk,
            snapshot_ms=snapshot_ms,
        )

    def lens_snapshot_from_ids(self, input_ids: Any) -> tuple[dict[int, Any], Any]:
        final_layer = self.model.n_layers - 1
        record_at = sorted({*self.band, final_layer})
        with self.torch.no_grad():
            with self.activation_recorder(self.model.layers, at=record_at) as recorder:
                self.model.forward(input_ids)
                activations = {
                    layer: recorder.activations[layer].detach() for layer in record_at
                }

        def select(layer: int) -> Any:
            return activations[layer][0, -1:].float()

        # transport + unembed must also run grad-free: lm_head requires grad,
        # and vocab-sized tensors with live graphs accumulate across a
        # generation (reference decorates the whole method with no_grad)
        with self.torch.no_grad():
            lens_logits: dict[int, Any] = {}
            for layer in self.band:
                residual = self.lens.transport(select(layer), layer)
                lens_logits[layer] = self.model.unembed(residual).float().cpu()

            model_logits = self.model.unembed(select(final_layer)).float().cpu()
        return lens_logits, model_logits

    def next_logits(self, ids: Any) -> Any:
        with self.torch.no_grad():
            hidden = self.model.forward(ids).last_hidden_state[:, -1]
            head = self.model._lm_head
            logits = head(hidden.to(head.weight.dtype).to(head.weight.device))
            softcap = getattr(self.model, "_logit_softcap", None)
            if softcap is not None:
                logits = softcap * self.torch.tanh(logits / softcap)
        return logits

    def features_from_snapshot(
        self,
        lens_logits: dict[int, Any],
        answer_token_id: int,
        first_answer_logprob: float,
        step_logprobs: list[float],
        answer_len: int,
        read_step: int,
        read_token: str,
    ) -> dict[str, Any]:
        ranks_ans: list[int] = []
        ranks_hedge: list[int] = []
        entropies: list[float] = []
        top1s: list[int] = []

        for layer in self.band:
            logits = lens_logits[layer][0].float()
            order = logits.argsort(descending=True)
            rank_of = self.torch.empty_like(order)
            rank_of[order] = self.torch.arange(len(order), device=order.device)
            ranks_ans.append(int(rank_of[answer_token_id].item()))
            if self.hedge_ids:
                ranks_hedge.append(
                    int(min(rank_of[token_id].item() for token_id in self.hedge_ids))
                )
            probs = logits.softmax(-1)
            ent = float(-(probs * probs.clamp_min(1e-12).log()).sum().item())
            entropies.append(ent)
            top1s.append(int(order[0].item()))

        n = len(entropies)
        ranks = [float(rank) for rank in ranks_ans]
        ignited = [idx for idx, rank in enumerate(ranks) if rank <= 10]
        if not step_logprobs:
            step_logprobs = [0.0]
        hedge_rank = min(ranks_hedge) if ranks_hedge else 0

        return {
            "bl_first_token_logprob": float(first_answer_logprob),
            "bl_mean_logprob": float(_mean(step_logprobs)),
            "bl_min_logprob": float(min(step_logprobs)),
            "bl_answer_len": float(answer_len),
            "ws_read_step": float(read_step),
            "ws_mean_entropy": float(_mean(entropies)),
            "ws_max_entropy": float(max(entropies)),
            "ws_late_entropy": float(_mean(entropies[2 * n // 3 :])),
            "ws_entropy_slope": float(_slope(entropies)),
            "ws_entropy_std": float(_std(entropies)),
            "ws_ignition_frac": float(_mean([1.0 if rank <= 10 else 0.0 for rank in ranks])),
            "ws_ignition_depth": float(ignited[0] / n) if ignited else 1.0,
            "ws_mean_log_rank": float(_mean([math.log1p(rank) for rank in ranks])),
            "ws_band_agreement": float(
                _mean([1.0 if token_id == answer_token_id else 0.0 for token_id in top1s])
            ),
            "ws_hedge_rank": float(math.log1p(hedge_rank)),
            "ws_read_token": read_token,
            "layer_entropies": [round(float(value), 4) for value in entropies],
        }

    def band_tokens_from_snapshot(self, lens_logits: dict[int, Any]) -> list[dict[str, Any]]:
        sample_indexes = sorted({0, len(self.band) // 2, len(self.band) - 1})
        rows: list[dict[str, Any]] = []
        for idx in sample_indexes:
            layer = self.band[idx]
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
        lens_logits: dict[int, Any],
        answer_token_id: int,
    ) -> dict[str, Any]:
        rows_by_id: list[dict[int, float]] = []
        best_prob_by_id: dict[int, float] = {}

        for layer in self.band:
            logits = lens_logits[layer][0].float()
            probs = logits.softmax(-1)
            take = min(10, int(probs.numel()))
            top_probs, top_ids = self.torch.topk(probs, take)
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
            answer_col = column_ids.index(int(answer_token_id))
        except ValueError:
            answer_col = -1

        return {
            "layers": [int(layer) for layer in self.band],
            "columns": columns,
            "values": values,
            "answer_col": answer_col,
        }


class Workspace:
    """A lazy local model workspace viewer."""

    def __init__(
        self,
        model_id: str = DEFAULT_MODEL,
        quant: str | None = "4bit",
        *,
        lens_path: str | os.PathLike[str] | None = None,
        lens_device: str = "auto",
        max_prompt_tokens: int = 1536,
        read_tokens: int = 3,
    ) -> None:
        self.model_id = model_id
        self.quant = normalize_quant(quant)
        self.lens_path = Path(lens_path) if lens_path is not None else None
        self.lens_device = lens_device.lower()
        self.max_prompt_tokens = int(max_prompt_tokens)
        self.read_tokens = max(1, int(read_tokens))
        self._runtime: _Runtime | None = None

    def snapshot(self, prompt: str | list[dict[str, Any]], max_new_tokens: int = 64) -> Snapshot:
        """Generate a short answer and capture its workspace heatmap."""
        runtime = self._ensure_runtime()
        rendered_prompt = build_prompt_from_input(runtime.tokenizer, prompt)
        # never silently reduce an explicit request; cap matches the sidecar
        max_new = max(1, min(int(max_new_tokens), 512))
        local = runtime.local_completion(rendered_prompt, max_new=max_new)
        return Snapshot(
            answer=local.answer,
            noise=float(local.risk),
            entropies=local.layer_entropies,
            grid=local.workspace_grid,
            features=local.features,
            band_tokens=local.band_tokens,
            model_id=runtime.model_id,
            quant=runtime.quant,
            prompt=message_content({"content": prompt}) if isinstance(prompt, str) else "",
            snapshot_ms=local.snapshot_ms,
        )

    def _ensure_runtime(self) -> _Runtime:
        if self._runtime is not None:
            return self._runtime

        if os.path.isdir("E:/hf-cache"):
            os.environ.setdefault("HF_HOME", "E:/hf-cache")

        import torch
        import transformers
        import jlens
        from jlens.hooks import ActivationRecorder

        load_frozen_norm_stats()
        hf_model = load_hf(self.model_id, self.quant, torch, transformers)
        tokenizer = transformers.AutoTokenizer.from_pretrained(self.model_id)
        model = jlens.from_hf(hf_model, tokenizer)

        lens_path = resolve_lens_path(self.model_id, self.lens_path)
        lens = jlens.JacobianLens.load(str(lens_path))
        band = [
            layer
            for layer in range(int(model.n_layers * BAND_LO), int(model.n_layers * BAND_HI))
            if layer in lens.jacobians
        ]
        if not band:
            raise RuntimeError(f"no fitted lens layers found in band for {lens_path}")

        lens = prepare_lens(lens, band, self.lens_device, self.quant, torch)
        router = load_router(self.model_id)
        self._runtime = _Runtime(
            model_id=self.model_id,
            quant=self.quant,
            tokenizer=tokenizer,
            model=model,
            lens=lens,
            band=band,
            router=router,
            stop_ids=stop_ids(tokenizer),
            hedge_ids=hedge_ids(tokenizer),
            torch=torch,
            activation_recorder=ActivationRecorder,
            max_prompt_tokens=self.max_prompt_tokens,
            read_tokens=self.read_tokens,
        )
        return self._runtime


def normalize_quant(quant: str | None) -> str:
    if quant is None:
        return "bf16"
    value = str(quant).lower()
    if value in {"none", "bf16", "bfloat16"}:
        return "bf16"
    if value == "4bit":
        return "4bit"
    raise ValueError("quant must be '4bit' or None")


def model_slug(model_id: str) -> str:
    return model_id.split("/")[-1].lower()


def resolve_lens_path(
    model_id: str,
    explicit_path: Path | None,
) -> Path:
    if explicit_path is not None:
        return explicit_path

    slug = model_slug(model_id)
    local = Path("out") / slug / "lens.pt"
    if local.exists():
        return local

    from huggingface_hub import hf_hub_download

    repo = os.environ.get("LENS_HUB_REPO", "solarkyle/jspace-lenses")
    return Path(hf_hub_download(repo_id=repo, filename=f"{slug}/lens.pt"))


def load_hf(model_id: str, quant: str, torch: Any, transformers: Any) -> Any:
    if quant == "4bit":
        if not torch.cuda.is_available():
            raise RuntimeError("quant='4bit' needs CUDA")
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
        return from_pretrained_any(model_id, transformers, **kwargs)
    if quant == "bf16":
        return load_bf16_with_fit_device_map(model_id, torch, transformers)
    raise ValueError(f"unsupported quant={quant!r}")


def load_bf16_with_fit_device_map(model_id: str, torch: Any, transformers: Any) -> Any:
    device_map = GEMMA4_DEVICE_MAP if "gemma-4" in model_id.lower() else "auto"
    if isinstance(device_map, dict):
        from accelerate import dispatch_model

        hf_model = from_pretrained_with_dtype(model_id, transformers, torch.bfloat16)
        return dispatch_model(
            hf_model,
            device_map=device_map,
            main_device="cpu",
            skip_keys=getattr(hf_model, "_skip_keys_device_placement", None),
        )
    return from_pretrained_with_dtype(
        model_id,
        transformers,
        torch.bfloat16,
        device_map=device_map,
    )


def from_pretrained_with_dtype(
    model_id: str,
    transformers: Any,
    dtype: Any,
    **kwargs: Any,
) -> Any:
    try:
        return from_pretrained_any(model_id, transformers, dtype=dtype, **kwargs)
    except TypeError as exc:
        if "dtype" not in str(exc):
            raise
        return from_pretrained_any(model_id, transformers, torch_dtype=dtype, **kwargs)


def from_pretrained_any(model_id: str, transformers: Any, **kwargs: Any) -> Any:
    try:
        return transformers.AutoModelForCausalLM.from_pretrained(model_id, **kwargs)
    except ValueError:
        return transformers.AutoModelForImageTextToText.from_pretrained(
            model_id,
            **kwargs,
        )


def prepare_lens(
    lens: Any,
    band: list[int],
    lens_device: str,
    quant: str,
    torch: Any,
) -> Any:
    lens.jacobians = {layer: lens.jacobians[layer] for layer in band}
    lens.source_layers = band
    target = choose_lens_device(lens, lens_device, quant, torch)
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

    def transport(self: Any, residual: Any, layer: int) -> Any:
        jacobian = self.jacobians[layer]
        if self._jspace_lens_device == "cpu":
            residual = residual.to(device="cpu", dtype=jacobian.dtype)
        else:
            residual = residual.to(device=jacobian.device, dtype=jacobian.dtype)
        return residual @ jacobian.T

    import types

    lens.transport = types.MethodType(transport, lens)
    return lens


def choose_lens_device(lens: Any, lens_device: str, quant: str, torch: Any) -> str:
    if lens_device in {"cpu", "cuda"}:
        return lens_device
    if not torch.cuda.is_available() or quant != "4bit":
        return "cpu"
    lens_bytes = sum(tensor.numel() * 2 for tensor in lens.jacobians.values())
    allocated = torch.cuda.memory_allocated()
    if allocated + lens_bytes > VRAM_BUDGET_BYTES:
        return "cpu"
    return "cuda"


def load_router(model_id: str) -> RollingRouter:
    slug = model_slug(model_id)
    if slug not in ROUTERS:
        raise RuntimeError(f"router weights for {slug!r} are not bundled")
    data = ROUTERS[slug]
    return RollingRouter(
        feature_names=list(data["features"]),
        weights=list(data["weights"]),
        bias=float(data["bias"]),
        norm_stats=ROUTER_NORM_STATS.get(slug),
    )


def load_frozen_norm_stats() -> dict[str, list[float]]:
    if RollingRouter.FROZEN is not None:
        return RollingRouter.FROZEN
    with resources.files("jspace").joinpath("norm_stats.json").open(
        encoding="utf-8"
    ) as handle:
        RollingRouter.FROZEN = json.load(handle)
    return RollingRouter.FROZEN


def has_package(name: str) -> bool:
    return importlib.util.find_spec(name) is not None


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


TERSE_SYSTEM = (
    "Answer directly and concisely in plain text. Lead with the answer itself. "
    "No markdown, no bold, no asterisks, no preamble, no restating the question."
)


def build_prompt_from_input(tokenizer: Any, prompt: str | list[dict[str, Any]]) -> str:
    if isinstance(prompt, str):
        messages = [{"role": "user", "content": prompt}]
    else:
        messages = prompt
    return build_prompt(tokenizer, messages)


def build_prompt(tokenizer: Any, messages: list[dict[str, Any]]) -> str:
    clean_messages = [
        {"role": str(msg.get("role", "user")), "content": message_content(msg)}
        for msg in messages
    ]
    if os.environ.get("LOCAL_TERSE", "1").lower() not in {"0", "false", "no"}:
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
    lines = [f"{msg['role']}: {msg['content']}" for msg in clean_messages]
    return "\n".join(lines) + "\nassistant:"


def strip_model_spillover(answer: str, model_id: str) -> str:
    answer = answer.strip()
    if "gemma" not in model_id.lower():
        return answer
    return answer.split("thought", 1)[0].strip()


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


def _numeric_features(features: dict[str, Any]) -> dict[str, float]:
    return {key: value for key, value in features.items() if isinstance(value, float)}


def _mean(values: list[float]) -> float:
    return sum(values) / len(values) if values else 0.0


def _std(values: list[float]) -> float:
    if not values:
        return 0.0
    mu = _mean(values)
    return math.sqrt(sum((value - mu) ** 2 for value in values) / len(values))


def _slope(values: list[float]) -> float:
    n = len(values)
    if n <= 1:
        return 0.0
    mean_x = (n - 1) / 2.0
    mean_y = _mean(values)
    denom = sum((idx - mean_x) ** 2 for idx in range(n))
    if denom == 0.0:
        return 0.0
    return sum((idx - mean_x) * (value - mean_y) for idx, value in enumerate(values)) / denom


def _dot(left: list[float], right: list[float]) -> float:
    return sum(a * b for a, b in zip(left, right))


def _sigmoid(logit: float) -> float:
    if logit >= 0:
        return 1.0 / (1.0 + math.exp(-logit))
    exp_logit = math.exp(logit)
    return exp_logit / (1.0 + exp_logit)
