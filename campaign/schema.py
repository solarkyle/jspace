"""Canonical campaign record schemas (handoff section 8).

Two record types flow through the whole campaign:

  PromptRecord   one per unique prompt, produced by build_manifest.py
  ResponseRecord one per (prompt, model, generation config, seed), produced
                 by the Modal runner and enriched by grading/adjudication

Both carry stable content hashes so retries are idempotent and deduplication
is auditable. Keep this module dependency-free (stdlib only) so every stage,
local or Modal, can import it.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field, asdict
from typing import Any

SCHEMA_VERSION = 1

DOMAINS = {"general", "medical", "therapy", "legal", "finance", "tools",
           "reasoning", "dialogue", "temporal"}
TASK_TYPES = {"closed_qa", "grounded_qa", "unanswerable", "false_premise",
              "dialogue", "multi_hop", "numeric", "tool_call", "long_doc",
              "life_decision"}
GRADER_TYPES = {"exact", "numeric", "tool", "entailment", "rubric", "llm"}


def content_hash(*parts: str) -> str:
    h = hashlib.sha256()
    for p in parts:
        h.update(p.encode("utf-8"))
        h.update(b"\x1e")
    return h.hexdigest()[:20]


@dataclass
class PromptRecord:
    source_dataset: str
    source_revision: str          # pinned Hub revision / commit / dated version
    source_row_id: str
    upstream_group: str           # de-dup key across mirrors/aggregators
    license: str
    commercial_train_allowed: bool
    domain: str
    task_type: str
    split_group: str              # article / entity / conversation grouping key
    system: str
    prompt: str
    context: str = ""
    references: list[str] = field(default_factory=list)
    aliases: list[str] = field(default_factory=list)
    answerable: bool = True
    grader_type: str = "exact"
    conversation_id: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)
    example_id: str = ""
    schema_version: int = SCHEMA_VERSION

    def __post_init__(self) -> None:
        if self.domain not in DOMAINS:
            raise ValueError(f"unknown domain {self.domain!r}")
        if self.task_type not in TASK_TYPES:
            raise ValueError(f"unknown task_type {self.task_type!r}")
        if self.grader_type not in GRADER_TYPES:
            raise ValueError(f"unknown grader_type {self.grader_type!r}")
        if not self.split_group:
            raise ValueError("split_group is required (leakage prevention)")
        if not self.example_id:
            self.example_id = content_hash(self.system, self.prompt, self.context)

    def to_json(self) -> str:
        return json.dumps(asdict(self), ensure_ascii=False)


@dataclass
class ResponseRecord:
    example_id: str
    model_id: str
    model_revision: str
    lens_revision: str
    quantization: str             # "bf16" | "nf4"
    generation_config_id: str     # e.g. "greedy_v1", "sampled_t07_v1"
    seed: int
    answer: str
    token_count: int
    logprob_features: dict[str, float] = field(default_factory=dict)
    onset_workspace_features: dict[str, float] = field(default_factory=dict)
    # one dict per captured prefix: {"frac": 0.5, "token_index": 12, ...features}
    prefix_workspace_features: list[dict[str, float]] = field(default_factory=list)
    deterministic_grade: dict[str, Any] = field(default_factory=dict)
    judge_grades: list[dict[str, Any]] = field(default_factory=list)
    # final adjudicated labels, e.g. {"eventual_response_error": true, ...}
    final_labels: dict[str, Any] = field(default_factory=dict)
    artifact_revision: str = ""
    response_id: str = ""
    schema_version: int = SCHEMA_VERSION

    def __post_init__(self) -> None:
        if not self.response_id:
            self.response_id = content_hash(
                self.example_id, self.model_id, self.model_revision,
                self.generation_config_id, str(self.seed))

    def to_json(self) -> str:
        return json.dumps(asdict(self), ensure_ascii=False)


def load_jsonl(path: str, cls):
    out = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            if line.strip():
                d = json.loads(line)
                d["schema_version"] = SCHEMA_VERSION
                out.append(cls(**d))
    return out
