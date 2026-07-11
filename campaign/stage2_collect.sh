#!/usr/bin/env bash
# Stage 2 post-generation collection: pull 4 shards, concat, grade
# deterministic rows (exact + tool AST), and emit blinded judge requests for
# the two LLM-graded sources (truthfulqa, facts_grounding). Judging runs on
# Codex/GPT-5.5 per prereg - NOT Anthropic models, NOT parallel subagents.
set -euo pipefail
cd "$(dirname "$0")/.."
PY=.venv/Scripts/python
SLUG=gemma-4-12b-it
export PYTHONIOENCODING=utf-8
mkdir -p out/campaign

echo "== pull 4 shards =="
: > out/campaign/stage2.jsonl
for s in 0 1 2 3; do
  modal volume get --force jlens-out "${SLUG}/campaign_stage2_shard${s}.jsonl" \
    "out/campaign/stage2_shard${s}.jsonl"
  cat "out/campaign/stage2_shard${s}.jsonl" >> out/campaign/stage2.jsonl
done
echo "total traces: $(wc -l < out/campaign/stage2.jsonl)"

echo "== deterministic grade =="
$PY -m campaign.grade_deterministic --input out/campaign/stage2.jsonl \
  --out out/campaign/stage2_graded.jsonl

echo "== emit blinded judge requests (for Codex judging) =="
$PY -m campaign.grade_claude emit --input out/campaign/stage2_graded.jsonl \
  --out out/campaign/stage2_judge_requests.jsonl

echo "COLLECT DONE - judge stage2_judge_requests.jsonl with Codex, then:"
echo "  grade_claude ingest -> build_feature_table -> score_gate_d"
