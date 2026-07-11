#!/usr/bin/env bash
# Post-generation collection: pull all 8 Stage 1 shards, concat, grade the
# deterministic rows, emit blinded judge requests, and split them into N shards
# for parallel Sonnet judging. Run this once all 8 ephemeral apps have stopped.
set -euo pipefail
cd "$(dirname "$0")/.."
PY=.venv/Scripts/python
SLUG=gemma-4-12b-it
NJUDGE=${1:-28}
export PYTHONIOENCODING=utf-8
mkdir -p out/campaign

echo "== pull 8 shards =="
: > out/campaign/stage1.jsonl
for s in 0 1 2 3 4 5 6 7; do
  modal volume get --force jlens-out "${SLUG}/campaign_stage1_shard${s}.jsonl" \
    "out/campaign/stage1_shard${s}.jsonl"
  cat "out/campaign/stage1_shard${s}.jsonl" >> out/campaign/stage1.jsonl
done
echo "total traces: $(wc -l < out/campaign/stage1.jsonl)"

echo "== deterministic grade =="
$PY -m campaign.grade_deterministic --input out/campaign/stage1.jsonl \
  --out out/campaign/stage1_graded.jsonl

echo "== emit blinded judge requests =="
$PY -m campaign.grade_claude emit --input out/campaign/stage1_graded.jsonl \
  --out out/campaign/stage1_judge_requests.jsonl

echo "== split judge requests into ${NJUDGE} shards =="
$PY -c "
import json
rows=[l for l in open('out/campaign/stage1_judge_requests.jsonl',encoding='utf-8') if l.strip()]
N=${NJUDGE}
for i in range(N):
    open(f'out/campaign/stage1_judge_shard{i}.jsonl','w',encoding='utf-8').writelines(rows[i::N])
print(f'{len(rows)} requests split into {N} shards (~{len(rows)//N} each)')
"
echo "COLLECT DONE — dispatch ${NJUDGE} Sonnet judge agents on stage1_judge_shard*.jsonl"
