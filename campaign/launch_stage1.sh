#!/usr/bin/env bash
# Stage 1 launch: build the manifest (excluding pilot prompts), validate, then
# shard the two-pass runner across N parallel Modal workers. Parallel workers
# cost the same total GPU-seconds as serial; they only cut wall-clock.
set -euo pipefail
cd "$(dirname "$0")/.."

PY=.venv/Scripts/python
N_SHARDS=${1:-8}
MODEL=google/gemma-4-12B-it
export PYTHONIOENCODING=utf-8 HF_HUB_DISABLE_SYMLINKS_WARNING=1

echo "== build stage1 manifest (excluding pilot) =="
$PY -m campaign.build_manifest --stage 1 --out campaign/manifests/stage1.jsonl \
  --exclude-manifest campaign/manifests/pilot.jsonl

echo "== validate =="
$PY -m campaign.validate_manifest campaign/manifests/stage1.jsonl

N=$(wc -l < campaign/manifests/stage1.jsonl)
echo "== launching ${N} prompts across ${N_SHARDS} shards =="
echo "   estimated cost at pilot blended rate (\$2.02/1k, context-heavy will run higher): \$$(awk "BEGIN{printf \"%.0f\", ${N}/1000*2.02}")"
echo "   long-context shards (hotpotqa/drop/grounded) route to A100 via JLENS_GPU"

# Fire each shard as a detached modal run; they write disjoint output files.
for s in $(seq 0 $((N_SHARDS-1))); do
  echo "  shard $s -> campaign_stage1_shard${s}.jsonl"
  $PY -m modal run modal_campaign.py::run \
    --manifest campaign/manifests/stage1.jsonl --model "$MODEL" \
    --tag stage1 --shard "$s" --n-shards "$N_SHARDS" --max-new 96 \
    --verify $([ "$s" = "0" ] && echo 30 || echo 0) \
    > "scratch/stage1_shard${s}.log" 2>&1 &
done
echo "launched ${N_SHARDS} shards in background; logs in scratch/stage1_shard*.log"
wait
echo "all shards complete"
