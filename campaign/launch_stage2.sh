#!/usr/bin/env bash
# Stage 2 launch: prospective validation generation. The manifest and prereg
# (campaign/PREREG_STAGE2.md, frozen model hashes) must already exist; this
# script only generates traces. 4 shards is plenty for 7.1k prompts.
set -euo pipefail
cd "$(dirname "$0")/.."

PY=.venv/Scripts/python
N_SHARDS=${1:-4}
MODEL=google/gemma-4-12B-it
export PYTHONIOENCODING=utf-8 HF_HUB_DISABLE_SYMLINKS_WARNING=1

test -f campaign/PREREG_STAGE2.md || { echo "no prereg; refusing"; exit 1; }
test -f out/campaign/frozen/frozen_meta.json || { echo "no frozen models; refusing"; exit 1; }

echo "== validate =="
$PY -m campaign.validate_manifest campaign/manifests/stage2.jsonl

N=$(wc -l < campaign/manifests/stage2.jsonl)
echo "== launching ${N} prompts across ${N_SHARDS} shards =="
echo "   estimated cost at stage-1 blended rate (~\$1.92/1k): \$$(awk "BEGIN{printf \"%.0f\", ${N}/1000*1.92}")"

for s in $(seq 0 $((N_SHARDS-1))); do
  echo "  shard $s -> campaign_stage2_shard${s}.jsonl"
  modal run modal_campaign.py::run \
    --manifest campaign/manifests/stage2.jsonl --model "$MODEL" \
    --tag stage2 --shard "$s" --n-shards "$N_SHARDS" --max-new 96 \
    --verify $([ "$s" = "0" ] && echo 30 || echo 0) \
    > "scratch/stage2_shard${s}.log" 2>&1 &
done
echo "launched ${N_SHARDS} shards in background; logs in scratch/stage2_shard*.log"
wait
echo "all shards complete"
