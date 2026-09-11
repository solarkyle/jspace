#!/usr/bin/env bash
# Pull the Gate C shards off the Modal volume, grade them, and score the gate.
# Every step here is CPU-only and free: grading is deterministic (the manifest is
# restricted to grader_type=exact sources precisely so no judge is needed).
set -euo pipefail
cd "$(dirname "$0")/.."

TAG=${1:-gatec}
N_SHARDS=${2:-4}
MODEL_SLUG=${3:-gemma-4-12b-it}
OUT=out/campaign
export PYTHONIOENCODING=utf-8

mkdir -p "$OUT"
echo "== download ${N_SHARDS} shard(s)"
for s in $(seq 0 $((N_SHARDS-1))); do
  modal volume get jlens-out "${MODEL_SLUG}/campaign_${TAG}_shard${s}.jsonl" \
    "${OUT}/${TAG}_shard${s}.jsonl" --force
done

echo "== concatenate"
cat "${OUT}/${TAG}_shard"*.jsonl > "${OUT}/${TAG}.jsonl"
wc -l < "${OUT}/${TAG}.jsonl"

echo "== grade (deterministic, free)"
python -m campaign.grade_deterministic \
  --input "${OUT}/${TAG}.jsonl" --out "${OUT}/${TAG}_graded.jsonl"

echo "== determinism check against the July traces"
python -m campaign.check_regen_determinism --input "${OUT}/${TAG}_graded.jsonl"

echo "== score Gate C"
python -m campaign.score_gate_c \
  --input "${OUT}/${TAG}_graded.jsonl" --out "${OUT}/gate_c.json"
