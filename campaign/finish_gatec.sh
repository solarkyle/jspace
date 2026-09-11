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
# Use the project interpreter. A bare `python` here silently dropped LightGBM from
# the scorer, because lightgbm is installed in .venv and not globally, and the
# scorer reports whichever models it can import rather than failing.
PY=${PY:-.venv/Scripts/python.exe}
[ -x "$PY" ] || PY=python

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
"$PY" -m campaign.grade_deterministic \
  --input "${OUT}/${TAG}.jsonl" --out "${OUT}/${TAG}_graded.jsonl"

echo "== determinism check against the July traces"
"$PY" -m campaign.check_regen_determinism --input "${OUT}/${TAG}_graded.jsonl"

echo "== score Gate C"
"$PY" -m campaign.score_gate_c \
  --input "${OUT}/${TAG}_graded.jsonl" --out "${OUT}/gate_c.json"
