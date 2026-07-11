#!/usr/bin/env bash
# Stage 0 pilot report driver: pull traces off Modal, grade, featurize, check
# splits, run the static bakeoff. Idempotent; re-run after any trace refresh.
set -euo pipefail
cd "$(dirname "$0")/.."

SLUG=gemma-4-12b-it
TAG=${1:-pilot}
PY=.venv/Scripts/python
mkdir -p out/campaign

echo "== pulling traces from Modal =="
modal volume get --force jlens-out "${SLUG}/campaign_${TAG}_shard0.jsonl" \
  "out/campaign/${TAG}.jsonl"
wc -l "out/campaign/${TAG}.jsonl"

echo "== grade =="
PYTHONIOENCODING=utf-8 $PY -m campaign.grade_deterministic \
  --input "out/campaign/${TAG}.jsonl" --out "out/campaign/${TAG}_graded.jsonl"

echo "== feature table =="
PYTHONIOENCODING=utf-8 $PY -m campaign.build_feature_table \
  --input "out/campaign/${TAG}_graded.jsonl" --out "out/campaign/${TAG}_features.jsonl"

echo "== split + leakage sanity =="
PYTHONIOENCODING=utf-8 $PY -m campaign.split_groups \
  --input "out/campaign/${TAG}_features.jsonl"

echo "== static bakeoff =="
PYTHONIOENCODING=utf-8 $PY -m campaign.train_baselines \
  --input "out/campaign/${TAG}_features.jsonl" \
  --out "out/campaign/${TAG}_bakeoff.json"
