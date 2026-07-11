#!/usr/bin/env bash
# Post-judging bakeoff: concat Sonnet verdicts, ingest, build features, run the
# full classifier bakeoff under LODO, and score Gate A. Run after all Sonnet
# judge shards have written verdicts_sonnet_stage1_shard*.jsonl.
set -euo pipefail
cd "$(dirname "$0")/.."
PY=.venv/Scripts/python
TABFM_PY="C:/Users/18632/Desktop/stuff/ufc_bet/.venv-tabfm/Scripts/python.exe"
export PYTHONIOENCODING=utf-8

echo "== concat verdicts + ingest =="
cat out/campaign/verdicts_sonnet_stage1_shard*.jsonl > out/campaign/verdicts_stage1.jsonl
echo "verdicts: $(wc -l < out/campaign/verdicts_stage1.jsonl)"
$PY -m campaign.grade_claude ingest --graded out/campaign/stage1_graded.jsonl \
  --verdicts out/campaign/verdicts_stage1.jsonl --out out/campaign/stage1_judged.jsonl

echo "== feature table =="
$PY -m campaign.build_feature_table --input out/campaign/stage1_judged.jsonl \
  --out out/campaign/stage1_features.jsonl

echo "== split + leakage sanity =="
$PY -m campaign.split_groups --input out/campaign/stage1_features.jsonl

echo "== core bakeoff (logistic + LightGBM) =="
$PY -m campaign.train_baselines --input out/campaign/stage1_features.jsonl \
  --out out/campaign/stage1_bakeoff.json

echo "== TabFM =="
"$TABFM_PY" -m campaign.train_tabfm --input out/campaign/stage1_features.jsonl \
  --out out/campaign/stage1_tabfm.json || echo "(tabfm venv unavailable, skipped)"

echo "== controls: CatBoost + MLP =="
$PY -m campaign.train_extra --input out/campaign/stage1_features.jsonl \
  --out out/campaign/stage1_extra.json

echo "== temporal CNN vs static =="
$PY -m campaign.train_temporal --input out/campaign/stage1_features.jsonl \
  --out out/campaign/stage1_temporal.json

echo "== GATE A =="
$PY -m campaign.score_gates --bakeoff out/campaign/stage1_bakeoff.json
