#!/usr/bin/env bash
# Gate C launch: regenerate long-answer Stage 1 rows, saving the things the first
# campaign threw away (token IDs, per-token logprobs, online prefix-LP features,
# fixed absolute-index checkpoints).
#
# Money guards, because this account has a card on file and credits are not a cap:
#   - bare `modal`, NOT `$PY -m modal` (modal is installed globally, not in .venv)
#   - JLENS_TIMEOUT_S bounds worst-case spend at (timeout x GPU rate x containers)
#   - retries=0 and max_containers are set in analysis/modal_campaign.py
#   - --max-new 96 MATCHES the original Stage 1 run; changing it breaks the
#     greedy-determinism check that lets us reuse the existing labels
set -euo pipefail
cd "$(dirname "$0")/.."

MANIFEST=${1:-campaign/manifests/gatec_smoke.jsonl}
TAG=${2:-gatec_smoke}
N_SHARDS=${3:-1}
VERIFY=${4:-20}

MODEL=google/gemma-4-12B-it
export PYTHONIOENCODING=utf-8 HF_HUB_DISABLE_SYMLINKS_WARNING=1
export JLENS_GPU=${JLENS_GPU:-L40S}
export JLENS_TIMEOUT_S=${JLENS_TIMEOUT_S:-1800}
export JLENS_MAX_CONTAINERS=${JLENS_MAX_CONTAINERS:-$N_SHARDS}

N=$(wc -l < "$MANIFEST")
EST=$(awk "BEGIN{printf \"%.2f\", ${N}/1000*2.02}")
WORST=$(awk "BEGIN{printf \"%.2f\", ${JLENS_TIMEOUT_S}/3600*1.95*${JLENS_MAX_CONTAINERS}}")
echo "== Gate C: ${N} prompts, ${N_SHARDS} shard(s), ${JLENS_GPU}, tag=${TAG}"
echo "   expected  ~\$${EST} at the measured \$2.02/1k blended rate"
echo "   worst case \$${WORST} if every container runs to the ${JLENS_TIMEOUT_S}s timeout"

mkdir -p scratch
for s in $(seq 0 $((N_SHARDS-1))); do
  echo "  shard $s -> campaign_${TAG}_shard${s}.jsonl"
  modal run analysis/modal_campaign.py::run \
    --manifest "$MANIFEST" --model "$MODEL" \
    --tag "$TAG" --shard "$s" --n-shards "$N_SHARDS" --max-new 96 \
    --verify $([ "$s" = "0" ] && echo "$VERIFY" || echo 0) \
    > "scratch/${TAG}_shard${s}.log" 2>&1 &
done
wait
echo "== done; logs in scratch/${TAG}_shard*.log"
