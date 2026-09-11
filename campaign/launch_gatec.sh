#!/usr/bin/env bash
# Gate C launch: regenerate long-answer Stage 1 rows, saving the things the first
# campaign threw away (token IDs, per-token logprobs, online prefix-LP features,
# fixed absolute-index checkpoints).
#
# Money guards, because this account has a card on file and credits are not a cap:
#   - bare `modal`, NOT `$PY -m modal` (modal is installed globally, not in .venv)
#   - JLENS_TIMEOUT_S bounds worst-case spend at (timeout x rate x containers)
#   - retries=0 and max_containers are set in analysis/modal_campaign.py
#   - cost is estimated from a MEASURED seconds-per-prompt and the rate of the GPU
#     actually selected, not from a blended dollars-per-1k figure. The pilot's
#     $2.02/1k came from a corpus dominated by 4-token trivia_qa answers; this
#     subset is the longest-context rows in the corpus and measured 10 s/prompt
#     on an L40S, which is about $5.42/1k.
#   - --max-new 96 MATCHES the original Stage 1 run
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
# Pin the model revision for anything you intend to reproduce later. Stage 1 and
# Stage 2 ran unpinned on 2026-07-10/11 and google/gemma-4-12B-it was modified on
# 2026-07-20, so those generations can no longer be recovered. The sha the cached
# image resolves to today is recorded in out/campaign/environment.json.
export JLENS_MODEL_REVISION=${JLENS_MODEL_REVISION:-}
# Measured on this subset. Override if you have a better number for your shape.
SEC_PER_PROMPT=${JLENS_SEC_PER_PROMPT:-10.0}

# Published list rates, US dollars per GPU-hour. Verify before relying on them;
# they change, and they exclude CPU and memory charges.
gpu_rate() {
  case "$1" in
    T4)           echo 0.59 ;;
    L4)           echo 0.80 ;;
    A10G)         echo 1.10 ;;
    A100-40GB)    echo 2.10 ;;
    L40S)         echo 1.95 ;;
    A100|A100-80GB) echo 2.50 ;;
    "RTX-PRO-6000") echo 3.03 ;;
    H100)         echo 3.95 ;;
    H200)         echo 4.54 ;;
    *)            echo "" ;;
  esac
}

RATE=$(gpu_rate "$JLENS_GPU")
if [ -z "$RATE" ]; then
  echo "WARNING: no known hourly rate for JLENS_GPU=${JLENS_GPU}; cost estimates skipped." >&2
  RATE=0
fi

N=$(wc -l < "$MANIFEST")
echo "== Gate C: ${N} prompts, ${N_SHARDS} shard(s), ${JLENS_GPU}, tag=${TAG}"
if [ "$RATE" != "0" ]; then
  awk -v n="$N" -v sp="$SEC_PER_PROMPT" -v rate="$RATE" -v to="$JLENS_TIMEOUT_S" \
      -v c="$JLENS_MAX_CONTAINERS" 'BEGIN{
    hours = n * sp / 3600;
    printf "   expected   $%.2f  (%.1f GPU-hr at %.0f s/prompt, $%.2f/hr)\n", hours*rate, hours, sp, rate;
    printf "   worst case $%.2f  (%d container(s) x %d s timeout)\n", to/3600*rate*c, c, to;
    printf "   per 1k     $%.2f\n", 1000*sp/3600*rate;
  }'
fi

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
