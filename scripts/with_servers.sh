#!/usr/bin/env bash
# Serve the policy and/or the reader with vLLM on one GPU, run a command against them, then stop the servers.
# Usage: bash scripts/with_servers.sh --gpu N [--policy-adapter DIR | --no-policy] [--reader MODEL_DIR] [--reader-gpu M] -- COMMAND...
#   The policy is the base model of configs/models.json ("policy"), with the LoRA adapter if given; COMMAND gets
#   POLICY_MODEL (the served name to pass as --policy-model), POLICY_URL and READER_URL.
#   With both on one GPU the policy takes 0.55 and the reader 0.35 of its memory; a server alone takes 0.85.
#   VLLM_BIN: the vllm executable of the GPU environment (default: vllm on PATH).
set -uo pipefail
GPU="" ADAPTER="" READER="" RGPU="" POLICY=1
while [ $# -gt 0 ] && [ "$1" != "--" ]; do
  case "$1" in
    --gpu) GPU=$2; shift 2 ;;
    --policy-adapter) ADAPTER=$2; shift 2 ;;
    --no-policy) POLICY=0; shift ;;
    --reader) READER=$2; shift 2 ;;
    --reader-gpu) RGPU=$2; shift 2 ;;
    *) echo "unknown option $1"; exit 2 ;;
  esac
done
[ "${1:-}" = "--" ] && shift
[ -n "$GPU" ] && [ $# -gt 0 ] || { echo "usage: with_servers.sh --gpu N [...] -- COMMAND..."; exit 2; }
: "${MODELS_DIR:?source configs/paths.env first}" "${WORK_DIR:?source configs/paths.env first}"
REPO_DIR="$(cd "$(dirname "$0")/.." && pwd)"
VLLM_BIN=${VLLM_BIN:-vllm}
RGPU=${RGPU:-$GPU}
PPORT=$((18100 + GPU)) RPORT=$((18200 + RGPU))
LOGS=$WORK_DIR/logs/servers
mkdir -p "$LOGS"
STAMP=$(date +%Y%m%d_%H%M%S)
BASE="$MODELS_DIR/$(${SIM_PY:-python3} -c "import json,sys; print(json.load(open(sys.argv[1]))['policy']['dir'])" "$REPO_DIR/configs/models.json")"
PIDS=()
stop_servers() { for p in "${PIDS[@]}"; do kill "$p" 2>/dev/null; done; sleep 8; for p in "${PIDS[@]}"; do kill -9 "$p" 2>/dev/null; done; PIDS=(); }
trap stop_servers EXIT
wait_up() {  # port pid
  for _ in $(seq 1 180); do curl -sf "http://127.0.0.1:$1/health" > /dev/null && return 0; kill -0 "$2" 2>/dev/null || return 1; sleep 5; done; return 1
}
SHARED=0
[ "$POLICY" = 1 ] && [ -n "$READER" ] && [ "$RGPU" = "$GPU" ] && SHARED=1
export VLLM_NO_USAGE_STATS=1
if [ "$POLICY" = 1 ]; then
  NAME=policy LORA=()
  if [ -n "$ADAPTER" ]; then NAME=$(basename "$ADAPTER"); LORA=(--enable-lora --max-lora-rank 16 --lora-modules "$NAME=$ADAPTER"); fi
  UTIL=0.85; [ "$SHARED" = 1 ] && UTIL=0.55
  CUDA_VISIBLE_DEVICES=$GPU OMP_NUM_THREADS=4 "$VLLM_BIN" serve "$BASE" --served-model-name policy ${LORA[@]+"${LORA[@]}"} --host 127.0.0.1 --port "$PPORT" \
    --gpu-memory-utilization "$UTIL" --max-model-len 16384 --dtype bfloat16 --seed 0 > "$LOGS/policy_${STAMP}.log" 2>&1 &
  PIDS+=($!)
  wait_up "$PPORT" "${PIDS[-1]}" || { echo "policy server did not come up; see $LOGS/policy_${STAMP}.log"; exit 1; }
  export POLICY_MODEL=$NAME POLICY_URL=http://127.0.0.1:$PPORT/v1/chat/completions
  echo "policy $NAME on GPU $GPU port $PPORT ($UTIL of its memory)"
fi
if [ -n "$READER" ]; then
  RUTIL=0.85; [ "$SHARED" = 1 ] && RUTIL=0.35
  CUDA_VISIBLE_DEVICES=$RGPU OMP_NUM_THREADS=4 "$VLLM_BIN" serve "$READER" --served-model-name reader --host 127.0.0.1 --port "$RPORT" \
    --gpu-memory-utilization "$RUTIL" --max-model-len 12288 --dtype bfloat16 --seed 0 --limit-mm-per-prompt '{"image":1}' > "$LOGS/reader_${STAMP}.log" 2>&1 &
  PIDS+=($!)
  wait_up "$RPORT" "${PIDS[-1]}" || { echo "reader server did not come up; see $LOGS/reader_${STAMP}.log"; exit 1; }
  export READER_URL=http://127.0.0.1:$RPORT/v1/chat/completions
  echo "reader $(basename "$READER") on GPU $RGPU port $RPORT ($RUTIL of its memory)"
fi
OMP_NUM_THREADS=1 "$@"
code=$?
stop_servers
exit $code
