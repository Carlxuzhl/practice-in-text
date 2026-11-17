# Sourced by the reproduction scripts. Needs configs/paths.env sourced, and:
#   SIM_PY   python of the simulator environment (requirements/sim.txt)
#   GPU_PY   python of the GPU environment (requirements/gpu.txt); VLLM_BIN its vllm executable
#   GPU      the GPU to use (one A6000 48 GB holds the policy and the reader side by side)
set -euo pipefail
: "${WORK_DIR:?source configs/paths.env first}" "${MODELS_DIR:?}" "${MOBILEGYM_DIR:?}" "${SIM_PY:?}" "${GPU_PY:?}" "${GPU:?}"
REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$REPO_DIR"
export PYTHONPATH="$REPO_DIR${PYTHONPATH:+:$PYTHONPATH}"
D=$WORK_DIR/data
A=$WORK_DIR/adapters
V=$WORK_DIR/views
L=$WORK_DIR/logs
mkdir -p "$D" "$A" "$V" "$L"
POLICY_BASE=$MODELS_DIR/Qwen3-8B
READER_BASE=$MODELS_DIR/Qwen3-VL-4B-Instruct
SERVE="bash scripts/with_servers.sh"

# run_logged LOG COMMAND...: run, keep the output, and print the run directory the command reported (RUN_DIR line).
run_logged() {
  local log=$1; shift
  "$@" 2>&1 | tee "$log" >&2
  grep '^RUN_DIR' "$log" | tail -1 | awk '{print $2}'
}
