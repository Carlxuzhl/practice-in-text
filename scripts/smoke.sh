#!/usr/bin/env bash
# Smoke test in a fresh setup: runs the plumbing end to end on a handful of items, not the study.
# CPU part (always): the tests; the live text view of two instances with the leakage checks; three episodes in the real
#   pipeline with stand-in models (screenshots and true pages kept); the bootstrap's self-test; the analysis tools on
#   those episodes.
# GPU part (--gpu N): serve the reader alone and read 4 of those screenshots; build alignment labels from them, train
#   the reader for one step on 8 examples and merge it; serve the policy (base model) beside the merged reader and run
#   one episode of the confirmation set.
# Needs configs/paths.env sourced and SIM_PY (and for the GPU part GPU_PY, VLLM_BIN and the two base models).
# Starts the simulator if nothing answers at $SIM_URL, and stops it at the end.
# Usage: bash scripts/smoke.sh [--gpu N]
set -euo pipefail
GPU=""
[ "${1:-}" = "--gpu" ] && GPU=$2
: "${WORK_DIR:?source configs/paths.env first}" "${MOBILEGYM_DIR:?}" "${SIM_PY:?set SIM_PY to the simulator environment python}"
REPO_DIR="$(cd "$(dirname "$0")/.." && pwd)"
cd "$REPO_DIR"
export PYTHONPATH="$REPO_DIR${PYTHONPATH:+:$PYTHONPATH}"
OUT=$WORK_DIR/smoke_$(date +%Y%m%d_%H%M%S)
mkdir -p "$OUT"
SIM_PID=""
cleanup() { [ -n "$SIM_PID" ] && kill "$SIM_PID" 2>/dev/null; true; }
trap cleanup EXIT
step() { echo; echo "=== $*"; }
run_dir() { grep '^RUN_DIR' "$1" | tail -1 | awk '{print $2}'; }

step "CPU tests"
"$SIM_PY" -m pytest -q tests 2>&1 | tee "$OUT/pytest.log"

step "simulator"
if ! curl -sf "${SIM_URL:-http://127.0.0.1:14173}" > /dev/null; then
  bash scripts/serve_sim.sh > "$OUT/sim.log" 2>&1 &
  SIM_PID=$!
  for _ in $(seq 1 60); do curl -sf "${SIM_URL:-http://127.0.0.1:14173}" > /dev/null && break; sleep 2; done
fi
curl -sf "${SIM_URL:-http://127.0.0.1:14173}" > /dev/null && echo "simulator up at ${SIM_URL:-http://127.0.0.1:14173}"

step "live text view and leakage checks"
for t in wechat.SetSignature alipay.CheckBalance clock.DeleteAlarm; do
  "$SIM_PY" -m textview.dump --task "$t" --seed 0 --check | tee "$OUT/textview_$t.txt" | tail -3
done

step "three episodes in the real pipeline with stand-in models (no GPU)"
"$SIM_PY" -m policy.episodes screenshots --instances configs/instances/confirmation.json --limit 3 --workers 3 --stub --png \
  --purpose smoke --reader-tag stub --tag _smoke_stub 2>&1 | tee "$OUT/stub.log"
STUB=$(run_dir "$OUT/stub.log")
"$SIM_PY" -m policy.collect_sft "$OUT/stub_examples.jsonl" "$STUB"

step "analysis tools on those episodes"
"$SIM_PY" -m evaluation.bootstrap --self-test > "$OUT/bootstrap_self_test.json" && echo "bootstrap self-test passed"
"$SIM_PY" -m reader.error_rate --runs "$STUB" | head -5
"$SIM_PY" - "$STUB" "$OUT/calibration_smoke.json" <<'PY'
import json, sys
from pathlib import Path
from common.records import rows
run = Path(sys.argv[1])
eps = [{"id": r["id"], "seed": r["seed"], "sample": r.get("sample", 0)} for r in rows(run)]
Path(sys.argv[2]).write_text(json.dumps({"calibration": eps, "unused": eps, "runs": [run.name]}))
PY
"$SIM_PY" -m reader.align_data --calibration-set "$OUT/calibration_smoke.json" --out "$OUT/align" --first 4 | tail -4

if [ -z "$GPU" ]; then echo; echo "SMOKE_CPU_PASSED $OUT"; exit 0; fi

step "GPU: serve the reader and read 4 screenshots"
: "${GPU_PY:?}" "${MODELS_DIR:?}"
READER=$MODELS_DIR/Qwen3-VL-4B-Instruct
bash scripts/with_servers.sh --gpu "$GPU" --no-policy --reader "$READER" -- \
  "$SIM_PY" -m reader.reread smoke "$STUB" --reader-tag smoke --limit 4 --workers 4 2>&1 | tee "$OUT/reread.log"
REREAD=$(run_dir "$OUT/reread.log")
"$SIM_PY" -m reader.error_rate --runs "$REREAD" | head -8

step "GPU: one alignment step on 8 examples, then merge"
N=$(wc -l < "$OUT/align/train.jsonl")
[ "$N" -ge 8 ] || { echo "only $N alignment examples"; exit 1; }
CUDA_VISIBLE_DEVICES=$GPU OMP_NUM_THREADS=8 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
  "$GPU_PY" -m reader.train_lora "$OUT/align/train.jsonl" --base "$READER" --out "$OUT/adapter_smoke" --limit 8 | tee "$OUT/align_train.log"
CUDA_VISIBLE_DEVICES=$GPU OMP_NUM_THREADS=8 "$GPU_PY" -m reader.merge_lora --base "$READER" --adapter "$OUT/adapter_smoke" --out "$OUT/reader_smoke" | tee "$OUT/merge.log"

step "GPU: the policy beside the merged reader, one episode"
bash scripts/with_servers.sh --gpu "$GPU" --reader "$OUT/reader_smoke" -- \
  "$SIM_PY" -m policy.episodes screenshots --instances configs/instances/confirmation.json --limit 1 --workers 1 \
  --purpose smoke --reader-tag smoke_aligned --tag _smoke_gpu 2>&1 | tee "$OUT/episode.log"
grep -q '^RUN_DIR' "$OUT/episode.log"
echo; echo "SMOKE_GPU_PASSED $OUT"
