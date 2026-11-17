#!/usr/bin/env bash
# Step 1: the program oracles record text-view demonstrations of the 29 templates (seeds 1000-1007, CPU only), and the
# start policy is a LoRA on them from Qwen3-8B (one GPU).
source "$(dirname "$0")/common.sh"
RUN=$(run_logged "$L/01_demonstrations.log" "$SIM_PY" -m policy.episodes text --instances configs/instances/demonstrations.json \
  --oracle --workers 3 --tag _demonstrations)
"$SIM_PY" -m policy.collect_sft "$D/demonstrations.jsonl" "$RUN"
CUDA_VISIBLE_DEVICES=$GPU OMP_NUM_THREADS=8 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
  "$GPU_PY" -m policy.train_lora "$D/demonstrations.jsonl" --base "$POLICY_BASE" --out "$A/start" --seed 0 | tee "$L/01_train_start.log"
echo "start policy: $A/start (demonstrations run $RUN)"
