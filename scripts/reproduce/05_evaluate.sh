#!/usr/bin/env bash
# Step 5: evaluate a policy on the confirmation set (29 templates x seeds 3000-3011, one sample, seed base 0), in the
# real pipeline with a given reader, or in text.
# Usage: 05_evaluate.sh <adapter dir> <reader model dir | text> <name> [--sort-reader-output]
#   e.g. 05_evaluate.sh $WORK_DIR/adapters/clean_s0_r2 $WORK_DIR/models/aligned_reader clean_s0_aligned
#        05_evaluate.sh $WORK_DIR/adapters/direct_s0_r2 $MODELS_DIR/Qwen3-VL-4B-Instruct direct_s0_original
#        05_evaluate.sh $WORK_DIR/adapters/clean_s0_r2 text clean_s0_text
source "$(dirname "$0")/common.sh"
ADAPTER=$1 READER=$2 NAME=$3
shift 3
COMMON=(--instances configs/instances/confirmation.json --seed-base 0 --workers 12 --tag "_eval_$NAME")
if [ "$READER" = text ]; then
  RUN=$(run_logged "$L/05_$NAME.log" $SERVE --gpu "$GPU" --policy-adapter "$ADAPTER" -- "$SIM_PY" -m policy.episodes text "${COMMON[@]}")
else
  RUN=$(run_logged "$L/05_$NAME.log" $SERVE --gpu "$GPU" --policy-adapter "$ADAPTER" --reader "$READER" -- \
    "$SIM_PY" -m policy.episodes screenshots "${COMMON[@]}" --purpose eval --reader-tag "$(basename "$READER")" "$@")
fi
echo "$NAME $RUN" | tee -a "$D/evaluation_runs.txt"
