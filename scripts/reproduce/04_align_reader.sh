#!/usr/bin/env bash
# Step 4: reader alignment. Labels from the calibration set (each screenshot's true text view in the reader's output
# format, layout order), a LoRA on the reader's language model, merged into a full model; then the acceptance test:
# the aligned reader reads the unused episodes' saved screenshots, and its element error rate is compared with the
# original reader's on the same screenshots (at least 30% lower, at most 1% unparsable).
# Usage: 04_align_reader.sh [first N]   (with N, align on the first N calibration screenshots only, e.g. 500)
source "$(dirname "$0")/common.sh"
FIRST=${1:-}
"$SIM_PY" -m reader.align_data --calibration-set "$D/calibration_set.json" --out "$D/align" --first "${FIRST:-500}"
NAME=aligned_reader EXAMPLES=$D/align/train.jsonl
if [ -n "$FIRST" ]; then NAME=aligned_reader_first$FIRST EXAMPLES=$D/align/train_first$FIRST.jsonl; fi
CUDA_VISIBLE_DEVICES=$GPU OMP_NUM_THREADS=8 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
  "$GPU_PY" -m reader.train_lora "$EXAMPLES" --base "$READER_BASE" --out "$A/$NAME" --seed 0 | tee "$L/04_train_$NAME.log"
CUDA_VISIBLE_DEVICES=$GPU OMP_NUM_THREADS=8 "$GPU_PY" -m reader.merge_lora --base "$READER_BASE" --adapter "$A/$NAME" --out "$WORK_DIR/models/$NAME"
CAL=$("$SIM_PY" -c "import json,sys; print(' '.join(json.load(open(sys.argv[1]))['runs']))" "$D/calibration_set.json")
CAL_DIRS=(); for r in $CAL; do CAL_DIRS+=("$WORK_DIR/runs/$r"); done
RUN=$(run_logged "$L/04_reread_$NAME.log" $SERVE --gpu "$GPU" --no-policy --reader "$WORK_DIR/models/$NAME" -- \
  "$SIM_PY" -m reader.reread "$NAME" "${CAL_DIRS[@]}" --reader-tag "$NAME" --episodes "$D/calibration_set.json" --key unused)
"$SIM_PY" -m reader.error_rate --before "${CAL_DIRS[@]}" --after "$RUN" --episodes "$D/calibration_set.json" --key unused --out "$D/acceptance_$NAME.json"
echo "aligned reader: $WORK_DIR/models/$NAME"
