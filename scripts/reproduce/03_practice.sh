#!/usr/bin/env bash
# Step 3: one training group, one training seed, two practice rounds. Every round trains a fresh LoRA from Qwen3-8B on
# the demonstrations and every episode the group kept so far (official successes only).
#   clean     practice on the true text view
#   generic   practice on the text view with generic noise
#   rate      practice on the text view with rate-matched noise (fitted on the calibration set)
#   direct    round 1 trains on the accepted episodes of the calibration set's first part; round 2 practises in the
#             real pipeline and keeps the episodes of its first 1,000 reads
#   combined  round 1 practises with noise fitted on the first part (and trains on its accepted episodes too);
#             round 2 as direct
# Text rounds: 29 templates x seed 1017 / 1018 x 4 samples; sampling seeds from --seed-base 1000 x training seed.
# Usage: 03_practice.sh <clean|generic|rate|direct|combined> <training seed 0|1>
source "$(dirname "$0")/common.sh"
GROUP=$1 S=$2
BASE_SEED=$((1000 * S))
KEPT=("$D/demonstrations.jsonl")
case "$GROUP" in
  clean) NOISE=() ;;
  generic) NOISE=(--noise generic) ;;
  rate) NOISE=(--noise "$D/rate_matched_noise.json") ;;
  combined) NOISE=(--noise "$D/rate_matched_noise_first_part.json") ;;
  direct) NOISE=() ;;
  *) echo "unknown group $GROUP"; exit 2 ;;
esac
train() {  # out examples...
  local out=$1; shift
  CUDA_VISIBLE_DEVICES=$GPU OMP_NUM_THREADS=8 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
    "$GPU_PY" -m policy.train_lora "$@" --base "$POLICY_BASE" --out "$out" --seed "$S" | tee "$L/03_train_$(basename "$out").log"
}
text_round() {  # round adapter
  local r=$1 adapter=$2
  RUN=$(run_logged "$L/03_${GROUP}_s${S}_r$r.log" $SERVE --gpu "$GPU" --policy-adapter "$adapter" -- \
    "$SIM_PY" -m policy.episodes text --instances "configs/instances/practice_round_$r.json" --samples 4 --seed-base "$BASE_SEED" \
    --workers 12 ${NOISE[@]+"${NOISE[@]}"} --tag "_${GROUP}_s${S}_r$r")
  "$SIM_PY" -m policy.collect_sft "$D/${GROUP}_s${S}_r$r.jsonl" "$RUN"
  KEPT+=("$D/${GROUP}_s${S}_r$r.jsonl")
}
direct_round_2() {  # adapter
  RUN=$(run_logged "$L/03_${GROUP}_s${S}_r2.log" $SERVE --gpu "$GPU" --policy-adapter "$1" --reader "$READER_BASE" -- \
    "$SIM_PY" -m policy.episodes screenshots --instances configs/instances/practice_round_2.json --samples 6 --seed-base "$BASE_SEED" \
    --workers 12 --purpose train --reader-tag original --tag "_${GROUP}_s${S}_r2")
  "$SIM_PY" -m policy.select budget --run "$RUN" --seed 1018 --samples 6 --budget 1000 --out "$D/${GROUP}_s${S}_r2_kept.json"
  "$SIM_PY" -m policy.select view --runs "$RUN" --episodes "$D/${GROUP}_s${S}_r2_kept.json" --key kept --out "$V/${GROUP}_s${S}_r2"
  "$SIM_PY" -m policy.collect_sft "$D/${GROUP}_s${S}_r2.jsonl" "$V/${GROUP}_s${S}_r2"
  KEPT+=("$D/${GROUP}_s${S}_r2.jsonl")
}
case "$GROUP" in
  clean|generic|rate)
    text_round 1 "$A/start"; train "$A/${GROUP}_s${S}_r1" "${KEPT[@]}"
    text_round 2 "$A/${GROUP}_s${S}_r1"; train "$A/${GROUP}_s${S}_r2" "${KEPT[@]}" ;;
  direct)
    KEPT+=("$D/calibration_first_part.jsonl"); train "$A/${GROUP}_s${S}_r1" "${KEPT[@]}"
    direct_round_2 "$A/${GROUP}_s${S}_r1"; train "$A/${GROUP}_s${S}_r2" "${KEPT[@]}" ;;
  combined)
    KEPT+=("$D/calibration_first_part.jsonl"); text_round 1 "$A/start"; train "$A/${GROUP}_s${S}_r1" "${KEPT[@]}"
    direct_round_2 "$A/${GROUP}_s${S}_r1"; train "$A/${GROUP}_s${S}_r2" "${KEPT[@]}" ;;
esac
echo "policy: $A/${GROUP}_s${S}_r2"
