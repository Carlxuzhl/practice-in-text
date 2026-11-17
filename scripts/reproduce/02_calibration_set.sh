#!/usr/bin/env bash
# Step 2: the calibration set. The start policy runs in the real pipeline (original reader) on the calibration
# instances (29 templates x seeds 1008-1016), keeping every screenshot with its true text view (--png). The first 2,000
# reads (whole episodes, by seed then template) are the calibration set; its first 1,000 the first part. Then:
# rate-matched noise is checked and fitted on the calibration set (and, for combined training, on the first part), and
# the judge-accepted episodes of the first part become training examples for direct training's first round.
source "$(dirname "$0")/common.sh"
RUN=$(run_logged "$L/02_calibration.log" $SERVE --gpu "$GPU" --policy-adapter "$A/start" --reader "$READER_BASE" -- \
  "$SIM_PY" -m policy.episodes screenshots --instances configs/instances/calibration.json --seed-base 0 --workers 12 \
  --purpose calibration --reader-tag original --png --tag _calibration)
"$SIM_PY" -m policy.select calibration-set --runs "$RUN" --out "$D/calibration_set.json"
"$SIM_PY" -m practice.noise check "$D/noise_check.json" "$RUN" --episodes "$D/calibration_set.json" --key calibration
"$SIM_PY" -m practice.noise fit "$D/rate_matched_noise.json" "$RUN" --episodes "$D/calibration_set.json" --key calibration
"$SIM_PY" -m practice.noise check "$D/noise_check_first_part.json" "$RUN" --episodes "$D/calibration_set.json" --key first_part
"$SIM_PY" -m practice.noise fit "$D/rate_matched_noise_first_part.json" "$RUN" --episodes "$D/calibration_set.json" --key first_part
"$SIM_PY" -m policy.select view --runs "$RUN" --episodes "$D/calibration_set.json" --key first_part --out "$V/calibration_first_part"
"$SIM_PY" -m policy.collect_sft "$D/calibration_first_part.jsonl" "$V/calibration_first_part"
echo "calibration run: $RUN"
