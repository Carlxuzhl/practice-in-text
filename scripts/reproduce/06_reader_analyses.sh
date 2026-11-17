#!/usr/bin/env bash
# Step 6 (CPU only): reader-side results from saved reads.
#   - recurrence of the original reader's misreads on the calibration screenshots, against independent errors at the
#     same rates (the rate-matched noise model), with new-episode predictability on the unused episodes;
#   - optionally the same for a re-read of the calibration screenshots by another reader or at another temperature
#     (reader/reread.py over the whole calibration collection; the control is then fitted on that reader's own reads).
# Usage: 06_reader_analyses.sh [reread run dir ...]
source "$(dirname "$0")/common.sh"
CAL=$("$SIM_PY" -c "import json,sys; print(' '.join(json.load(open(sys.argv[1]))['runs']))" "$D/calibration_set.json")
CAL_DIRS=(); for r in $CAL; do CAL_DIRS+=("$WORK_DIR/runs/$r"); done
"$SIM_PY" -m reader.recurrence --runs "${CAL_DIRS[@]}" --episodes "$D/calibration_set.json" --control "$D/rate_matched_noise.json" \
  --out "$D/recurrence_original.json"
for run in "$@"; do
  "$SIM_PY" -m reader.recurrence --runs "$run" --episodes "$D/calibration_set.json" --out "$D/recurrence_$(basename "$run").json"
done
