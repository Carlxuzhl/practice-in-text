#!/usr/bin/env bash
# Download the AndroidControl split files and shards 0-3 of the training data from the public bucket, and check them
# against the bucket's md5. The data is not redistributed with this code; check its licence in the AndroidControl
# repository (github.com/google-research/google-research/tree/master/android_control) before you use it.
# About 10 GB. Usage: bash androidcontrol/download.sh   (into $ANDROIDCONTROL_DIR/raw, default $WORK_DIR/androidcontrol/raw)
set -euo pipefail
: "${WORK_DIR:?source configs/paths.env first}"
D=${ANDROIDCONTROL_DIR:-$WORK_DIR/androidcontrol}/raw
B=https://storage.googleapis.com/gresearch/android_control
mkdir -p "$D"
cd "$D"
for f in splits.json test_subsplits.json android_control-0000{0,1,2,3}-of-00020; do
  if [ -s "$f" ]; then echo "skip $f (exists)"; continue; fi
  curl -fL --retry 3 --retry-delay 10 -sS -o "$f.part" "$B/$f"
  mv "$f.part" "$f"
  echo "done $f"
done
cd - > /dev/null
"${SIM_PY:-python}" -c "from androidcontrol import data as L; r = L.check_md5(); print(r); raise SystemExit(0 if all(r.values()) else 1)"
