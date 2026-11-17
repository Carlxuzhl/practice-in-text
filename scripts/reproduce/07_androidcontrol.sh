#!/usr/bin/env bash
# Step 7 (optional): the two reader-side measurements on AndroidControl. Download the data first
# (bash androidcontrol/download.sh, about 10 GB; not redistributed here).
#  1. CPU: which screens can be sampled (androidcontrol.pool), then the sample (androidcontrol.sample): about 2,000
#     alignment and 1,000 held-out screenshots, each with its text view and training label.
#  2. GPU: a smoke check (8 alignment screenshots, at least 7 must parse), then the original reader reads both sets.
#  3. GPU: reader alignment on the alignment set (the same recipe as MobileGym's), merged; the aligned reader reads the
#     held-out set.
#  4. CPU: the two measurements (recurrence, alignment), the descriptive breakdown and the box-width check.
source "$(dirname "$0")/common.sh"
POOL=$(run_logged "$L/07_ac_pool.log" "$SIM_PY" -m androidcontrol.pool --parts 4)
SAMPLE=$(run_logged "$L/07_ac_sample.log" "$SIM_PY" -m androidcontrol.sample "$POOL")
READ=("$SIM_PY" -m androidcontrol.read)
$SERVE --gpu "$GPU" --no-policy --reader "$READER_BASE" -- \
  "${READ[@]}" "$SAMPLE/reads_smoke" "$SAMPLE/manifest.jsonl" --purpose ac_smoke --reader-tag original --set align --limit 8
n=$("$SIM_PY" -c "import json, sys; from reader.reader import parse_reader; print(sum(1 for l in open(sys.argv[1]) if parse_reader(json.loads(l)['raw'])[0]['elements']))" "$SAMPLE/reads_smoke/reads.jsonl")
[ "$n" -ge 7 ] || { echo "only $n of 8 smoke reads parse; stopping"; exit 1; }
$SERVE --gpu "$GPU" --no-policy --reader "$READER_BASE" -- \
  "${READ[@]}" "$SAMPLE/reads_original" "$SAMPLE/manifest.jsonl" --purpose ac_original --reader-tag original --workers 12
CUDA_VISIBLE_DEVICES=$GPU OMP_NUM_THREADS=8 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
  "$GPU_PY" -m reader.train_lora "$SAMPLE/align_examples.jsonl" --base "$READER_BASE" --out "$A/ac_aligned_reader" --seed 0 | tee "$L/07_ac_train.log"
CUDA_VISIBLE_DEVICES=$GPU OMP_NUM_THREADS=8 "$GPU_PY" -m reader.merge_lora --base "$READER_BASE" --adapter "$A/ac_aligned_reader" --out "$WORK_DIR/models/ac_aligned_reader"
$SERVE --gpu "$GPU" --no-policy --reader "$WORK_DIR/models/ac_aligned_reader" -- \
  "${READ[@]}" "$SAMPLE/reads_aligned" "$SAMPLE/manifest.jsonl" --purpose ac_aligned --reader-tag ac_aligned --set heldout --workers 12
"$SIM_PY" -m androidcontrol.analyze "$SAMPLE" all
"$SIM_PY" -m androidcontrol.breakdown "$SAMPLE"
"$SIM_PY" -m androidcontrol.boxcheck "$SAMPLE"
echo "AndroidControl results in $SAMPLE (results_all.json, breakdown.json, boxcheck.json)"
