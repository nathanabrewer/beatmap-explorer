#!/bin/sh
# Runs on the rented GPU as a Foreman script job, from the staged folder
# (train/stage.py). Time-boxed to fit the run's hour: builds part of the
# BeatSaver dataset on the machine's own disk (best maps first), trains for
# the time left, scores held-out maps, and leaves checkpoints, logs and
# scores in $FM_OUT (default ~/fm-out), which Foreman copies home every few
# minutes and at the end. Settings (environment): PREP_MINUTES, TRAIN_MINUTES,
# EVAL_LIMIT, SIZE, BATCH, WORKERS.
set -u
here=$(cd "$(dirname "$0")" && pwd)
out="${FM_OUT:-$HOME/fm-out}"
data="${DATA_DIR:-$HOME/data/beatsaver}"
mkdir -p "$out" "$data"
cd "$here/.."
cpus=$( (nproc || sysctl -n hw.ncpu) 2>/dev/null)
workers="${WORKERS:-$(( cpus < 16 ? cpus : 16 ))}"
started=$(date +%s)
limit=""; command -v timeout >/dev/null && limit="timeout 360"
stamp() { echo "[$(( $(date +%s) - started ))s] $*" | tee -a "$out/timeline.txt"; }
{
  echo "cpus: $cpus"
  (free -g 2>/dev/null | awk '/Mem:/{print "ram: "$2" GB"}') || true
  df -h "$HOME" | awk 'NR==2{print "disk free: "$4}'
  nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader 2>/dev/null || true
  python3 -c "import torch; print('torch', torch.__version__, 'cuda', torch.version.cuda, 'gpu', torch.cuda.is_available())"
} > "$out/machine.txt" 2>&1
cat "$out/machine.txt"
stamp "installing requirements"
python3 -m pip install -q -r train/requirements.txt 2>&1 | tail -3
stamp "preparing BeatSaver maps for ${PREP_MINUTES:-14} minutes with $workers workers"
python3 -I train/prepare.py beatsaver --manifest train/dataset/manifest.jsonl --out "$data" --workers "$workers" \
  --best-first --max-minutes "${PREP_MINUTES:-14}" > "$out/prepare.log" 2>&1
tail -2 "$out/prepare.log"
cp "$data/failed.jsonl" "$out/prepare-failed.jsonl" 2>/dev/null || true
stamp "prepared $(ls "$data" | grep -c '\.json\.gz$') songs"
stamp "training for ${TRAIN_MINUTES:-22} minutes"
python3 -I train/train.py --data "$data" --out "$out/run" --size "${SIZE:-small}" --steps 1000000 --batch "${BATCH:-64}" \
  --lr 4e-4 --warmup 500 --workers 8 --eval-every 1500 --max-minutes "${TRAIN_MINUTES:-22}" --save-minutes 4 > "$out/train.log" 2>&1
grep -E '^(==|stopped|done|model|[0-9]+ training)' "$out/train.log" | tail -6
stamp "scoring held-out maps"
$limit python3 -I train/evaluate.py --ckpt "$out/run/best.pt" --data "$data" --limit "${EVAL_LIMIT:-8}" \
  --out "$out/run/eval.json" > "$out/eval.log" 2>&1
tail -1 "$out/eval.log"
stamp "done"
