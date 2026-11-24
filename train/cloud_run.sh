#!/bin/sh
# Runs on the rented GPU as a Foreman script job, from the staged folder
# (train/stage.py). Time-boxed to fit the run's hour: builds part of the
# BeatSaver dataset on the machine's own disk (best maps first), trains for
# the time left, scores held-out maps, and leaves checkpoints, logs and
# scores in $FM_OUT (default ~/fm-out), which Foreman copies home every few
# minutes and at the end. Settings (environment): PREP_MINUTES, MIN_FREE_GB
# (preparing stops when the disk has this little left, so a small disk holds
# the best maps that fit), TRAIN_MINUTES, SIZE, BATCH (the most; train.py
# lowers it if the worst case does not fit the card), LR, WARMUP, SAVE_MINUTES
# (large checkpoints are about 0.8 GB, each one copied home), EVAL_LIMIT,
# EVAL_SECONDS, WORKERS.
set -u
here=$(cd "$(dirname "$0")" && pwd)
out="${FM_OUT:-$HOME/fm-out}"
data="${DATA_DIR:-$HOME/data/beatsaver}"
mkdir -p "$out" "$data"
cd "$here/.."
# CPUs the container may use: its cgroup quota (nproc and os.cpu_count can
# report 1 or the whole host instead), capped at 24 workers so the downloads
# stay a modest load on BeatSaver's server.
cpus=$(python3 - <<'PY'
import os
def quota():
    try:
        q, p = open("/sys/fs/cgroup/cpu.max").read().split()[:2]
        if q != "max": return int(q) / int(p)
    except (OSError, ValueError): pass
    try:
        q = int(open("/sys/fs/cgroup/cpu/cpu.cfs_quota_us").read()); p = int(open("/sys/fs/cgroup/cpu/cpu.cfs_period_us").read())
        if q > 0: return q / p
    except (OSError, ValueError): pass
    return os.cpu_count() or 1
print(max(1, int(quota())))
PY
)
workers="${WORKERS:-$(( cpus < 24 ? cpus : 24 ))}"
started=$(date +%s)
limit=""; command -v timeout >/dev/null && limit="timeout ${EVAL_SECONDS:-360}"
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
# Images built on a distribution Python refuse pip without this flag (PEP 668);
# this machine is thrown away after the run.
python3 -m pip install -q --break-system-packages -r train/requirements.txt > "$out/pip.log" 2>&1 \
  || python3 -m pip install -q -r train/requirements.txt >> "$out/pip.log" 2>&1
if ! python3 -c "import librosa, soundfile, numpy, torch" 2>>"$out/pip.log"; then
  stamp "requirements did not install; see pip.log"; tail -5 "$out/pip.log"; exit 1
fi
stamp "preparing BeatSaver maps for ${PREP_MINUTES:-14} minutes with $workers workers"
python3 -I train/prepare.py beatsaver --manifest train/dataset/manifest.jsonl --out "$data" --workers "$workers" \
  --best-first --max-minutes "${PREP_MINUTES:-14}" --min-free-gb "${MIN_FREE_GB:-8}" > "$out/prepare.log" 2>&1
tail -2 "$out/prepare.log"
du -sh "$data" | awk '{print "prepared data: "$1}' | tee -a "$out/timeline.txt"
df -h "$data" | awk 'NR==2{print "disk free: "$4}' | tee -a "$out/timeline.txt"
cp "$data/failed.jsonl" "$out/prepare-failed.jsonl" 2>/dev/null || true
songs=$(ls "$data" | grep -c '\.json\.gz$')
stamp "prepared $songs songs"
if [ "$songs" -lt "${MIN_SONGS:-50}" ]; then
  stamp "too few songs to train on; stopping (see prepare-failed.jsonl)"; exit 1
fi
stamp "training for ${TRAIN_MINUTES:-22} minutes"
# Fewer, growable memory blocks: batches change shape every step.
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True python3 -I train/train.py --data "$data" --out "$out/run" --size "${SIZE:-small}" --steps 1000000 --batch "${BATCH:-64}" \
  --lr "${LR:-4e-4}" --warmup "${WARMUP:-500}" --workers 8 --eval-every 1500 --max-minutes "${TRAIN_MINUTES:-22}" --save-minutes "${SAVE_MINUTES:-4}" > "$out/train.log" 2>&1
grep -E '^(==|stopped|done|model|batch [0-9]|[0-9]+ training)' "$out/train.log" | tail -7
stamp "scoring held-out maps"
$limit python3 -I train/evaluate.py --ckpt "$out/run/best.pt" --data "$data" --limit "${EVAL_LIMIT:-8}" \
  --out "$out/run/eval.json" > "$out/eval.log" 2>&1
tail -1 "$out/eval.log"
stamp "done"
