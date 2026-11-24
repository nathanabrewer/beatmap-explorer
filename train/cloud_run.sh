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
# EVAL_SECONDS, WORKERS, MAX_DATA_GB (default: the container's memory cap
# less 24 GB).
#
# Results that don't depend on the launching computer: with RUN_BUCKET and its
# key (AWS_ACCESS_KEY_ID, AWS_SECRET_ACCESS_KEY, RUN_BUCKET_REGION) set, this
# machine pushes $FM_OUT to s3://$RUN_BUCKET/runs/$FM_RUN/ every 5 minutes and
# once more whenever the script ends, and keeps a resumable state (model,
# optimizer, step, time used) under runs/$FM_RUN/state/. RESUME_RUN=<an earlier
# run's name> continues that run's training here; TRAIN_MINUTES stays the
# total for the whole training, across machines.
set -u
here=$(cd "$(dirname "$0")" && pwd)
out="${FM_OUT:-$HOME/fm-out}"
data="${DATA_DIR:-$HOME/data/beatsaver}"
state="${STATE_DIR:-$HOME/fm-state}"
mkdir -p "$out" "$data" "$state"
cd "$here/.."
# The run bucket and its key, staged by stage.py --secrets.
if [ -f train/run.env ]; then set -a; . train/run.env; set +a; fi
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
# Thread pools sized to the CPUs this container may use, not the host's: a
# 208-core host behind a 25-core quota gave the trainer 1,003 threads, and
# every small CPU step then waited on all of them while the GPU sat idle.
pool=$(( cpus < 6 ? cpus : 6 ))
export OMP_NUM_THREADS="$pool" MKL_NUM_THREADS="$pool" OPENBLAS_NUM_THREADS="$pool"
# The container's memory cap (free shows the whole host's). Training reads
# the prepared data through the page cache; data bigger than the cap got a
# run's job supervisor killed an hour into training on a 29 GB machine.
mem_gb=$(python3 -c '
for path in ("/sys/fs/cgroup/memory.max", "/sys/fs/cgroup/memory/memory.limit_in_bytes"):
    try:
        v = open(path).read().strip()
    except OSError:
        continue
    if v != "max" and int(v) < 1 << 50:
        print(int(v) // 2**30)
        break
else:
    print(0)
')
mem_gb="${mem_gb:-0}"
max_data="${MAX_DATA_GB:-}"
if [ -z "$max_data" ] && [ "$mem_gb" -gt 0 ]; then max_data=$(( mem_gb > 40 ? mem_gb - 24 : 16 )); fi
started=$(date +%s)
limit=""; command -v timeout >/dev/null && limit="timeout ${EVAL_SECONDS:-360}"
stamp() { echo "[$(( $(date +%s) - started ))s] $*" | tee -a "$out/timeline.txt"; }
{
  echo "cpus: $cpus"
  (free -g 2>/dev/null | awk '/Mem:/{print "host ram: "$2" GB"}') || true
  echo "memory cap: ${mem_gb:-?} GB"
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
pushers=""
final_push() {
  [ -n "${RUN_BUCKET:-}" ] || return 0
  [ -n "$pushers" ] && kill $pushers 2>/dev/null
  for i in 1 2 3 4 5; do
    python3 train/run_storage.py push "$out" "${FM_RUN:-run}" >> "$out/storage.log" 2>&1 \
      && python3 train/run_storage.py push "$state" "${FM_RUN:-run}/state" >> "$out/storage.log" 2>&1 \
      && { echo "[run_storage] final push done" >> "$out/storage.log"; return 0; }
    sleep 30
  done
  echo "[run_storage] final push FAILED" >> "$out/storage.log"
}
if [ -n "${RUN_BUCKET:-}" ]; then
  python3 train/run_storage.py push "$out" "${FM_RUN:-run}" --every 300 >> "$out/storage.log" 2>&1 &
  pushers="$!"
  python3 train/run_storage.py push "$state" "${FM_RUN:-run}/state" --every 300 >> "$out/storage.log" 2>&1 &
  pushers="$pushers $!"
  trap final_push EXIT
  trap 'final_push; trap - EXIT; exit 143' TERM INT
  stamp "pushing results to s3://$RUN_BUCKET/runs/${FM_RUN:-run}/ every 5 minutes"
fi
resume=""
if [ -n "${RESUME_RUN:-}" ]; then
  mkdir -p "$out/run"
  if python3 train/run_storage.py get "$RESUME_RUN/state/resume.pt" "$state/resume-from.pt" >> "$out/storage.log" 2>&1; then
    python3 train/run_storage.py get "$RESUME_RUN/run/best.pt" "$out/run/best.pt" >> "$out/storage.log" 2>&1 || true
    python3 train/run_storage.py get "$RESUME_RUN/run/log.jsonl" "$out/run/log.jsonl" >> "$out/storage.log" 2>&1 || true
    resume="--resume $state/resume-from.pt"
    stamp "resuming run $RESUME_RUN"
  else
    stamp "could not fetch $RESUME_RUN's state; stopping"; exit 1
  fi
fi
stamp "preparing BeatSaver maps for ${PREP_MINUTES:-14} minutes with $workers workers, at most ${max_data:-unlimited} GB (memory cap ${mem_gb} GB)"
python3 -I train/prepare.py beatsaver --manifest train/dataset/manifest.jsonl --out "$data" --workers "$workers" \
  --best-first --max-minutes "${PREP_MINUTES:-14}" --min-free-gb "${MIN_FREE_GB:-8}" ${max_data:+--max-gb "$max_data"} > "$out/prepare.log" 2>&1
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
  --lr "${LR:-4e-4}" --warmup "${WARMUP:-500}" --workers 8 --eval-every 1500 --max-minutes "${TRAIN_MINUTES:-22}" --save-minutes "${SAVE_MINUTES:-4}" \
  --state "$state" $resume >> "$out/train.log" 2>&1
grep -E '^(==|stopped|done|model|batch [0-9]|[0-9]+ training)' "$out/train.log" | tail -7
stamp "scoring held-out maps"
$limit python3 -I train/evaluate.py --ckpt "$out/run/best.pt" --data "$data" --limit "${EVAL_LIMIT:-8}" \
  --out "$out/run/eval.json" > "$out/eval.log" 2>&1
tail -1 "$out/eval.log"
stamp "done"
