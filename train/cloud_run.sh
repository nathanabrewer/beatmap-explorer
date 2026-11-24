#!/bin/sh
# Runs on the rented GPU as a Foreman script job, from the staged folder
# (train/stage.py): builds the BeatSaver dataset on the machine's own disk,
# trains, evaluates on held-out songs, and leaves checkpoints, logs and scores
# in ~/fm-out, which Foreman copies home while it runs and at the end.
# Settings come from the environment: SIZE, STEPS, BATCH, PREP_WORKERS.
set -eu
here=$(cd "$(dirname "$0")" && pwd)
out="${FM_OUT:-$HOME/fm-out}"
data="${DATA_DIR:-$HOME/data/beatsaver}"
mkdir -p "$out" "$data"
cd "$here/.."
{
  echo "machine: $(nproc) CPUs, $(free -g | awk '/Mem:/{print $2}') GB RAM, disk $(df -h "$HOME" | awk 'NR==2{print $4}') free"
  nvidia-smi --query-gpu=name,memory.total --format=csv,noheader || true
  python3 -c "import torch; print('torch', torch.__version__, 'cuda', torch.version.cuda, 'gpu ok', torch.cuda.is_available())"
} | tee "$out/machine.txt"
python3 -m pip install -q -r train/requirements.txt
python3 -I train/prepare.py beatsaver --manifest train/dataset/manifest.jsonl --out "$data" --workers "${PREP_WORKERS:-12}" 2>&1 | tee "$out/prepare.log"
cp "$data/failed.jsonl" "$out/prepare-failed.jsonl" 2>/dev/null || true
python3 -I train/train.py --data "$data" --out "$out/run" --size "${SIZE:-base}" --steps "${STEPS:-150000}" \
  --batch "${BATCH:-64}" --workers 8 --eval-every 2000 2>&1 | tee "$out/train.log"
python3 -I train/evaluate.py --ckpt "$out/run/best.pt" --data "$data" --limit 150 --out "$out/run/eval.json" 2>&1 | tee "$out/eval.log"
