#!/usr/bin/env bash
set -euo pipefail
PROJECT=/nvme3/projects/memory_vla
SOURCE=/nvme3/projects/GR00T/official-baseline-9b37aa1
PYTHON=/nvme3/projects/GR00T/official-baseline-env/bin/python
TRAIN_GPU="${GROOT_TRAIN_GPU:-1}"
FREE_MIB=$(nvidia-smi --id="$TRAIN_GPU" --query-gpu=memory.free --format=csv,noheader,nounits)
if (( FREE_MIB < 36864 )); then
  echo "GPU$TRAIN_GPU has $FREE_MIB MiB free; this official training configuration requires at least 36864 MiB free." >&2
  exit 1
fi
export CUDA_VISIBLE_DEVICES="$TRAIN_GPU"
export PYTHONPATH="$SOURCE"
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export TOKENIZERS_PARALLELISM=false
export OMP_NUM_THREADS=8
export NO_ALBUMENTATIONS_UPDATE=1
export GROOT_COMMIT_HASH=9b37aa1ce69c73c6d165233fa88128283bba4508
cd "$SOURCE"
exec "$PYTHON" -u "$PROJECT/train/official_baseline/launch.py"
