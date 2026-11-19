#!/bin/sh
set -eu

# Shared evaluation runtime; keep account-local packages out of the environment.
unset PYTHONPATH PYTHONHOME LD_LIBRARY_PATH LD_PRELOAD ISAAC_PATH CARB_APP_PATH EXP_PATH
export CONDA_PREFIX="${B1K_SIM_PREFIX:-/nvme3/envs/b1k2025-sim}"
export PATH="$CONDA_PREFIX/bin:$PATH"
export PYTHONNOUSERSITE=1 PYTHONUNBUFFERED=1
export OMNI_KIT_ACCEPT_EULA=YES
B1K_PROJECT_ROOT=$(CDPATH= cd -- "$(dirname -- "$0")/.." && pwd)
export OMNIGIBSON_DATA_PATH="${OMNIGIBSON_DATA_PATH:-$B1K_PROJECT_ROOT/assets/simulation}"
export OMNIGIBSON_APPDATA_PATH="${OMNIGIBSON_APPDATA_PATH:-/nvme3/tmp/b1k2025-$(id -u)/appdata}"
export OMNIGIBSON_HEADLESS="${OMNIGIBSON_HEADLESS:-true}"
export OMNIGIBSON_GPU_ID="${OMNIGIBSON_GPU_ID:-1}"
exec "$CONDA_PREFIX/bin/python" -s "$@"
