#!/usr/bin/env bash
# Public B1K training entry; model/training implementation stays in src/.
set -euo pipefail

project_root="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
source_dir="$project_root"
if [[ ! -f "$source_dir/src/v1_groot_memoryvla/experiment/launch_task21_ddp.py" ]]; then
    source_dir="$project_root/deployments/20261110_task0_baseline_resume_5000"
fi
python_path="$project_root/src/v1_groot_memoryvla/.venv/bin/python"
if [[ ! -x "$python_path" ]]; then
    python_path="$project_root/envs/n1d6-p0/bin/python"
fi

train_args=()
show_help=false
if [[ $# -eq 0 ]]; then
    train_args+=(--help)
    show_help=true
fi
while [[ $# -gt 0 ]]; do
    case "$1" in
        --source-dir|--python-path)
            if [[ $# -lt 2 || "$2" == --* ]]; then
                echo "$1 requires a path" >&2
                exit 2
            fi
            if [[ "$1" == --source-dir ]]; then source_dir="$2"; else python_path="$2"; fi
            shift 2
            ;;
        -h|--help) show_help=true; train_args+=(--help); shift ;;
        *) train_args+=("$1"); shift ;;
    esac
done

if [[ ! -f "$source_dir/src/v1_groot_memoryvla/experiment/launch_task21_ddp.py" ]]; then
    echo "Training source missing: $source_dir (use --source-dir)" >&2
    exit 2
fi
if [[ ! -x "$python_path" ]]; then
    echo "Python environment missing: $python_path (use --python-path)" >&2
    exit 2
fi
source_dir="$(cd -- "$source_dir" && pwd)"
python_path="$(cd -- "$(dirname -- "$python_path")" && pwd)/$(basename -- "$python_path")"
if $show_help; then
    echo "Usage: train/run_groot_train.sh [--source-dir DIR] [--python-path PYTHON] TRAIN_ARGS"
    echo "Source: $source_dir"
    echo "Python: $python_path"
    echo "--dry-run prints the launch plan without starting training."
fi

export PYTHONPATH="$source_dir/src${PYTHONPATH:+:$PYTHONPATH}"
export PYTHONNOUSERSITE=1 PYTHONDONTWRITEBYTECODE=1 NO_ALBUMENTATIONS_UPDATE=1
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 OMP_NUM_THREADS="${OMP_NUM_THREADS:-4}"
cd -- "$source_dir"
exec "$python_path" -u -m v1_groot_memoryvla.experiment.launch_task21_ddp "${train_args[@]}"
