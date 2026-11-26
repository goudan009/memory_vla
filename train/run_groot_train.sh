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

source_version=v1
train_args=()
show_help=false
if [[ $# -eq 0 ]]; then
    train_args+=(--help)
    show_help=true
fi
while [[ $# -gt 0 ]]; do
    case "$1" in
        --source-version)
            [[ $# -ge 2 ]] || { echo "Missing source version" >&2; exit 2; }
            source_version="$2"; shift 2 ;;
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

case "$source_version" in
    v1) module=v1_groot_memoryvla.experiment.launch_task21_ddp; entry=src/v1_groot_memoryvla/experiment/launch_task21_ddp.py ;;
    v2.1) module=v2_1_groot_memoryvla.train; entry=src/v2_1_groot_memoryvla/train.py ;;
    *) echo "Unknown source version: $source_version" >&2; exit 2 ;;
esac
if [[ ! -f "$source_dir/$entry" ]]; then
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
    echo "Source version: $source_version"
    echo "V2.1: --source-version v2.1 --stage stage1|memory|continued (preview unless --execute)"
    echo "Source: $source_dir"
    echo "Python: $python_path"
    echo "--dry-run prints the launch plan without starting training."
    echo "Exp10: --experiment experiment_10 --fusion-variant gated_eos|dit_attention --fusion-state-lambda FLOAT"
    echo "With dit_attention use --fusion-dit-layers 8 or --fusion-dit-layers 24 (one zero-based block index per run)."
    echo "Use an explicit Task40 split, bounded --steps, and custom fine-tuning with all base components frozen."
fi

export PYTHONPATH="$source_dir/src${PYTHONPATH:+:$PYTHONPATH}"
export PYTHONNOUSERSITE=1 PYTHONDONTWRITEBYTECODE=1 NO_ALBUMENTATIONS_UPDATE=1
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 OMP_NUM_THREADS="${OMP_NUM_THREADS:-4}"
cd -- "$source_dir"
if [[ "$source_version" == v2.1 ]]; then
    args=(); dry=false; execute=false
    for arg in "${train_args[@]}"; do
        case "$arg" in --dry-run) dry=true ;; --execute) execute=true; args+=("$arg") ;; *) args+=("$arg") ;; esac
    done
    if $dry && $execute; then echo "Cannot combine --dry-run and --execute" >&2; exit 2; fi
    train_args=("${args[@]}")
fi
exec "$python_path" -u -m "$module" "${train_args[@]}"
