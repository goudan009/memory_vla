#!/usr/bin/env bash
set -euo pipefail
PROJECT=/nvme3/projects/memory_vla
SOURCE=/nvme3/projects/GR00T/official-baseline-9b37aa1
PYTHON=/nvme3/projects/GR00T/official-baseline-env/bin/python
CHECKPOINT="$PROJECT/outputs/official_baseline_task0_34_40_1000/checkpoint-1000"
case "${1:-}" in
  server)
    export CUDA_VISIBLE_DEVICES="${POLICY_GPU:-1}"
    export PYTHONPATH="$SOURCE"
    export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
    cd "$SOURCE"
    exec "$PYTHON" -u gr00t/eval/run_gr00t_server.py \
      --model-path "$CHECKPOINT" --embodiment-tag BEHAVIOR_R1_PRO \
      --use-sim-policy-wrapper --host 127.0.0.1 --port 5558
    ;;
  eval)
    task="${2:-0}"
    case "$task" in 0|34|40) ;; *) echo 'Task must be 0, 34 or 40' >&2; exit 2 ;; esac
    cd "$PROJECT"
    exec bash evaluation/scripts/run_groot_eval.sh \
      --version V_1 --group official --mode single1 --gpu-index "${SIM_GPU:-0}" \
      --preset official-eval --task "$task" --seed 0 \
      --policy-host 127.0.0.1 --policy-port 5558 \
      --policy-checkpoint "$CHECKPOINT" \
      --training-id official_baseline_task0_34_40_1000
    ;;
  *) echo "Usage: bash $0 server | eval [0|34|40]" >&2; exit 2 ;;
esac
