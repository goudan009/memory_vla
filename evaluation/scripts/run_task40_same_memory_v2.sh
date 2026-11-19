#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/../.."
source_run=${1:?provide the validated source run}
weights=${2:?provide the MemoryVLA checkpoint directory}
cycles=${3:-60}
port=${TASK40_PORT:-5572}
groot_root=${TASK40_GROOT_ROOT:-/nvme3/projects/GR00T}
base=Output/test_01
mkdir -p "$base"
for ((number=1; ; number++)); do
    printf -v run '%s/%s/experiment_%02d' "$PWD" "$base" "$number"
    if mkdir "$run" 2>/dev/null; then break; fi
    [[ -d "$run" ]]
done
echo "$run"
mkdir "$run"/{logs,metrics,videos,checkpoints}
cp -r "$source_run"/{anchor.json,start.json,sim_checkpoint.pt,memory_online.pt,policy_rng.pt,prefix} "$run/checkpoints/"
cp "$source_run/original/0009/observation.npz" "$run/checkpoints/anchor_live_observation.npz"
printf 'source=%s\nweights=%s\ncycles=%s\nport=%s\n' "$(realpath "$source_run")" "$(realpath "$weights")" "$cycles" "$port" > "$run/logs/config.txt"

env PYTHONPATH="$PWD/src" CUDA_VISIBLE_DEVICES="${TASK40_POLICY_GPU:-1}" HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 \
    "$groot_root/official-baseline-env/bin/python" -u evaluation/scripts/task40_replay_policy.py \
    --checkpoint "$(realpath "$weights")" --source-run "$run/checkpoints" --output "$run/metrics" --port "$port" \
    > "$run/logs/policy.log" 2>&1 &
policy_pid=$!
trap 'kill "$policy_pid" 2>/dev/null || true; wait "$policy_pid" 2>/dev/null || true' EXIT
for ((attempt=0; attempt<180; attempt++)); do
    if rg -q 'TASK40_POLICY_READY' "$run/logs/policy.log"; then break; fi
    kill -0 "$policy_pid"
    sleep 1
done
rg -q 'TASK40_POLICY_READY' "$run/logs/policy.log"

for repeat in 1 2 3; do
    printf -v log 'repeat_%02d.log' "$repeat"
    env -u OMNIGIBSON_DATA_PATH OMNIGIBSON_GPU_ID="${TASK40_SIM_GPU:-0}" bash scripts/b1k_python.sh \
        evaluation/scripts/task40_replay_probe.py --output "$run" --source-run "$run/checkpoints" \
        --repeat-number "$repeat" --max-prefix 0 --cycles "$cycles" --port "$port" \
        --behavior "$groot_root/BEHAVIOR-1K-task-progress" --groot "$groot_root/Isaac-GR00T-n1d6-9b37aa1" \
        > "$run/logs/$log" 2>&1
done
python3 evaluation/scripts/task40_same_memory_v2_audit.py "$run"
