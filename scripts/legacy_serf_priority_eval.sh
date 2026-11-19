#!/bin/bash
# Legacy SERF/BEHAVIOR priority evaluator. Not used by the GR00T N1.6 queue.
set -eu

PROJECT_ROOT=/nvme3/projects/memory_vla
CONFIG=$PROJECT_ROOT/configs/evaluation/base_config.yaml
LOGDIR=$PROJECT_ROOT/.hermes

TASKS=(
  collecting_childrens_toys
  putting_shoes_on_rack
  assembling_gift_baskets
)

echo "$(date): Priority eval START (task 21/22/26)" > $LOGDIR/eval-priority.log

for task in "${TASKS[@]}"; do
  echo "$(date): === Starting task: $task ===" >> $LOGDIR/eval-priority.log

  sed -i "s/^  name: .*/  name: $task/" "$CONFIG"
  sed -i "s/^eval_instance_ids: .*/eval_instance_ids: [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]/" "$CONFIG"
  echo "$(date): Config patched -> task=$task, instances=[0..9]" >> $LOGDIR/eval-priority.log

  OMNIGIBSON_HEADLESS=true \
  OMNIGIBSON_GPU_ID=0 \
  CUDA_VISIBLE_DEVICES=0 \
  PYTHONDONTWRITEBYTECODE=1 \
  bash $PROJECT_ROOT/scripts/evaluate_b1k.sh >> $LOGDIR/eval-priority.log 2>&1 || \
    echo "$(date): WARNING - eval for $task exited non-zero" >> $LOGDIR/eval-priority.log

  echo "$(date): === Done: $task ===" >> $LOGDIR/eval-priority.log
done

echo "$(date): Priority eval COMPLETE" >> $LOGDIR/eval-priority.log

python3 -c "
import json, csv
from pathlib import Path
root = Path('${PROJECT_ROOT}/outputs/evaluation/legacy_pi_behavior')
results = []
for tname in ['collecting_childrens_toys', 'putting_shoes_on_rack', 'assembling_gift_baskets']:
    metrics = root / tname / 'seed_0' / 'metrics'
    if not metrics.exists(): continue
    scores = []
    for f in sorted(metrics.glob('*_[0-9].json')):
        if 'step_stats' in f.name: continue
        d = json.load(open(f))
        scores.append(d['q_score']['final'])
    if scores:
        avg = sum(scores)/len(scores)
        print(f'{tname}: avg={avg:.2f} ({len(scores)} inst) scores={[int(s*10)/10 for s in scores]}')
" >> $LOGDIR/eval-priority.log 2>&1
