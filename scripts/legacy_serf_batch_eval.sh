#!/bin/bash
# Legacy SERF/BEHAVIOR batch evaluator. Not used by the GR00T N1.6 queue.
set -eu

PROJECT_ROOT=/nvme3/projects/memory_vla
CONFIG=$PROJECT_ROOT/configs/evaluation/base_config.yaml
LOGDIR=$PROJECT_ROOT/.hermes

TASKS=(
  turning_on_radio
  picking_up_trash
  putting_away_Halloween_decorations
  cleaning_up_plates_and_food
  can_meat
  setting_mousetraps
  hiding_Easter_eggs
  picking_up_toys
  rearranging_kitchen_furniture
  putting_up_Christmas_decorations_inside
  set_up_a_coffee_station_in_your_kitchen
  putting_dishes_away_after_cleaning
  preparing_lunch_box
  loading_the_car
  carrying_in_groceries
  bringing_in_wood
  moving_boxes_to_storage
  bringing_water
  tidying_bedroom
  outfit_a_basic_toolbox
  sorting_vegetables
  collecting_childrens_toys
  putting_shoes_on_rack
  boxing_books_up_for_storage
  storing_food
  clearing_food_from_table_into_fridge
  assembling_gift_baskets
  sorting_household_items
  getting_organized_for_work
  clean_up_your_desk
  setting_the_fire
  clean_boxing_gloves
  wash_a_baseball_cap
  wash_dog_toys
  hanging_pictures
  attach_a_camera_to_a_tripod
  clean_a_patio
  clean_a_trumpet
  spraying_for_bugs
  spraying_fruit_trees
  make_microwave_popcorn
  cook_cabbage
  chop_an_onion
  slicing_vegetables
  chopping_wood
  cook_hot_dogs
  cook_bacon
  freeze_pies
  canning_food
  make_pizza
)

echo "$(date): Batch evaluation START" > $LOGDIR/batch-eval.log
echo "$(date): serve should be running on port 8000 (GPU 1)" >> $LOGDIR/batch-eval.log

if ! nc -z 127.0.0.1 8000 2>/dev/null; then
  echo "$(date): ERROR - serve not running on port 8000" >> $LOGDIR/batch-eval.log
  exit 1
fi

for task in "${TASKS[@]}"; do
  echo "$(date): === Starting task: $task ===" >> $LOGDIR/batch-eval.log

  # 用 sed 改 config: task name 和 eval_instance_ids
  sed -i "s/^  name: .*/  name: $task/" "$CONFIG"
  sed -i "s/^eval_instance_ids: .*/eval_instance_ids: [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]/" "$CONFIG"
  echo "$(date): Config patched -> task=$task, instances=[0..9]" >> $LOGDIR/batch-eval.log

  # 跑评估
  OMNIGIBSON_HEADLESS=true \
  OMNIGIBSON_GPU_ID=0 \
  CUDA_VISIBLE_DEVICES=0 \
  PYTHONDONTWRITEBYTECODE=1 \
  bash $PROJECT_ROOT/scripts/evaluate_b1k.sh >> $LOGDIR/batch-eval.log 2>&1 || \
    echo "$(date): WARNING - eval for $task exited with non-zero" >> $LOGDIR/batch-eval.log

  # 汇总当前任务结果
  RESULT_DIR=$PROJECT_ROOT/outputs/evaluation/legacy_pi_behavior/$task/seed_0/metrics
  if [ -d "$RESULT_DIR" ]; then
    echo "$(date): $task results:" >> $LOGDIR/batch-eval.log
    for f in $RESULT_DIR/*_[0-9].json; do
      [ -f "$f" ] && python3 -c "
import json,sys
d=json.load(open(sys.argv[1]))
print('  %s: Q=%s steps=%s' % (sys.argv[1].split('/')[-1], d['q_score']['final'], d['time']['simulator_steps']))
" "$f" >> $LOGDIR/batch-eval.log 2>/dev/null || true
    done
  fi
  echo "$(date): === Done: $task ===" >> $LOGDIR/batch-eval.log
done

echo "$(date): Batch evaluation COMPLETE" >> $LOGDIR/batch-eval.log

# 汇总所有结果到 CSV
python3 -c "
import json, csv
from pathlib import Path
root = Path('${PROJECT_ROOT}/outputs/evaluation/legacy_pi_behavior')
results = []
for task_dir in sorted(root.iterdir()):
    metrics = task_dir / 'seed_0' / 'metrics'
    if not metrics.exists():
        continue
    for f in sorted(metrics.glob('*_[0-9].json')):
        if 'step_stats' in f.name:
            continue
        try:
            d = json.load(open(f))
            q = d['q_score']['final']
            steps = d['time']['simulator_steps']
            sim_time = d['time']['simulator_time']
            instance_id = f.stem.split('_')[-2]
            results.append((task_dir.name, instance_id, q, steps, round(sim_time,1)))
        except:
            pass
out = Path('${PROJECT_ROOT}/outputs/baseline_results.csv')
with open(out, 'w', newline='') as fp:
    w = csv.writer(fp)
    w.writerow(['task','instance_id','q_score','steps','sim_time'])
    for r in results:
        w.writerow(r)
print('Wrote %d results to %s' % (len(results), out))
" >> $LOGDIR/batch-eval.log 2>&1

echo "$(date): Results saved to $PROJECT_ROOT/outputs/baseline_results.csv" >> $LOGDIR/batch-eval.log
