"""Check data-dependent recommendations without launching a GPU job."""
import json
from pathlib import Path
import shutil
import subprocess

import pytest

ROOT = Path(__file__).resolve().parents[1]
NODE = shutil.which('node') or '/home/1024201092WYH/.nvm/versions/node/v22.23.1/bin/node'


def recommend(**changes):
    if not Path(NODE).is_file():
        pytest.skip('Node.js is required for recommendation checks')
    request = dict(datasets=[dict(task=task, training_profile=dict(total_episodes=200,
        total_windows=total, train_windows=train, split_seed=42, validation_percent=10))
        for task,total,train in [(0,1780,1606),(34,1962,1775),(40,2631,2368)]],
        gpus=[dict(index=i,total_mb=81920) for i in range(2)],
        checkpoint=dict(training_profile=dict(kind='base',memory_enabled=False)))
    request.update(changes)
    script = "const {recommendTraining}=require('./train/dashboard.js'); process.stdout.write(JSON.stringify(recommendTraining(JSON.parse(require('fs').readFileSync(0,'utf8')))));"
    result = subprocess.run([NODE, '-e', script], cwd=ROOT, input=json.dumps(request),
        capture_output=True, text=True, check=True)
    return json.loads(result.stdout)


@pytest.mark.parametrize('cards,steps,windows', [(1,18000,4),(2,18000,4),(3,24000,3),(4,18000,4)])
def test_three_task_budget_tracks_global_batch_not_gpu_count(cards, steps, windows):
    result = recommend(gpus=[dict(index=i,total_mb=81920) for i in range(cards)])
    assert result['supported']
    params = result['values']
    assert params['steps'] == steps and params['global_windows_per_step'] == windows
    assert windows % cards == 0
    assert params['microbatch'] == 4 and params['learning_rate'] == 1e-5
    assert (params['memory_conditioning'], params['memory_length']) == ('residual',64)
    assert params['disable_early_stopping'] and params['validation_interval'] == 3000
    assert [r['train_episodes'] for r in result['tasks']] == [180,180,180]
    assert not any(r['estimated'] for r in result['tasks'])


def test_backbone_start_and_memory_start_have_distinct_budgets():
    behavior = recommend(checkpoint=dict(training_profile=dict(kind='behavior')))
    assert behavior['values']['steps'] == 11000
    trained = recommend(checkpoint=dict(training_profile=dict(kind='trained',memory_enabled=True,
        memory_conditioning='residual', memory_length=256)))
    assert trained['values']['steps'] == 8000
    assert trained['values']['learning_rate'] == 5e-6
    assert trained['values']['memory_conditioning'] == 'residual'
    assert trained['values']['memory_length'] == 256
    assert recommend(arm='baseline')['values'] == recommend(arm='memory_v1')['values']


def test_data_split_and_smallest_gpu_change_recommendation():
    assert recommend(validation_enabled=False)['values']['steps'] == 20000
    changed = recommend(validation_percent=20)
    assert changed['values']['steps'] == 16000 and all(r['estimated'] for r in changed['tasks'])
    assert all(r['estimated'] for r in recommend(seed=123)['tasks'])
    assert recommend(gpus=[dict(total_mb=81920),dict(total_mb=40960)])['values']['microbatch'] == 2


@pytest.mark.parametrize('mode', ['full', 'custom', 'lora'])
def test_backbone_modes_recommend_small_microbatch(mode):
    result = recommend(fine_tuning_mode=mode)
    assert result['supported'] and result['values']['microbatch'] == 1
    assert result['values']['learning_rate'] == (2e-6 if mode == 'full' else 1e-5)


@pytest.mark.parametrize('change', [dict(datasets=[]),dict(gpus=[]),dict(fine_tuning_mode='unknown'),
    dict(checkpoint=None),dict(gpus=[dict(total_mb=24576)]),dict(datasets=[dict(task=1)]),
    dict(validation_percent=0)])
def test_missing_or_unsupported_inputs_do_not_invent_runnable_recommendations(change):
    result = recommend(**change)
    assert not result['supported'] and result['reason']
    assert 'values' not in result


def test_late_response_from_previous_device_is_identified_for_discard():
    source=(ROOT/'train/dashboard.js').read_text()
    api=source[source.index('  async function api('):source.index('  function message(')]
    script="""
const assert=require('node:assert/strict');
let trainingDevice='a100',finish;
global.fetch=()=>new Promise(resolve=>finish=resolve);
"""+api+"""
(async()=>{
  const pending=api('options');trainingDevice='a6000';
  finish({ok:true,json:async()=>({host:'a100'})});
  await assert.rejects(pending,error=>error.name==='DeviceChanged');
})().catch(error=>{console.error(error);process.exitCode=1;});
"""
    result=subprocess.run([NODE],input=script,text=True,capture_output=True)
    assert result.returncode==0,result.stderr
