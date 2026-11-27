import json
import os
from pathlib import Path
import subprocess
import sys

import pytest

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'evaluation/scripts'))
from groot_eval_resume import continuation


def fixture(gpus=('0',), counts=(2,3)):
    layout={gpu:[dict(instances=[int(gpu)*5+i*2,int(gpu)*5+i*2+1],quota=10,offset=0)
                 for i in range(2)] for gpu in gpus}
    result=dict(status='interrupted',evaluation_mode='official-eval',task_id=21,requested_episodes=20*len(gpus),
                configuration=dict(mode='dual2' if len(gpus)>1 else 'single2',n_envs_per_gpu=2,
                 instances={gpu:[x for slot in slots for x in slot['instances']] for gpu,slots in layout.items()},
                 episode_layout=layout,preset='official-eval',base_seed=42,seed_mode='sequence',
                 policy_server='172.28.62.17:5556'))
    records=[dict(gpu_index=int(gpu),env_idx=i,valid=True,instance_id=slot['instances'][j%2],seed=42+i+j)
             for gpu,slots in layout.items() for i,slot in enumerate(slots) for j in range(counts[i])]
    result['recorded_episodes']=len(records)
    return result,records


@pytest.mark.parametrize('gpus',[('0',),('0','1')])
@pytest.mark.parametrize('amount',[1,3,17,100])
def test_exact_additional_budget_preserves_prefix_and_original_input(gpus,amount):
    result,records=fixture(gpus)
    original=json.dumps(result,sort_keys=True)
    plan=continuation(result,records,amount)
    assert plan['additional_episodes']==amount
    assert plan['requested_episodes']==len(records)+amount
    assert sum(s['quota'] for slots in plan['child_layout'].values() for s in slots)==amount
    for gpu,slots in plan['child_layout'].items():
        for slot in slots:
            index=slot['env_index']
            assert slot['offset']==sum(r['gpu_index']==int(gpu) and r['env_idx']==index for r in records)
            assert slot['instances']==result['configuration']['episode_layout'][gpu][index]['instances']
    assert json.dumps(result,sort_keys=True)==original


def test_second_continuation_and_completed_run():
    result,records=fixture()
    first=continuation(result,records,1)
    gpu,slots=next(iter(first['child_layout'].items()));slot=slots[0]
    records.append(dict(gpu_index=int(gpu),env_idx=slot['env_index'],valid=True,
                        instance_id=slot['instances'][slot['offset']%len(slot['instances'])]))
    result.update(requested_episodes=first['requested_episodes'],status='complete')
    result['configuration']['episode_layout']=first['episode_layout']
    with pytest.raises(ValueError,match='原计划已完成'):continuation(result,records)
    second=continuation(result,records,11)
    assert second['requested_episodes']==len(records)+11


@pytest.mark.parametrize('gpus',[('0',),('0','1')])
@pytest.mark.parametrize('status',['interrupted','cancelled','complete'])
def test_cli_accepts_one_more_episode_for_single_and_dual(tmp_path,gpus,status):
    result,records=fixture(gpus)
    result.update(status=status,resume_allowed=False)
    (tmp_path/'results.json').write_text(json.dumps(result))
    (tmp_path/'episodes.jsonl').write_text(''.join(json.dumps(r)+'\n' for r in records))
    proc=subprocess.run(['bash',str(ROOT/'evaluation/scripts/run_groot_eval.sh'),'--resume-from',str(tmp_path),
                         '--additional-episodes','1','--dry-run'],capture_output=True,text=True)
    assert proc.returncode==0,proc.stderr
    plan,_=json.JSONDecoder().raw_decode(proc.stdout)
    assert plan['additional_episodes']==1 and plan['requested_episodes']==len(records)+1
    assert len(plan['commands'])==1
    command=plan['commands'][0]
    assert command[command.index('--n-envs')+1]=='1'
    assert command[command.index('--episodes')+1]=='1'
    child=subprocess.run(command+['--dry-run'],capture_output=True,text=True)
    assert child.returncode==0,child.stderr


def test_original_environment_identity_survives_single_slot_continuation(tmp_path):
    from groot_eval_results import EpisodeWriter
    meta=dict(task_id=21,task_name='collecting_childrens_toys',episodes=2,
              episode_layout=[dict(instances=[5,6,7,8,9],offset=3,quota=2,env_index=1)])
    (tmp_path/'run_meta.json').write_text(json.dumps(meta))
    writer=EpisodeWriter(tmp_path)
    assert writer.record_env_idx==1
    writer.append(instance_id=8,success=False,valid=True,task_progress=20,sim_steps=8,seed=46)
    writer.append(instance_id=9,success=False,valid=True,task_progress=20,sim_steps=8,seed=47)
    records=[json.loads(line) for line in (tmp_path/'episodes.jsonl').read_text().splitlines()]
    assert [r['env_idx'] for r in records]==[1,1]
    assert [r['episode_in_slot'] for r in records]==[4,5]


@pytest.mark.parametrize('saved_layout', [True, False])
def test_finish_original_budget_with_one_remaining_episode(tmp_path, saved_layout):
    result, records = fixture(('0',), counts=(10,9))
    if not saved_layout:
        result['configuration'].pop('episode_layout')
    (tmp_path/'results.json').write_text(json.dumps(result))
    (tmp_path/'episodes.jsonl').write_text(''.join(json.dumps(r)+'\n' for r in records))
    proc = subprocess.run(['bash', str(ROOT/'evaluation/scripts/run_groot_eval.sh'),
                           '--resume-from', str(tmp_path), '--dry-run'], capture_output=True, text=True)
    assert proc.returncode == 0, proc.stderr
    plan, _ = json.JSONDecoder().raw_decode(proc.stdout)
    assert plan['additional_episodes'] == 1 and plan['requested_episodes'] == 20
    command = plan['commands'][0]
    assert command[command.index('--n-envs')+1] == '1'
    slot = json.loads(command[command.index('--episode-layout')+1])[0]
    assert slot['env_index'] == 1 and slot['offset'] == 9 and slot['quota'] == 1


def test_resume_engineering_snapshot_only_changes_bookkeeping(tmp_path):
    from groot_device_eval_remote import resume_source
    original=tmp_path/'original';live=tmp_path/'live'
    for root in (original,live): (root/'evaluation/scripts').mkdir(parents=True)
    for name in ('run_groot_eval.sh','groot_eval_resume.py','groot_eval_results.py'):
        (live/'evaluation/scripts'/name).write_text('current '+name)
    (original/'evaluation/scripts/scientific.py').write_text('frozen physics')
    (original/'web_source_manifest.json').write_text('{"digest":"original"}')
    (tmp_path/'deployments').mkdir()
    source=resume_source(tmp_path,original,live)
    assert (source/'evaluation/scripts/scientific.py').read_text()=='frozen physics'
    assert (source/'evaluation/scripts/groot_eval_resume.py').read_text()=='current groot_eval_resume.py'
    assert not (original/'evaluation/scripts/groot_eval_resume.py').exists()
    manifest=json.loads((source/'web_source_manifest.json').read_text())
    assert manifest['continuation_engineering']['original_source']==str(original)
    assert resume_source(tmp_path,original,live)==source
