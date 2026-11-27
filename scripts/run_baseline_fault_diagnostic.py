"""Workbench-managed bounded fault reproductions; never launches training."""
import json
import os
from pathlib import Path
import sys
import time
from urllib.request import urlopen

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'evaluation/scripts'))
from groot_inference_launch import InferenceManager
from groot_web_launch import LaunchManager

OUT=ROOT/'Output/test_02/experiment_01/baseline_diagnostic'
CHECKPOINT='/nvme3/data_wyh/outputs/memory_vla/v1_groot_memoryvla/training/v2.1__test01__exp14__run001/run/checkpoints/v2.1__test01__exp14__run001__step006000'


def save(name,value):
    (OUT/name).write_text(json.dumps(value,indent=2,ensure_ascii=False)+'\n')


def services():
    manager=InferenceManager(ROOT)
    jobs=[]
    for index,steps in enumerate((4,8,16)):
        request=dict(checkpoint=CHECKPOINT,gpu_index=index%2,port=5564+index,
                     source_dir='/nvme3/projects/memory_vla/deployments/web_live',
                     python_path='/nvme3/projects/memory_vla/envs/n1d6-p0/bin/python',
                     seed=42,inference_steps=steps,action_diagnostics=True)
        preview=manager.prepare(request)
        save(f'inference{steps}_preview.json',dict(request=request,preview=preview))
        print(json.dumps(dict(inference_steps=steps,preview=preview)),flush=True)
        job=manager.start(request)
        jobs.append(dict(request=request,job=job))
        save('diagnostic_services.json',jobs)
        print(json.dumps(job),flush=True)


def simulation(steps,chunk):
    os.environ['GROOT_ACTION_FAULT_TRACE']='1'
    manager=LaunchManager(ROOT)
    inference=InferenceManager(ROOT)
    manager.policy_resolver=inference.resolve
    request=dict(version='V_2_1',group='ablation_action_fault',mode='dual2',preset='fast-debug',
                 task=40,seed=0,seed_mode='fixed',policy_host='172.28.62.17',
                 policy_port={4:5564,8:5565,16:5566}[steps],debug_max_steps=256,
                 action_steps=chunk,record_video=True,video_steps_per_render=2,
                 run_tag=f'test02_exp01_stage1_step6000_N{steps}_H{chunk}',wait_for_gpu=True)
    preview=manager.prepare(request)
    save(f'N{steps}_H{chunk}_preview.json',dict(request=request,preview=preview))
    print(json.dumps(preview),flush=True)
    job=manager.start(request)
    save(f'N{steps}_H{chunk}_job.json',job)
    print(json.dumps(job),flush=True)


def status():
    """Archive actual owned-job terminal state and current workbench resources."""
    summary=[]
    for path in sorted(OUT.glob('N*_H*_job.json')):
        job=json.loads(path.read_text())
        live=LaunchManager._process_start(job['pid'])==job.get('process_start') and job.get('process_start') is not None
        result=Path(job['output'])/'results.json'
        data=json.loads(result.read_text()) if result.exists() else {}
        job['state']='running' if live else 'complete' if data.get('status') in ('complete','completed') else 'failed'
        job['checked_at']=time.strftime('%Y-%m-%d %H:%M:%S')
        job['result_ready']=result.exists()
        save(path.name,job)
        summary.append(dict(variant=path.stem.removesuffix('_job'),job=job['id'],state=job['state'],
                            steps=data.get('total_action_steps'),episodes=data.get('completed_episodes')))
    with urlopen('http://127.0.0.1:8765/api/services',timeout=20) as response:
        resources=json.load(response)
    save('resource_snapshot.json',dict(checked_at=time.strftime('%Y-%m-%d %H:%M:%S'),services=resources))
    print(json.dumps(summary),flush=True)


def stop_services(steps):
    manager=InferenceManager(ROOT)
    records=json.loads((OUT/'diagnostic_services.json').read_text())
    for item in records:
        if item['request']['inference_steps'] not in steps:continue
        n=item['request']['inference_steps']
        for h in (8,16):
            job=json.loads((OUT/f'N{n}_H{h}_job.json').read_text())
            result=json.loads((Path(job['output'])/'results.json').read_text())
            if LaunchManager._process_start(job['pid'])==job.get('process_start') or result.get('status') not in ('complete','completed'):
                raise RuntimeError('Dependent diagnostic is not complete')
        with urlopen('http://127.0.0.1:8765/api/launch/jobs',timeout=20) as response:
            active=json.load(response)['jobs']
        if any(j.get('state') in ('running','queued','loading') and j.get('policy_port')==item['request']['port'] for j in active):
            raise RuntimeError('Workbench still reports a policy dependency')
        actual=manager.resolve('172.28.62.17',item['request']['port'])
        if actual is None or actual['pid']!=item['job']['pid'] or actual['checkpoint']!=CHECKPOINT:
            raise RuntimeError('Service ownership changed')
        request={k:actual[k] for k in ('pid','port','checkpoint','start_ticks')}
        result=manager.stop(request)
        save(f'inference{n}_stop.json',dict(request=request,result=result,checked_at=time.strftime('%Y-%m-%d %H:%M:%S')))
        print(json.dumps(result),flush=True)


if __name__=='__main__':
    if sys.argv[1]=='services':services()
    elif sys.argv[1]=='simulation':simulation(int(sys.argv[2]),int(sys.argv[3]))
    elif sys.argv[1]=='status':status()
    elif sys.argv[1]=='stop-services':stop_services([int(n) for n in sys.argv[2:]])
    else:raise ValueError('services, status, stop-services STEPS... or simulation STEPS CHUNK')
