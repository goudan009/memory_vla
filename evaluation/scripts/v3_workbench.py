"""V3 stages for the existing BenchmarkManager/worker, including GPU0 queue.

The manager owns all processes and records. Waiting never allocates CUDA or
touches GPU1/VR. Invoke preview before start; an existing HTTP server can still
show and stop the jobs in its shared runs directory without a restart.
"""
import argparse
import json
import os
from pathlib import Path
import shlex
import subprocess
import time
from benchmark_dashboard import atomic_json, local_gpus, read_json, BenchmarkManager


KINDS={'v3_preflight':'V3 · 真实数据工程门禁','v3_train':'V3 · 单轮1000步角色检索训练',
       'v3_evaluate':'V3 · 100窗口七模式与12组闭环配对'}
GPU_UUID='GPU-49fbbb98-a702-2adb-f148-a55b95f77252'


def prepare(manager,request):
    if set(request)-{'kind','gpu','notes','preflight','training_job'}:
        raise ValueError('Unsupported V3 workbench arguments')
    kind=request['kind']
    if kind not in KINDS or request.get('gpu',0)!=0:
        raise ValueError('V3 is authorized on GPU0 only; GPU1 is reserved for VR')
    actual=subprocess.check_output(['nvidia-smi','--id=0','--query-gpu=uuid','--format=csv,noheader'],text=True).strip()
    if actual!=GPU_UUID:
        raise ValueError('GPU0 identity changed; do not risk allocating the VR GPU')
    root=manager.root
    original=Path('/nvme3/projects/memory_vla')
    evidence=root/'Output/test_03/experiment_01/evidence'
    if read_json(evidence/'audit.json',{}).get('passed') is not True:
        raise ValueError('V3 independent asset and role audit not completed')
    py=str(manager.python)
    checkpoint=original/'checkpoints/v1__test01__exp11__step006000'
    dataset=original/'assets/datasets/b1k2025_groot_task40_v1_200'
    config=dict(kind=kind,gpu=0,gpu_uuid=GPU_UUID,model_version='v3',test_id='test03',experiment_id='exp01',
        run_id='run001',notes=request.get('notes',''),code_root=str(root),asset_root=str(original),
        category='engineering_selfcheck' if kind=='v3_preflight' else 'exploratory_validation')
    stages=[dict(name='V3 · GPU0工作台资源队列（保留GPU1 VR）',argv=[py,str(Path(__file__)),
        'wait','--root',str(root),'--run','{run}'],report='resource_ready.json')]
    common=['--checkpoint',str(checkpoint),'--dataset',str(dataset),'--evidence',str(evidence)]
    if kind in ('v3_preflight','v3_train'):
        phase='preflight' if kind=='v3_preflight' else 'train'
        args=[py,'-m','v3_groot_role_memory.training',*common,'--phase',phase,
              '--output','{run}/'+phase+'.json']
        if kind=='v3_train':
            proof=Path(request.get('preflight',''))
            if not proof.is_file() or read_json(proof,{}).get('passed') is not True:
                raise ValueError('Training requires archived completed V3 preflight proof')
            config.update(steps=1000,preflight=str(proof))
            args+=['--preflight',str(proof)]
        else:
            config['steps']=4
        stages.append(dict(name=KINDS[kind],argv=args,report=phase+'.json',steps=config['steps']))
    else:
        training=manager._directory(request.get('training_job'))
        trained=read_json(training/'train.json',{})
        if trained.get('passed') is not True or trained.get('steps')!=1000:
            raise ValueError('Evaluate only the complete 1000-step final V3 checkpoint')
        final=Path(trained['checkpoint'])
        if not (final/'COMPLETE.json').is_file():
            raise ValueError('Final checkpoint completion identity missing')
        config.update(training_job=training.name,checkpoint=str(final))
        protocol=evidence/'evaluation_protocol.json'
        stages+=[dict(name='V3 · 100固定窗口七模式机制诊断',argv=[py,'-m','v3_groot_role_memory.evaluation',
            '--checkpoint',str(final),'--dataset',str(dataset),'--protocol',str(protocol),
            '--output','{run}/mechanism.json'],report='mechanism.json'),
            dict(name='V3 · 12组V1/V3同起点配对闭环',argv=[py,str(root/'evaluation/scripts/v3_task40_paired_evaluation.py'),
            '--root',str(original),'--checkpoint',str(final),'--protocol',str(protocol),'--port','5597',
            '--output','{run}/paired.json'],report='paired.json')]
    return dict(config=config,title=KINDS[kind],stages=stages,gpu=next(x for x in local_gpus() if x['index']==0),
        required_free_mb=36000,ready=True,queue_before_cuda=True,commands=[shlex.join(s['argv']) for s in stages])


def wait(root,run):
    manager=BenchmarkManager(root)
    logs=run/'watchdog_log.jsonl'
    while True:
        own=read_json(run/'plan.json')
        rows=manager.snapshot()['jobs']
        # Existing same-device work has priority, including its not-yet-started
        # GPU stages. Later queued V3 work must not deadlock the earlier job.
        blockers=[r for r in rows if r['id']!=run.name and r.get('config',{}).get('gpu',0)==0
            and r['status'] in ('running','starting','stopping')
            and (not r['kind'].startswith('v3_') or r['created_at']<own['created_at'])]
        device=next(x for x in local_gpus() if x['index']==0)
        now=time.time()
        sample=dict(timestamp=now,blockers=[dict(id=r['id'],stage=r.get('stage'),status=r['status']) for r in blockers],
            gpu0=device,gpu_uuid=GPU_UUID,next_check_at=now+30,reason='same-GPU existing work and verified peak-memory margin')
        with logs.open('a') as stream:
            stream.write(json.dumps(sample,ensure_ascii=False)+'\n')
        atomic_json(run/'watchdog_state.json',sample)
        if not blockers and device['free_mb']>=36000:
            atomic_json(run/'resource_ready.json',dict(passed=True,**sample))
            return
        print(json.dumps(dict(queue_waiting=True,**sample),ensure_ascii=False),flush=True)
        if (run/'stop_requested').exists():
            raise SystemExit('Workbench stop requested while queued')
        time.sleep(30)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action',choices=['preview','start','wait'])
    parser.add_argument('--root',type=Path,required=True)
    parser.add_argument('--run',type=Path)
    parser.add_argument('--kind',choices=list(KINDS))
    parser.add_argument('--preflight')
    parser.add_argument('--training-job')
    args=parser.parse_args()
    if args.action=='wait':
        wait(args.root,args.run);return
    request=dict(kind=args.kind,gpu=0,notes='V3 user-authorized GPU0 fallback; preserve GPU1 VR; plan-v002')
    if args.preflight:request['preflight']=args.preflight
    if args.training_job:request['training_job']=args.training_job
    manager=BenchmarkManager(args.root)
    result=manager.prepare(request) if args.action=='preview' else manager.start(request)
    print(json.dumps(result,ensure_ascii=False,indent=2))


if __name__=='__main__':
    main()
