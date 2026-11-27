"""Training-only supervisor for the user-authorized V2_1 Test02/Exp01 restart.

GPU mutations use the existing workbench. No evaluation or inference dispatch.
"""
import fcntl
import hashlib
import json
import math
from datetime import datetime, timedelta
from pathlib import Path
import subprocess
import sys
from urllib.parse import urlencode
from urllib.request import ProxyHandler, Request, build_opener

ROOT = Path(__file__).resolve().parents[1]
EXPERIMENT = ROOT/'Output/test_02/experiment_01'
DIRECTORY = EXPERIMENT/'evidence/training_watch'
RUN = 'v2_1__test02__exp01__run001'
PREFLIGHT = RUN+'__preflight001'
ARM = 'baseline'
STAGE = 'continued'
REQUEST_FILE = 'continued-preflight-request.json'
REMOTE_EVIDENCE = '/nvme3/data_wyh/outputs/memory_vla/v2_1/test02/exp01/evidence'
ACTIVE = {'queued','waiting_for_gpus','submitted','initializing','training','validating','running'}
OPENER = build_opener(ProxyHandler({}))


def now():
    return datetime.now().astimezone()


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    pending=path.with_suffix(path.suffix+'.pending')
    pending.write_text(json.dumps(value,indent=2,ensure_ascii=False)+'\n')
    pending.replace(path)


def api(method, route, body=None):
    req=Request('http://127.0.0.1:8765/api/training/'+route,
                data=None if body is None else json.dumps(body).encode(),
                headers={'Content-Type':'application/json'},method=method)
    with OPENER.open(req,timeout=55) as response:
        result=json.load(response)
    if result.get('error'):
        raise RuntimeError(result['error'])
    return result


def source_digest():
    root=ROOT/'src/v2_1_groot_memoryvla';h=hashlib.sha256()
    for p in sorted(root.rglob('*')):
        if p.is_file() and p.suffix in ('.py','.json') and '__pycache__' not in p.parts:
            h.update(str(p.relative_to(root)).encode());h.update(p.read_bytes())
    return h.hexdigest()


def detail(run):
    return api('GET','detail?'+urlencode(dict(device='a100',run=run,arm=ARM)))


def mutation(state, name, route, payload, preview=False):
    entry=state.setdefault('actions',{}).get(name)
    if entry:
        if entry['status']=='done':return entry['result']
        raise RuntimeError('Uncertain prior mutation; inspect jobs and journal: '+name)
    if preview:
        write(DIRECTORY/(name+'-preview.json'),api('POST','preview',payload))
    state['actions'][name]=dict(status='sending',at=now().isoformat(),payload=payload)
    write(DIRECTORY/'state.json',state)
    result=api('POST',route,payload)
    state['actions'][name].update(status='done',result=result)
    write(DIRECTORY/(name+'-result.json'),result)
    write(DIRECTORY/'state.json',state)
    return result


def remote_inventory(config):
    sys.path.insert(0,str(ROOT/'train'))
    from dashboard import TrainingManager
    manager=TrainingManager(ROOT);manager.close()
    code=('import json,subprocess,shutil\nfrom pathlib import Path\n'
          'p=Path('+repr(config['argv'][config['argv'].index('--output')+1])+')\n'
          'live=[]\n'
          'for file in Path("/proc").glob("[0-9]*/cmdline"):\n'
          ' try:\n'
          '  argv=file.read_bytes().decode().strip("\\0").split("\\0")\n'
          '  if ("--output" in argv and argv[argv.index("--output")+1]==str(p)) or ("--controller" in argv and argv[argv.index("--controller")+1]==str(p.parent/"config/v21_controller.json")):live.append(int(file.parent.name))\n'
          ' except (OSError,UnicodeError,IndexError):pass\n'
          'print(json.dumps(dict(free_bytes=shutil.disk_usage(p.parent).free,live_pids=live,'
          'stop=json.loads((p/"stop.json").read_text()) if (p/"stop.json").exists() else None,'
          'gpu=subprocess.check_output(["nvidia-smi","--query-gpu=index,uuid,memory.used,utilization.gpu",'
          '"--format=csv,noheader"],text=True))))\n')
    return json.loads(manager._ssh('python3 -',code,timeout=30))


def archive(pre):
    sys.path.insert(0,str(ROOT/'train'))
    from dashboard import TrainingManager
    manager=TrainingManager(ROOT);manager.close()
    code=('import json\nfrom pathlib import Path\n'
          'p=Path('+repr(pre['config']['argv'][pre['config']['argv'].index('--output')+1])+')\n'
          'dest=Path('+repr(REMOTE_EVIDENCE)+')\ndest.mkdir(parents=True,exist_ok=True)\n'
          'out={}\n'
          'for name in ["preflight.json","resume_diagnostics.json","data_split.json","config.json"]:\n'
          ' value=json.loads((p/name).read_text());out[name]=value\n'
          ' target=dest/('+repr(PREFLIGHT+'-')+'+name)\n'
          ' pending=target.with_suffix(".pending");pending.write_text(json.dumps(value,ensure_ascii=False,indent=2));pending.replace(target)\n'
          'print(json.dumps(dict(path=str(dest/'+repr(PREFLIGHT+'-preflight.json')+'),files=out)))\n')
    result=json.loads(manager._ssh('python3 -',code,timeout=30))
    write(DIRECTORY/'preflight-archive.json',result)
    return result['path']


def tick(state):
    if state.get('status') in ('blocked','completed','user_paused'):return
    jobs={j['run']:j for j in api('GET','jobs?device=a100')['jobs']}
    if RUN not in jobs:
        if not state.get('gate'):
            pre=detail(PREFLIGHT)
            state.update(phase=STAGE+'_preflight',observed=pre['state'])
            if pre['state']['status'] in ACTIVE:return
            write(DIRECTORY/'preflight-detail.json',pre)
            if pre['state']['status']!='completed' or not (pre.get('preflight') or {}).get('passed'):
                raise RuntimeError('Real '+STAGE+' preflight failed; evidence retained')
            if pre['preflight']['source_digest']!=source_digest():
                raise RuntimeError('Source changed after preflight; formal start held')
            state['gate']=archive(pre)
            state['source_digest']=pre['preflight']['source_digest']
        if state['source_digest']!=source_digest():
            raise RuntimeError('Current source differs from certified preflight')
        if PREFLIGHT in jobs:
            pre=detail(PREFLIGHT)
            mutation(state,'purge-preflight','delete',dict(device='a100',run=PREFLIGHT,arm=ARM,
                delete_files=True,expected_output=jobs[PREFLIGHT]['output'],purge_preflight=True,evidence_path=state['gate']))
        request=json.loads((EXPERIMENT/REQUEST_FILE).read_text())
        request.update(run_id=RUN,preflight_only=False,accept_preflight=state['gate'],
                       notes='User authorized V2_1 Test02/Exp01 '+STAGE+' formal3000; exact real-preflight source; no evaluation dispatch')
        mutation(state,'start-formal','start',request,preview=True)
        state.update(phase=STAGE+'_initializing',status='running')
        return
    current=detail(RUN)
    state.update(observed=current['state'],phase=STAGE+'_training')
    write(DIRECTORY/'latest-detail.json',current)
    resources=remote_inventory(current['config']);write(DIRECTORY/'resources.json',resources)
    status=current['state']['status'];step=current['state'].get('step',0)
    if state.get('last_step')!=step:state['last_progress_at']=now().isoformat()
    state['last_step']=step
    rows=current.get('metrics',[])
    finite=all(math.isfinite(r[k]) for r in rows for k in ('fm','state','grad_norm') if isinstance(r.get(k),(float,int)))
    if not finite:
        if status in ACTIVE:mutation(state,'nonfinite-stop','stop',dict(device='a100',run=RUN,arm=ARM))
        raise RuntimeError('Nonfinite metrics; stop requested through workbench')
    proof=state.setdefault('resume_proof',{})
    if status=='completed':
        if step!=3000 or not any(c['step']==3000 for c in current['checkpoints']):
            raise RuntimeError('Completion missing exact step3000 checkpoint')
        state.update(status='completed',completed_at=now().isoformat())
        return
    if status in ('failed','cancelled'):raise RuntimeError('Formal training '+status)
    if not proof:
        if resources['stop'] or status in ('stopped','paused'):
            state['status']='user_paused';return
        if status=='training' and step>=1:
            target=step+5
            proof.update(target=target,requested_at=now().isoformat())
            mutation(state,'pause-proof','stop',dict(device='a100',run=RUN,arm=ARM,step=target))
    elif not proof.get('resumed'):
        if resources['stop'] and resources['stop'].get('stop_after_step')!=proof['target']:
            state['status']='user_paused';return
        if status in ('stopped','paused'):
            if resources['live_pids']:return
            payload=dict(device='a100',run=RUN,arm=ARM,preview=True)
            preview=api('POST','resume',payload);write(DIRECTORY/'resume-preview.json',preview)
            if not preview.get('resume_step')==proof['target']:
                raise RuntimeError('Unexpected pause checkpoint')
            payload.pop('preview')
            result=mutation(state,'resume-proof','resume',payload)
            proof.update(resumed=True,result=result)
    elif status in ('stopped','paused'):
        state['status']='user_paused'
    elif step>proof['target']:
        proof.update(next_update_verified=True,observed_step=step)


def main():
    DIRECTORY.mkdir(parents=True,exist_ok=True)
    with (DIRECTORY/'lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        file=DIRECTORY/'state.json'
        state=json.loads(file.read_text()) if file.exists() else dict(status='running',run=RUN)
        try:tick(state)
        except Exception as error:
            state.update(status='blocked',error=repr(error))
        state.update(checked_at=now().isoformat(),next_check_at=(now()+timedelta(seconds=120)).isoformat())
        if state['status'] in ('blocked','completed','user_paused'):state['next_check_at']=None
        write(file,state)
        with (DIRECTORY/'log.jsonl').open('a') as stream:stream.write(json.dumps(state,ensure_ascii=False)+'\n')
        print(json.dumps(state,ensure_ascii=False))
        other=EXPERIMENT/'evidence'/('training_watch' if STAGE=='memory' else 'memory_training_watch')/'state.json'
        other_state=json.loads(other.read_text()) if other.exists() else {}
        if state['next_check_at'] is None and other_state.get('status') in ('blocked','completed','user_paused'):
            subprocess.run(['systemctl','--user','stop','memory-vla-v2-1-training.timer'],check=True,timeout=15)


if __name__=='__main__':
    if sys.argv[1:]==['--memory']:
        RUN='v2_1__test02__exp01__run002'
        PREFLIGHT=RUN+'__preflight001'
        ARM='memory_v1'
        STAGE='memory'
        REQUEST_FILE='memory-preflight-request.json'
        DIRECTORY=EXPERIMENT/'evidence/memory_training_watch'
    elif sys.argv[1:]:
        raise SystemExit('Only --memory or no arguments are supported')
    main()
