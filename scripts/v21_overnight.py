"""Persistent client of the existing workbench; fixed Exp14 plan-v002 only.

Each timer tick advances at most one external mutation. GPU processes are
created exclusively by the workbench's preview/start/resume interfaces.
"""
from datetime import datetime, timedelta
import argparse
import fcntl
import json
import math
from pathlib import Path
import sys
from urllib.error import HTTPError
from urllib.parse import urlencode
from urllib.request import ProxyHandler, Request, build_opener

ROOT = Path(__file__).resolve().parents[1]
DIRECTORY = ROOT / 'Output/test_01/experiment_14/evidence/automation'
TRAINING = '/nvme3/data_wyh/outputs/memory_vla/v1_groot_memoryvla/training'
ARCHIVE = '/nvme3/data_wyh/outputs/memory_vla/v2_1/experiment14/evidence/automation'
RUNS = {stage: f'v2.1__test01__exp14__run{number:03d}'
        for stage, number in [('stage1', 1), ('memory', 2), ('continued', 3)]}
ACTIVE = {'queued', 'waiting_for_gpus', 'submitted', 'initializing', 'training', 'validating', 'running'}


def now():
    return datetime.now().astimezone()


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    pending = path.with_suffix(path.suffix + '.pending')
    pending.write_text(json.dumps(value, ensure_ascii=False, indent=2) + '\n')
    pending.replace(path)


class Workbench:
    def __init__(self):
        self.opener = build_opener(ProxyHandler({}))
        sys.path.insert(0, str(ROOT / 'train'))
        from dashboard import TrainingManager
        self.training = TrainingManager(ROOT)
        self.training.close()  # one-shot RPC helper; no extra background scheduler

    def api(self, method, path, body=None):
        request = Request('http://127.0.0.1:8765' + path,
                          data=json.dumps(body).encode() if body is not None else None,
                          headers={'Content-Type': 'application/json'}, method=method)
        try:
            with self.opener.open(request, timeout=540) as response:
                result = json.load(response)
        except HTTPError as error:
            raise RuntimeError(f'{method} {path}: {error.read().decode()}') from error
        if isinstance(result, dict) and result.get('error'):
            raise RuntimeError(result['error'])
        return result

    def remote(self, code):
        return json.loads(self.training._ssh('python3 -', code, timeout=90))

    def resources(self, run):
        return self.remote(
            'import json,shutil,subprocess\nfrom pathlib import Path\n'
            f'p=Path({(TRAINING + "/" + run)!r})\n'
            'stop=p/"run/stop.json"\n'
            'events=sorted((p/"config").glob("resume-event-*.json"))\n'
            'live=[]\n'
            'for f in Path("/proc").glob("[0-9]*/cmdline"):\n'
            ' try:\n'
            '  a=f.read_bytes().decode().strip("\\0").split("\\0")\n'
            '  if any(flag in a and a[a.index(flag)+1]==str(target) for flag,target in '
            '[("--output",p/"run"),("--controller",p/"config/v21_controller.json")]):live.append(int(f.parent.name))\n'
            ' except (OSError,UnicodeError,IndexError):pass\n'
            'print(json.dumps(dict(free_bytes=shutil.disk_usage(p.parent).free, '
            'stop=json.loads(stop.read_text()) if stop.exists() else None, '
            'resume_events=[json.loads(f.read_text()) for f in events], live=live, '
            'gpu=subprocess.check_output(["nvidia-smi","--query-gpu=index,uuid,memory.used,utilization.gpu",'
            '"--format=csv,noheader"],text=True))))\n')

    def archive(self, run):
        return self.remote(
            'import json\nfrom pathlib import Path\n'
            f'p=Path({(TRAINING + "/" + run)!r}); dest=Path({ARCHIVE!r})/{run!r}\n'
            'dest.mkdir(parents=True,exist_ok=True)\n'
            'names=["run/preflight.json","run/calibration.json","run/resume_diagnostics.json",'
            '"run/data_split.json","config/web_launch.json","config/source_manifest.json"]\n'
            'saved={}\n'
            'for name in names:\n'
            ' f=p/name\n'
            ' if f.is_file():\n'
            '  value=json.loads(f.read_text()); target=dest/f.name\n'
            '  pending=target.with_suffix(".pending"); pending.write_text(json.dumps(value,ensure_ascii=False,indent=2)+"\\n"); pending.replace(target)\n'
            '  saved[f.name]=value\n'
            'log=p/"logs/controller.log"\n'
            'if log.exists():(dest/"controller-tail.log").write_text(log.read_text(errors="replace")[-50000:])\n'
            'print(json.dumps(dict(path=str(dest/"preflight.json"),files=saved)))\n')

    def freeze_serving(self):
        synced = self.api('POST','/api/training/sync',dict(device='a100',once=True))
        if synced.get('state')!='synced' or not synced.get('digest'):
            raise RuntimeError('Serving source synchronization not complete')
        digest = synced['digest']
        return self.remote(
            'import hashlib,json,shutil\nfrom pathlib import Path\n'
            'p=Path("/nvme3/projects/memory_vla/deployments/web_live")\n'
            'm=json.loads((p/"web_source_manifest.json").read_text())\n'
            f'assert m["digest"]=={digest!r}\n'
            'dest=p.parent/("inference_"+m["digest"])\n'
            'if not dest.exists():shutil.copytree(p,dest,symlinks=True)\n'
            'for name,sha in m["files"].items():\n'
            ' assert hashlib.sha256((dest/name).read_bytes()).hexdigest()==sha, name\n'
            'assert (dest/"src/v2_1_groot_memoryvla/baseline_policy.py").is_file()\n'
            'print(json.dumps(dict(source=str(dest),digest=m["digest"],files=len(m["files"]))))\n')


class Chain:
    def __init__(self, workbench, state, save, evidence=write):
        self.w, self.state, self.save, self.evidence = workbench, state, save, evidence

    def block(self, reason):
        self.state.update(status='blocked', reason=reason)
        self.save()

    def mutation(self, key, path, body, preview=None):
        """Journal before sending. An ambiguous start is reconciled, never resent."""
        actions = self.state.setdefault('actions', {})
        if key in actions:
            raise RuntimeError(f'Unresolved prior mutation {key}; inspect workbench before retry')
        if preview:
            self.evidence(DIRECTORY / (key + '-preview.json'), self.w.api('POST', preview, body))
        actions[key] = dict(status='sending', at=now().isoformat(), path=path, request=body)
        self.save()
        result = self.w.api('POST', path, body)
        actions[key].update(status='returned', result=result)
        self.save()
        self.evidence(DIRECTORY / (key + '-result.json'), result)
        return result

    def detail(self, run, arm):
        detail = self.w.api('GET', '/api/training/detail?' + urlencode(dict(device='a100',run=run,arm=arm)))
        state = detail['state']
        metrics = detail.get('metrics', [])
        last = state.get('last') or {}
        if any(isinstance(last.get(k), (int,float)) and not math.isfinite(last[k])
               for k in ('fm', 'state', 'grad_norm')):
            if state.get('status') in ACTIVE:
                key = run + '-nonfinite-stop'
                if key not in self.state.get('actions', {}):
                    self.mutation(key, '/api/training/stop', dict(device='a100',run=run,arm=arm))
            self.block(f'{run}: nonfinite training metrics; workbench stop requested')
        seconds = [r['seconds'] for r in metrics if r.get('seconds',0)>0]
        mean = sum(seconds)/len(seconds) if seconds else None
        item = self.state.setdefault('observed', {}).setdefault(run, {})
        if item.get('step') != state.get('step'):
            item['last_progress_at'] = now().isoformat()
        item.update(status=state.get('status'),step=state.get('step'),checked_at=now().isoformat(),
                    seconds_per_step=mean, remaining_update_hours=(detail['config']['steps']-state.get('step',0))*mean/3600 if mean else None)
        self.save()
        return detail

    def request(self, stage, preflight, gate=None):
        run = RUNS[stage] + ('__preflight001' if preflight else '')
        result = dict(device='a100',version='v2.1',task=40,stage=stage,run_id=run,
                      dataset='/nvme3/data_wyh/datasets/b1k2025_groot_task40_v1_200',
                      labels='/nvme3/data_wyh/datasets/v2_1_task40_labels/task40_labels.parquet',
                      checkpoint=self.state['checkpoints']['stage1']['checkpoint'],gpus=[0,1,2,3],
                      steps=3000,seed=42,microbatch=2,learning_rate=1e-5,global_windows_per_step=2,
                      weak_weight=.5,state_weight=.01,prior='topology',fusion_layers=0,
                      preflight_only=preflight,enqueue=True,
                      notes='plan-v002 unattended fixed V2.1 training; engineering preflight' if preflight
                      else 'plan-v002 fixed V2.1 formal training, persistent workbench timer')
        if gate: result['accept_preflight'] = gate
        return result

    def complete(self, stage, detail):
        budget = 6000 if stage=='stage1' else 3000
        if detail['state'].get('status') != 'completed': return False
        checkpoints = [c for c in detail['checkpoints'] if c['step']==budget]
        if detail['state'].get('step') != budget or len(checkpoints)!=1:
            self.block(f'{stage}: completed status without exact COMPLETE/training_state at {budget}')
            return False
        self.state.setdefault('checkpoints', {})[stage] = dict(name=stage,run=detail['run'],arm=detail['arm'],
            checkpoint=checkpoints[0]['path'],source=detail['config']['source'])
        self.save()
        return True

    def training(self, stage, jobs):
        slot = self.state.setdefault('stages', {}).setdefault(stage, {})
        run = RUNS[stage]
        arm = 'memory_v1' if stage=='memory' else 'baseline'
        pre = run + '__preflight001'
        if not slot.get('gate'):
            if pre not in jobs:
                key = pre + '-start'
                if key in self.state.get('actions', {}):
                    self.block(f'{pre}: submission result uncertain or job removed; inspect journal')
                else:
                    self.state.update(status='running',phase=stage+'_preflight')
                    self.mutation(key, '/api/training/start', self.request(stage, True), '/api/training/preview')
                return
            detail = self.detail(pre, arm)
            if self.state.get('status')=='blocked': return
            status = detail['state'].get('status')
            if status in ACTIVE: return
            if status!='completed' or not (detail.get('preflight') or {}).get('passed'):
                archive = self.w.archive(pre)
                self.evidence(DIRECTORY / (pre + '-failed.json'), archive)
                self.block(f'{pre}: real preflight did not pass; outputs retained for debugging')
                return
            archive = self.w.archive(pre)
            if not archive['files']['preflight.json']['passed']:
                self.block(f'{pre}: archived preflight not passed'); return
            self.evidence(DIRECTORY / (pre + '-archive.json'), archive)
            slot['gate'] = archive['path']
            self.save()
            return
        if not slot.get('purged'):
            if pre in jobs:
                # The workbench rechecks terminal state, no live ranks and no consumers.
                resources = self.w.resources(pre)
                if resources['live']: return
                self.mutation(pre+'-purge', '/api/training/delete', dict(device='a100',run=pre,arm=arm,
                    delete_files=True,expected_output=TRAINING+'/'+pre,purge_preflight=True,evidence_path=slot['gate']))
            slot['purged'] = True
            self.save()
            return
        if run not in jobs:
            key = run+'-start'
            if key in self.state.get('actions', {}):
                self.block(f'{run}: submission uncertain or job removed; inspect journal')
            else:
                self.state.update(status='running',phase=stage+'_train')
                self.mutation(key, '/api/training/start', self.request(stage, False, slot['gate']), '/api/training/preview')
            return
        detail = self.detail(run, arm)
        if self.state.get('status')=='blocked': return
        status = detail['state'].get('status')
        resources = self.w.resources(run)
        self.state['resources'] = dict(run=run,at=now().isoformat(),**resources)
        if resources['free_bytes']<50*2**30:
            self.state['disk_warning'] = 'Less than 50GiB free; investigate before next stage'
        # Resume validation is owned only by this explicit pause, never user pauses.
        proof = slot.setdefault('resume_proof', {})
        if not proof:
            if resources['stop']:
                self.block(f'{run}: existing user stop request; automatic progression held'); return
            if status in ('initializing','training','validating','running'):
                # Remote queries and journaling may outlive one update.
                # Reserve a small future boundary without changing total budget.
                target = detail['state'].get('step',0)+10
                proof.update(target=target,requested_at=now().isoformat())
                self.save()
                self.mutation(run+'-pause', '/api/training/stop', dict(device='a100',run=run,arm=arm,step=target))
            elif status not in ACTIVE:
                self.block(f'{run}: stopped before scheduler-owned resume validation')
            return
        if not proof.get('resumed'):
            if resources['stop'] and resources['stop'].get('stop_after_step')!=proof['target']:
                self.block(f'{run}: stop point changed by user; automatic progression held'); return
            if status=='stopped' and not resources['live']:
                if detail['state'].get('step')!=proof['target'] or not detail['resume']['available']:
                    self.block(f'{run}: pause checkpoint does not match planned proof'); return
                key = run+'-resume'
                existing = self.state.get('actions', {}).get(key)
                if existing:
                    # A response lost after successful dispatch is adopted from the
                    # workbench's durable resume-event and actual controller.
                    if resources['resume_events']:
                        proof['resumed'] = True; self.save()
                    else: self.block(f'{run}: ambiguous resume submission; inspect journal')
                    return
                self.evidence(DIRECTORY/(key+'-preview.json'), self.w.api('POST','/api/training/resume',
                    dict(device='a100',run=run,arm=arm,preview=True)))
                self.mutation(key,'/api/training/resume',dict(device='a100',run=run,arm=arm))
                proof['resumed']=True
                self.save()
            elif status not in ACTIVE and status!='stopped':
                self.block(f'{run}: {status} during strict resume validation')
            return
        if detail['state'].get('step',0)>proof['target']:
            proof['next_update_verified'] = True
        if resources['stop']:
            self.block(f'{run}: new user pause after resume proof; automatic progression held'); return
        if status in ('stopped','paused','cancelled','failed'):
            # During resume initialization a stale stopped state can briefly
            # remain, but the live controller and event prove a launch exists.
            if status=='stopped' and resources['live'] and not proof.get('next_update_verified'): return
            self.block(f'{run}: {status}; no automatic configuration changes or restart'); return
        if self.complete(stage, detail) and not proof.get('next_update_verified'):
            self.block(f'{run}: missing next-update resume proof')

    def tick(self):
        if self.state.get('status') in ('blocked','completed'): return
        jobs = {j['run']:j for j in self.w.api('GET','/api/training/jobs?device=a100')['jobs']}
        first = self.detail(RUNS['stage1'], 'baseline')
        if self.state.get('status')=='blocked': return
        if not self.complete('stage1', first):
            status = first['state'].get('status')
            self.state.update(phase='wait_stage1',status='running')
            self.state['resources'] = dict(run=RUNS['stage1'],at=now().isoformat(),**self.w.resources(RUNS['stage1']))
            if status not in ACTIVE:
                self.block(f'Stage1 {status}; respecting pause/failure, later stages held')
            return
        for stage in ('memory','continued'):
            if stage not in self.state.get('checkpoints', {}):
                self.training(stage, jobs)
                return
        from v21_eval_chain import tick
        source = self.state.get('serving_source')
        if not source:
            self.block('Reviewed immutable V2.1 serving source has not been registered')
            return
        self.state.update(phase='eval', status='running')
        if tick(self.w.api, self.state.setdefault('eval', {}), self.save,
                [dict(self.state['checkpoints'][s],training_source=self.state['checkpoints'][s]['source'],
                      source=source) for s in ('stage1','memory','continued')]):
            self.state.update(status='completed',phase='archive',completed_at=now().isoformat())
        elif self.state['eval'].get('status')=='blocked':
            self.block(self.state['eval'].get('reason','Evaluation blocked; inspect eval state'))


def main():
    parser=argparse.ArgumentParser(__doc__)
    parser.add_argument('--freeze-serving',action='store_true',help='Sync and register reviewed immutable serving source')
    args=parser.parse_args()
    DIRECTORY.mkdir(parents=True,exist_ok=True)
    with (DIRECTORY/'chain.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        file = DIRECTORY/'watchdog_state.json'
        state = json.loads(file.read_text()) if file.exists() else dict(plan='plan-v002',status='running')
        save = lambda: write(file,state)
        record = dict(checked_at=now().isoformat())
        try:
            workbench=Workbench()
            if args.freeze_serving:
                frozen=workbench.freeze_serving()
                state.update(serving_source=frozen['source'],serving_manifest=frozen)
                write(DIRECTORY/'serving-source.json',frozen)
                save()
            else:
                Chain(workbench,state,save).tick()
            state['last_error'] = None
        except Exception as error:
            state['last_error'] = repr(error)
            record['error'] = repr(error)
        state.update(last_checked_at=now().isoformat(),next_check_at=(now()+timedelta(seconds=120)).isoformat(),
                     next_check_reason='same-workbench training dependencies, preflight, cleanup, resume, official evaluation')
        save()
        record.update(status=state['status'],phase=state.get('phase'),observed=state.get('observed'),reason=state.get('reason'))
        with (DIRECTORY/'watchdog_log.jsonl').open('a') as stream:
            stream.write(json.dumps(record,ensure_ascii=False)+'\n')
        print(json.dumps(record,ensure_ascii=False))


if __name__=='__main__': main()
