"""4090 controller for an optional single-GPU simulation device."""
import hashlib
import json
import os
from pathlib import Path
import re
import shlex
import subprocess
import threading

from groot_web_launch import (
    LaunchManager, TASK40_BASELINE_EXPERIMENT, TASK40_DEFAULT_EXPERIMENT,
    TASK40_NEXT_EXPERIMENT, task40_artifacts,
)


TASK40_EXPERIMENT = Path('Output/test_01/experiment_09')
TASK40_RUNNER = Path('evaluation/scripts')
TASK40_ANCHOR = Path('Output/test_01/experiment_01/checkpoints')
TASK40_WEIGHTS = Path('Output/task40_memory_overnight/replay/original_step4000')


class DeviceEvaluationManager:
    def __init__(self, root, device_id, device, training):
        self.root=Path(root); self.device_id=device_id; self.device=device
        self.host=device['host']; self.project=device['paths']['project']
        self.training=training; self.policy_resolver=None
        self.lock=threading.Lock()
        self.closed=threading.Event(); self.jobs=[]; self.error=None
        self.thread=threading.Thread(target=self._watch,daemon=True); self.thread.start()

    def _watch(self):
        while not self.closed.is_set():
            try:
                before = {row['id'] for row in self.jobs}
                rows=[self._localize(row) for row in self._remote('snapshot')]
                observed = {row['id'] for row in rows}
                with self.lock:
                    started_during_poll = [row for row in self.jobs
                                           if row['id'] not in before and row['id'] not in observed]
                    self.jobs = started_during_poll + rows
                    self.error = None
            except (ValueError,OSError,subprocess.SubprocessError) as error: self.error=str(error)
            self.closed.wait(5)

    def close(self): self.closed.set()

    def _sync_task40_dependencies(self, request=None):
        # Keep this allowlist separate from the workbench source manifest and snapshots.
        files = [TASK40_RUNNER/'paired_sim_diagnostic.py']
        files.extend(TASK40_ANCHOR/name for name in (
            'start.json', 'anchor.json', 'anchor_live_observation.npz', 'sim_checkpoint.pt'))
        decisions = sorted((self.root/TASK40_ANCHOR).glob('prefix/[0-9][0-9][0-9][0-9]/decision.json'))
        if not decisions:
            raise ValueError('Task40 检查点缺少历史决策记录')
        files.extend(path.relative_to(self.root) for path in decisions)
        files.extend(TASK40_WEIGHTS/name for name in (
            'config.json', 'embodiment_id.json', 'processor_config.json', 'statistics.json'))
        directed = None
        request = request or {}
        if request.get('experiment') == TASK40_BASELINE_EXPERIMENT and request.get('restore_checkpoint'):
            manager = LaunchManager(self.root)
            checkpoint = manager._task40_path(request['restore_checkpoint'], '失败检查点')
            lineage = manager._task40_failure_lineage(checkpoint)
            prefix = Path(lineage['prefix'])
            files.extend((checkpoint/name).relative_to(self.root) for name in (
                'sim.pt', 'memory.pt', 'policy_rng_online.pt', 'observation.npz', 'physical.json', 'source.json'))
            files.extend((prefix/name).relative_to(self.root) for name in ('initial.json', 'trial.json', 'truth.jsonl'))
            for index in range(lineage['start_decision'] + 1):
                files.extend((prefix/f'{index:04d}'/name).relative_to(self.root)
                             for name in ('decision.json', 'observation.npz'))
                if index < lineage['start_decision']:
                    files.append((prefix/f'{index:04d}'/'executed.npz').relative_to(self.root))
            directed = checkpoint.relative_to(self.root).as_posix()
        hashes = {path.as_posix(): hashlib.sha256((self.root/path).read_bytes()).hexdigest() for path in files}
        digest = hashlib.sha256(json.dumps(hashes, sort_keys=True).encode()).hexdigest()
        storage = getattr(self, 'device', {}).get('evaluation', {}).get('task40_baseline_storage')
        baseline = request.get('experiment') == TASK40_BASELINE_EXPERIMENT
        dependency_root = (Path(storage)/'dependencies' if baseline and storage else
                           Path(self.project)/'setup/task40_dependencies')
        destination = dependency_root/digest
        self.training._ssh(shlex.join(['mkdir', '-p', str(destination)]))
        subprocess.run(['rsync', '-aL', '--relative', '--checksum', '--from0', '--files-from=-', '--timeout=30',
                        '-e', 'ssh -o BatchMode=yes -o ConnectTimeout=5', str(self.root)+'/',
                        self.host+':'+str(destination)+'/'], input='\0'.join(hashes)+'\0',
                       capture_output=True, text=True, timeout=90, check=True)
        return {'digest': digest, 'files': hashes, 'directed': directed,
                'base': str(dependency_root), 'paths': {
            'GROOT_TASK40_RUNNER': TASK40_RUNNER.as_posix(),
            'GROOT_TASK40_ANCHOR': TASK40_ANCHOR.as_posix(),
            'GROOT_TASK40_WEIGHTS': TASK40_WEIGHTS.as_posix(),
        }}

    def _remote(self, action, payload=None, task40_dependencies=None):
        payload=dict(payload or {})
        collect = payload.get('task40_collect') is True
        if collect and action in ('prepare','start'):
            gpu = payload.get('sim_gpu', 0)
            if type(gpu) is not int or gpu != 0:
                raise ValueError('RTX 6000 Ada 只有 GPU0')
            payload['sim_gpu'] = gpu
        if payload.get('resume_from'):
            from groot_eval_control import EvaluationControl
            run=EvaluationControl(self.root).run({'run':payload['resume_from']})
            marker=json.loads((run/'.device.json').read_text())
            payload['resume_from']=marker['remote_output']
        if not collect and not payload.get('resume_from') and action in ('prepare','start'):
            if payload.get('mode','single1') not in ('single1','single2') or payload.get('gpu_index',0)!=0:
                raise ValueError('RTX 6000 Ada 只有 GPU0，请选择单卡模式')
            payload['gpu_index']=0
        identity=None
        if action in ('prepare','start') and self.policy_resolver:
            host,port='172.28.62.17',5556
            if payload.get('resume_from'):
                saved=json.loads((run/'results.json').read_text())['configuration']
                host,port=saved['policy_server'].rsplit(':',1)
            policy_host = payload.get('policy_host', host)
            policy_port = payload.get('policy_port', None if collect else port)
            if not collect or (policy_host == host and type(policy_port) is int and 1024 <= policy_port <= 65535):
                identity = self.policy_resolver(policy_host, int(policy_port))
        evaluation = self.device['evaluation']
        request=dict(action=action,payload=payload,policy_identity=identity,project=self.project,
                     environment=evaluation['environment'], task40_dependencies=task40_dependencies,
                     task40_baseline_storage=evaluation.get('task40_baseline_storage'))
        module=(self.root/'evaluation/scripts/groot_device_eval_remote.py').read_text()
        code='import sys\nsys.path.insert(0,'+repr(self.project+'/deployments/web_live/evaluation/scripts')+')\n'
        if action in ('finish', 'delete'):
            # Existing runs use frozen source snapshots with older control behavior.
            # Ship only the current result-control code; leave experiment code intact.
            control=(self.root/'evaluation/scripts/groot_eval_control.py').read_text()
            code+='import types\ncontrol=types.ModuleType("groot_eval_control")\n'
            code+='exec('+repr(control)+',control.__dict__)\nsys.modules["groot_eval_control"]=control\n'
        code+=module+'\nprint(json.dumps(dispatch('+repr(request)+'),ensure_ascii=False))\n'
        result=subprocess.run(['ssh','-o','BatchMode=yes','-o','ConnectTimeout=5',self.host,'python3 -'],
            input=code,capture_output=True,text=True,timeout=60)
        if result.returncode: raise ValueError((result.stderr or result.stdout).strip()[-2500:])
        return json.loads(result.stdout)

    def options(self):
        result = dict(self._remote('options'), device=self.device_id, label=self.device['label'])
        collection = result.get('task40_collect')
        evidence = self.root / TASK40_EXPERIMENT / 'metrics' / f'dual_{self.device_id}_gpu0_validated.json'
        if isinstance(collection, dict) and evidence.is_file():
            result['task40_collect'] = {**collection, 'sim_slots': {**collection.get('sim_slots', {}), '0': 2}}
        return result

    def prepare(self, request):
        with self.training.lock:
            self.training._sync()
            dependencies=self._sync_task40_dependencies(request) if request.get('task40_collect') is True else None
        return self._remote('prepare',request,dependencies)

    def start(self, request):
        with self.lock:
            collect=request.get('task40_collect') is True
            with self.training.lock:
                self.training._sync(force=True)
                dependencies=self._sync_task40_dependencies(request) if collect else None
            row = self._localize(self._remote('start',request,dependencies), sync=not collect)
            self.jobs = [row, *(old for old in self.jobs if old['id'] != row['id'])]
            return row

    def _localize_task40(self, row, sync):
        remote = Path(row['output'])
        remote_base = Path(self.device['evaluation']['environment'].get(
            'GROOT_TASK40_EXPERIMENT_ROOT', Path(self.project)/TASK40_EXPERIMENT))
        local_base = Path(os.environ.get('GROOT_TASK40_EXPERIMENT_ROOT', self.root/TASK40_EXPERIMENT)).resolve()
        group = row.get('group', TASK40_DEFAULT_EXPERIMENT)
        storage = self.device['evaluation'].get('task40_baseline_storage')
        if group == TASK40_BASELINE_EXPERIMENT and storage:
            remote_base = Path(storage)/group
            local_base = local_base.with_name(group)
        elif group in (TASK40_NEXT_EXPERIMENT, TASK40_BASELINE_EXPERIMENT):
            remote_base = remote_base.with_name(group)
            local_base = local_base.with_name(group)
        elif group != TASK40_DEFAULT_EXPERIMENT:
            raise ValueError('不支持的 Task40 采集组')
        remote_root = remote_base/'runs'
        if remote.parent != remote_root or not re.fullmatch(r'[a-f0-9]{12}', remote.name):
            raise ValueError('Task40 采集输出不在所选远端实验的 runs 目录内')
        local_root = local_base/'runs'
        local = local_root/remote.name
        if local.resolve().parent != local_root:
            raise ValueError('Task40 采集输出不在所选本地实验的 runs 目录内')
        marker_path = local/'.device.json'
        terminal = row.get('state') in ('complete', 'failed', 'stopped', 'cancelled')
        sealed = False
        if group == TASK40_BASELINE_EXPERIMENT and terminal:
            try:
                marker = json.loads(marker_path.read_text())
                sealed = (marker.get('sealed_job') == row['id'] and marker.get('remote_output') == str(remote)
                          and (row['state'] != 'complete' or (local/'results.json').is_file()))
            except (OSError, ValueError):
                pass
        if sync and row.get('output_exists') and not sealed:
            try:
                local.mkdir(parents=True, exist_ok=True)
                subprocess.run(['rsync', '-a', '--partial-dir=.rsync-partial', '--timeout=10',
                                '--exclude=/appdata/', '--exclude=/.device.json',
                                '-e', 'ssh -o BatchMode=yes -o ConnectTimeout=5',
                                self.host+':'+str(remote)+'/', str(local)+'/'],
                               capture_output=True, text=True, timeout=20, check=True)
                marker = dict(device=self.device_id, remote_output=str(remote), source=row.get('source'))
                if group == TASK40_BASELINE_EXPERIMENT and terminal:
                    marker['sealed_job'] = row['id']
                marker_path.write_text(json.dumps(marker))
            except (OSError, subprocess.SubprocessError) as error:
                detail = getattr(error, 'stderr', None)
                if isinstance(detail, bytes):
                    detail = detail.decode(errors='replace')
                row['artifact_sync_error'] = (detail or str(error)).strip()[-2000:]
        if 'plan' in row:
            row['plan'] = dict(row['plan'], output=str(local))
        return dict(row, device=self.device_id, remote_output=str(remote), output=str(local),
                    result_ready=(local/'results.json').is_file(), artifacts=task40_artifacts(local))

    def _localize(self, row, sync=True):
        row=dict(row)
        if row.get('kind') == 'task40_collect':
            return self._localize_task40(row, sync)
        remote_folder=row['folder']; path=Path(remote_folder)
        folder=(path.parent/(self.device_id+'_'+path.name)).as_posix()
        local=self.root/'evaluation'/folder
        local.mkdir(parents=True,exist_ok=True)
        marker=dict(device=self.device_id,remote_run=remote_folder,remote_output=row['output'],source=row.get('source'))
        (local/'.device.json').write_text(json.dumps(marker))
        requests=local/'.runtime/preview/requests'
        if requests.exists():
            subprocess.run(['ssh','-o','BatchMode=yes','-o','ConnectTimeout=5',self.host,
                shlex.join(['mkdir','-p',row['output']+'/.runtime/preview/requests'])],
                capture_output=True,text=True,timeout=10,check=True)
            subprocess.run(['rsync','-a','--timeout=10','-e','ssh -o BatchMode=yes -o ConnectTimeout=5',
                str(requests)+'/',self.host+':'+row['output']+'/.runtime/preview/requests/'],
                capture_output=True,text=True,timeout=15,check=True)
        subprocess.run(['rsync','-a','--partial','--timeout=10','--exclude=.controller.json','--exclude=.runtime/preview/requests/',
            '-e','ssh -o BatchMode=yes -o ConnectTimeout=5',self.host+':'+row['output']+'/',str(local)+'/'],
            capture_output=True,text=True,timeout=20,check=True)
        return dict(row,device=self.device_id,remote_output=row['output'],folder=folder,output=str(local))

    def snapshot(self):
        return [dict(row,device_error=self.error) for row in self.jobs]

    def _action(self, action, request):
        # Reuse the local path validation; the stored marker binds the remote run.
        from groot_eval_control import EvaluationControl
        run=EvaluationControl(self.root).run(request)
        marker=json.loads((run/'.device.json').read_text())
        if marker['device']!=self.device_id: raise ValueError('评估设备不匹配')
        result=self._remote(action,dict(run=marker['remote_run']))
        if action=='delete':
            result=EvaluationControl(self.root).delete(request)
        elif action=='finish':
            # The remote action has confirmed its controller is stopped. Reflect
            # the paused state immediately, including for sealed cache entries.
            EvaluationControl(self.root).finish(request)
        return result

    def pause(self, request): return self._action('pause',request)
    def finish(self, request): return self._action('finish',request)
    def delete(self, request): return self._action('delete',request)
    def stop_restore(self, request): return self._remote('stop_restore',request)
