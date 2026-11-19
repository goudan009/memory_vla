"""4090 controller for an optional single-GPU simulation device."""
import json
from pathlib import Path
import shlex
import subprocess
import threading


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
                rows=[self._localize(row) for row in self._remote('snapshot')]
                self.jobs=rows;self.error=None
            except (ValueError,OSError,subprocess.SubprocessError) as error: self.error=str(error)
            self.closed.wait(5)

    def close(self): self.closed.set()

    def _remote(self, action, payload=None):
        payload=dict(payload or {})
        if payload.get('resume_from'):
            from groot_eval_control import EvaluationControl
            run=EvaluationControl(self.root).run({'run':payload['resume_from']})
            marker=json.loads((run/'.device.json').read_text())
            payload['resume_from']=marker['remote_output']
        if not payload.get('resume_from') and action in ('prepare','start'):
            if payload.get('mode','single1') not in ('single1','single2') or payload.get('gpu_index',0)!=0:
                raise ValueError('RTX 6000 Ada 只有 GPU0，请选择单卡模式')
            payload['gpu_index']=0
        identity=None
        if action in ('prepare','start') and self.policy_resolver:
            host,port='172.28.62.17',5556
            if payload.get('resume_from'):
                saved=json.loads((run/'results.json').read_text())['configuration']
                host,port=saved['policy_server'].rsplit(':',1)
            identity=self.policy_resolver(payload.get('policy_host',host),int(payload.get('policy_port',port)))
        request=dict(action=action,payload=payload,policy_identity=identity,project=self.project,
                     environment=self.device['evaluation']['environment'])
        module=(self.root/'evaluation/scripts/groot_device_eval_remote.py').read_text()
        code='import sys\nsys.path.insert(0,'+repr(self.project+'/deployments/web_live/evaluation/scripts')+')\n'
        code+=module+'\nprint(json.dumps(dispatch('+repr(request)+'),ensure_ascii=False))\n'
        result=subprocess.run(['ssh','-o','BatchMode=yes','-o','ConnectTimeout=5',self.host,'python3 -'],
            input=code,capture_output=True,text=True,timeout=60)
        if result.returncode: raise ValueError((result.stderr or result.stdout).strip()[-2500:])
        return json.loads(result.stdout)

    def options(self):
        return dict(self._remote('options'),device=self.device_id,label=self.device['label'])

    def prepare(self, request):
        with self.training.lock: self.training._sync()
        return self._remote('prepare',request)

    def start(self, request):
        with self.lock:
            with self.training.lock: self.training._sync(force=True)
            return self._localize(self._remote('start',request))

    def _localize(self, row):
        row=dict(row); remote_folder=row['folder']; path=Path(remote_folder)
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
        return result

    def pause(self, request): return self._action('pause',request)
    def delete(self, request): return self._action('delete',request)
