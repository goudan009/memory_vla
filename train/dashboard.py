"""Web training control and one-way development sync; no model imports."""

import datetime
import hashlib
import json
import os
from pathlib import Path
import shlex
import subprocess
import sys
import threading
sys.path.insert(0, str(Path(__file__).resolve().parent))
from run_identity import identity_bundle

PROJECT = '/nvme3/projects/memory_vla'
LIVE = PROJECT + '/deployments/web_live'
SKIP = {'.venv', '__pycache__', '.pytest_cache', '.git', '.ruff_cache'}
SYNC_TREES = ['src/v1_groot_memoryvla', 'src/v2_groot_memoryvla', 'src/v2_1_groot_memoryvla', 'src/v1_2_groot_baseline', 'src/v3_bodymemory',
              'baselines/groot_n1d6', 'train', 'scripts', 'configs', 'evaluation/scripts', 'evaluation/sim', 'evaluation/configs']


class TrainingManager:
    def __init__(self, root, host='172.28.62.17', device_id='a100', device=None):
        self.root = Path(root).resolve()
        self.host = host
        self.device_id = device_id
        self.device = device or {}
        self.label = self.device.get('label', 'A100')
        self.live = self.device.get('paths', {}).get('project', PROJECT) + '/deployments/web_live'
        self.record = self.root/'outputs/shared/training_dashboard/sync_state.json'
        if device_id != 'a100':
            self.record = self.record.parent/device_id/'sync_state.json'
        self.lock = threading.Lock()
        self.state_lock = threading.Lock()
        self.closed = threading.Event()
        self.state = dict(enabled=False, state='idle', host=host, live_source=self.live,
                          last_synced=None, digest=None, error=None)
        if self.record.exists():
            saved = json.loads(self.record.read_text())
            if saved.get('host') == host:
                self.state.update(saved)
        self.thread = threading.Thread(target=self._watch, daemon=True)
        self.thread.start()

    def _save(self, **changes):
        with self.state_lock:
            self.state.update(changes)
            self.record.parent.mkdir(parents=True, exist_ok=True)
            temp = self.record.with_suffix('.pending')
            temp.write_text(json.dumps(self.state, indent=2))
            temp.replace(self.record)

    def sync_state(self):
        with self.state_lock:
            return dict(self.state)

    def _ssh(self, command, code=None, timeout=30):
        result = subprocess.run(['ssh', '-o', 'BatchMode=yes', '-o', 'ConnectTimeout=5',
            self.host, command], input=code, capture_output=True, text=True, timeout=timeout)
        if result.returncode:
            try:
                error = json.loads(result.stdout)['error']
            except (ValueError, KeyError):
                error = (result.stderr or result.stdout or self.label+' SSH失败（设备可能离线）').strip()[-2000:]
            raise ValueError(error)
        return result.stdout

    def _remote(self, action, payload=None, source_digest=None, resume_controller=None):
        request = dict(action=action, payload=payload or {}, source_digest=source_digest,
                       paths=self.device.get('paths', {}), device_id=self.device_id,
                       device_label=self.label, single_gpu=self.device.get('single_gpu', False))
        if resume_controller is not None:
            request['resume_controller']=resume_controller
        # Ship the lightweight V2_1 backend with this RPC, including the first
        # preview before the new source tree has been synchronized remotely.
        backend_path = self.root/'train/v21_workbench.py'
        if not backend_path.is_file():
            backend_path = Path(__file__).with_name('v21_workbench.py')
        backend = backend_path.read_text()
        bundle = ('import sys, types\n_v21_module = types.ModuleType("v21_workbench")\n'
                  'sys.modules["v21_workbench"] = _v21_module\n'
                  'exec('+repr(backend)+', _v21_module.__dict__)\n')
        code = identity_bundle() + bundle + 'REQUEST = ' + repr(request) + chr(10) + (self.root/'train/dashboard_remote.py').read_text()
        return json.loads(self._ssh('python3 -', code, timeout=60))

    def manifest(self):
        files = []
        for relative in SYNC_TREES:
            directory = self.root/relative
            for current, directories, names in os.walk(directory):
                directories[:] = sorted(d for d in directories if d not in SKIP)
                for name in sorted(names):
                    path = Path(current)/name
                    if path.suffix in ['.pyc', '.pyo']:
                        continue
                    if path.is_symlink() and not path.resolve().is_relative_to(self.root):
                        raise ValueError('同步范围包含项目外软链接：'+str(path))
                    files.append(path)
        files.extend(self.root/name for name in ['pyproject.toml', 'uv.lock', 'evaluation/evaluation.md',
                     'evaluation/rollout_policy.py'] if (self.root/name).is_file())
        hashes = {str(p.relative_to(self.root)): hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(files)}
        for target, source in self.external_sources().items():
            if Path(source).is_file():
                hashes[target]=hashlib.sha256(Path(source).read_bytes()).hexdigest()
                continue
            for current, directories, names in os.walk(source, followlinks=True):
                directories[:] = sorted(d for d in directories if d not in SKIP and d!='appdata')
                for name in sorted(names):
                    if Path(name).suffix in ('.pyc','.pyo'): continue
                    path=Path(current)/name
                    hashes[(Path(target)/path.relative_to(source)).as_posix()]=hashlib.sha256(path.read_bytes()).hexdigest()
        digest = hashlib.sha256(json.dumps(hashes,sort_keys=True).encode()).hexdigest()
        return dict(digest=digest, files=hashes, local_source=str(self.root),
                    created_at=datetime.datetime.now().isoformat())

    def external_sources(self):
        config=self.root/'configs/workbench_sources.json'
        return json.loads(config.read_text()) if config.exists() else {}

    def _sync(self, force=False):
        manifest = self.manifest()
        previous = self.sync_state()
        if not force and previous.get('digest') == manifest['digest'] and previous.get('state')=='synced':
            return previous
        self._save(state='syncing', error=None)
        try:
            self._ssh(shlex.join(['mkdir','-p',self.live]))
            # Preserve project compatibility links so realpath-based launchers
            # keep their canonical directory. External vendor trees are materialized.
            external=self.external_sources()
            groups=[('',str(self.root),[name for name in manifest['files'] if not any(name==prefix or name.startswith(prefix+'/') for prefix in external)])]
            groups.extend((prefix,source,[name[len(prefix)+1:] for name in manifest['files'] if name.startswith(prefix+'/')]) for prefix,source in external.items() if not Path(source).is_file())
            for target,source in external.items():
                if not Path(source).is_file():continue
                self._ssh(shlex.join(['mkdir','-p',str(Path(self.live)/Path(target).parent)]))
                result=subprocess.run(['rsync','-aL','--timeout=30','-e','ssh -o BatchMode=yes -o ConnectTimeout=5',
                    source,self.host+':'+self.live+'/'+target],capture_output=True,text=True,timeout=120)
                if result.returncode:raise ValueError(result.stderr.strip()[-2000:])
            for prefix,source,names in groups:
                destination=self.live+'/'+prefix
                self._ssh(shlex.join(['mkdir','-p',destination]))
                result = subprocess.run(['rsync','-aL' if prefix else '-a','--timeout=30','--from0','--files-from=-',
                    '-e','ssh -o BatchMode=yes -o ConnectTimeout=5',source+'/',self.host+':'+destination+'/'],
                    input='\0'.join(names)+'\0',capture_output=True,text=True,timeout=120)
                if result.returncode:
                    raise ValueError(result.stderr.strip()[-2000:])
            if self.manifest()['digest'] != manifest['digest']:
                raise ValueError('同步期间源码有改动，将在下一次同步更新；稍后再启动训练')
            code=chr(10).join(['from pathlib import Path','import hashlib,json',
                'root=Path('+repr(self.live)+')','manifest='+repr(manifest),
                'assert all(hashlib.sha256((root/name).read_bytes()).hexdigest()==sha for name,sha in manifest["files"].items()), "Remote source checksum mismatch"',
                'p=root/"web_source_manifest.json"',
                'previous=json.loads(p.read_text()).get("files",{}) if p.exists() else {}',
                'for name in set(previous)-set(manifest["files"]):',
                '    stale=root/name',
                '    if stale.resolve().is_relative_to(root) and stale.is_file(): stale.unlink()',
                'pending=p.with_suffix(".pending")',
                'pending.write_text(json.dumps(manifest))','pending.replace(p)'])
            self._ssh('python3 -', code)
            self._save(state='synced', digest=manifest['digest'], last_synced=datetime.datetime.now().isoformat(),
                       file_count=len(manifest['files']), error=None)
            return self.sync_state()
        except Exception as error:
            self._save(state='error',error=str(error))
            raise

    def _watch(self):
        while not self.closed.wait(4):
            if self.sync_state()['enabled'] and self.lock.acquire(blocking=False):
                try:
                    self._sync()
                except Exception:
                    pass  # The sync error is visible and the next poll retries.
                finally:
                    self.lock.release()

    def sync(self, request):
        if not isinstance(request,dict) or set(request)-{'enabled','once'}:
            raise ValueError('同步参数不正确')
        for key in request:
            if type(request[key]) is not bool:
                raise ValueError('同步开关必须为布尔值')
        with self.lock:
            if 'enabled' in request:
                self._save(enabled=request['enabled'])
            if request.get('once') or request.get('enabled'):
                return self._sync(force=True)
            return self.sync_state()

    def options(self):
        result=self._remote('catalog')
        import sys
        sys.path.insert(0,str(self.root/'evaluation/scripts'))
        from groot_tasks import TASK_NAMES
        registered={item['task']:item for item in result['datasets']}
        tasks=[]
        for task,name in TASK_NAMES.items():
            item=dict(registered.get(task,dict(task=task,dataset=None,split=None)),task_name=name,ready=task in registered)
            raw=self.root/f'assets/datasets/b1k2025/data/task-{task:04d}'
            episodes=len(list(raw.glob('*.parquet')))
            cameras={camera:len(list((self.root/f'assets/datasets/b1k2025/videos/task-{task:04d}/observation.images.rgb.{camera}').glob('*.mp4')))
                for camera in ['head','left_wrist','right_wrist']}
            preparation=self.dataset_status(task)
            if item['ready']:
                preparation=dict(preparation,status='completed',dataset=item['dataset'],split=item['split'],
                                 verified_on=self.label)
            item.update(raw_episodes=episodes,raw_cameras=cameras,can_prepare=episodes==200 and all(n==200 for n in cameras.values()),
                preparation=preparation)
            if self.device_id != 'a100':
                item['can_prepare']=(self.root/f'assets/datasets/b1k2025_groot_task{task}_v1_200/meta/info.json').is_file()
            tasks.append(item)
        result['datasets']=tasks
        batch_path=self.root/'outputs/shared/training_dashboard/datasets/all_tasks/status.json'
        if self.device_id != 'a100':
            batch_path=self.record.parent/'datasets/all_tasks/status.json'
        return dict(result, host=self.host, device_id=self.device_id, device_label=self.label,
                    single_gpu=self.device.get('single_gpu', False), sync=self.sync_state(),
                    dataset_queue=json.loads(batch_path.read_text()) if batch_path.exists() else None)

    def dataset_status(self, task):
        path=self.root/f'outputs/shared/training_dashboard/datasets/task{task}/status.json'
        if self.device_id != 'a100':
            path=self.record.parent/f'datasets/task{task}/status.json'
        return json.loads(path.read_text()) if path.exists() else dict(status='not_prepared')

    def prepare_dataset(self, request):
        if not isinstance(request,dict) or set(request)!={'task'} or type(request['task']) is not int or not 0<=request['task']<50:
            raise ValueError('请选择0–49的任务')
        task=request['task']
        if self.device_id != 'a100':
            return self._prepare_device_dataset(task)
        with self.lock:
            state=self.dataset_status(task)
            if state['status'] in ['queued','starting','converting','syncing','completed']:
                return state
            work=self.root/f'outputs/shared/training_dashboard/datasets/task{task}'
            work.mkdir(parents=True,exist_ok=True)
            python=self.root/'src/v1_groot_memoryvla/.venv/bin/python'
            if not python.is_file():
                raise ValueError('本机B1K数据转换环境不存在')
            raw=self.root/'assets/datasets/b1k2025'
            if len(list((raw/f'data/task-{task:04d}').glob('*.parquet')))!=200 or any(
                len(list((raw/f'videos/task-{task:04d}/observation.images.rgb.{camera}').glob('*.mp4')))!=200
                for camera in ['head','left_wrist','right_wrist']):
                raise ValueError('本机缺少所选任务的200条示教或三路相机视频，请先下载原始数据')
            state=dict(status='starting',task=task,log=str(work/'prepare.log'))
            (work/'status.json').write_text(json.dumps(state))
            with (work/'prepare.log').open('ab') as log:
                proc=subprocess.Popen([str(python),'-u',str(self.root/'train/prepare_web_dataset.py'),
                    '--root',str(self.root),'--host',self.host,'--task',str(task)],stdin=subprocess.DEVNULL,
                    stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
            return dict(state,pid=proc.pid)

    def _prepare_device_dataset(self, task):
        """Transfer the existing registered task; retain the original split."""
        with self.lock:
            state=self.dataset_status(task)
            if state['status'] in ['starting','syncing','completed']:
                return state
            work=self.record.parent/f'datasets/task{task}'
            work.mkdir(parents=True,exist_ok=True)
            state=dict(status='starting',task=task,log=str(work/'prepare.log'))
            (work/'status.json').write_text(json.dumps(state))
            with (work/'prepare.log').open('ab') as log:
                proc=subprocess.Popen([sys.executable,'-u',str(self.root/'train/prepare_device.py'),
                    '--root',str(self.root),'--device',self.device_id,'--task',str(task),
                    '--dataset-only'],stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,
                    start_new_session=True)
            return dict(state,pid=proc.pid)

    def prepare(self, request):
        return dict(self._remote('preview',request),sync=self.sync_state())

    def start(self, request):
        with self.lock:
            self._remote('preview',request)
            state=self._sync(force=True)
            return self._remote('start',request,source_digest=state['digest'])

    def snapshot(self):
        return dict(self._remote('jobs'),sync=self.sync_state())

    def detail(self, request):
        result=self._remote('detail',request)
        # The remote detail API verifies the job identity first. Only read the
        # matching local archive; never read an arbitrary user-supplied path.
        import re
        run,arm=result.get('run'),result.get('arm')
        if all(isinstance(x,str) and re.fullmatch(r'[A-Za-z0-9_.-]+',x) and x not in ('.','..') for x in (run,arm,self.device_id)):
            root=(self.root/'reports/training/runs').resolve()
            directory=root/self.device_id/run/arm
            path=directory/'training_report.md'
            latest=directory/'latest.json'
            if path.resolve().is_relative_to(root) and latest.resolve().is_relative_to(root) and path.is_file() and latest.is_file():
                try:
                    record=json.loads(latest.read_text())
                    if record.get('run')==run and record.get('arm')==arm and record.get('device')==self.device_id:
                        result['analysis_report']=dict(text=path.read_text(),event=record.get('event'),
                            step=record.get('job',{}).get('step'),status=record.get('job',{}).get('status'))
                except (OSError,ValueError):
                    pass
        return result

    def stop(self, request):
        return self._remote('stop',request)

    def delete(self, request):
        with self.lock:
            return self._remote('delete',request)

    def delete_checkpoint(self, request):
        with self.lock:
            return self._remote('checkpoint-delete', request)

    def resume(self, request):
        with self.lock:
            return self._remote('resume',request,
                resume_controller=(self.root/'train/web_training_resume.py').read_text())

    def cancel(self, request):
        return self._remote('cancel',request)

    def close(self):
        self.closed.set()
