"""Local quick-benchmark jobs and reports; no Torch or simulator imports."""
import ast
import hashlib
import json
import os
from pathlib import Path
import re
import shlex
import signal
import subprocess
import sys
import threading
import time
import uuid

import libero_dashboard


TITLES = {'network': '网络回归', 'action': '真实动作短训练', 'libero': 'LIBERO 环境检查'}
SUITES = {'libero_spatial': 'Spatial', 'libero_object': 'Object', 'libero_goal': 'Goal',
          'libero_10': 'Long-10', 'libero_90': 'Long-90'}
LEGACY = {'retrieval_fixed': ('network', '检索修复 · 首次验收'),
          'action_smoke_memory': ('action', 'Memory · 100 步验收'),
          'action_smoke_baseline': ('action', 'Baseline · 100 步验收'),
          'action_smoke_direct_empty': ('action', '关闭历史 · 100 步验收'),
          'libero_smoke': ('libero', 'LIBERO Spatial · 首次验收'),
          'libero_goal_smoke': ('libero', 'LIBERO Goal · 首次验收'),
          'libero_10_smoke': ('libero', 'LIBERO Long-10 · 首次验收')}


def read_json(path, default=None):
    try:
        return json.loads(path.read_text())
    except FileNotFoundError:
        return default


def atomic_json(path, value):
    temp = path.with_suffix('.pending')
    temp.write_text(json.dumps(value, ensure_ascii=False, indent=2)+'\n')
    temp.replace(path)


def process_start(pid):
    try:
        fields = (Path('/proc')/str(pid)/'stat').read_text().rsplit(')', 1)[1].split()
        return fields[19] if fields[0] != 'Z' else None
    except FileNotFoundError:
        return None


def local_gpus():
    result = subprocess.run(['nvidia-smi', '--query-gpu=index,name,memory.free,memory.total',
                             '--format=csv,noheader,nounits'], capture_output=True, text=True, timeout=5)
    if result.returncode:
        raise ValueError('无法读取本机 GPU：'+result.stderr.strip())
    return [dict(index=int(i), name=n.strip(), free_mb=int(f), total_mb=int(t))
            for line in result.stdout.splitlines() for i, n, f, t in [line.split(',')]]


class BenchmarkManager:
    def __init__(self, root):
        self.root = Path(root).resolve()
        self.output = self.root/'outputs/benchmarks'
        self.runs = self.output/'web'
        self.python = self.root/'src/v1_groot_memoryvla/.venv/bin/python'
        self.lock = threading.Lock()

    def options(self):
        needed = {
            'network': [self.python, self.root/'outputs/diagnostics/v1_cognitive_16000_weights.pt',
                        self.root/'outputs/diagnostics/v1_real_replay_features.pt'],
            'action': [self.python, self.root/'checkpoints/groot_n1d6_base/config.json',
                       self.root/'outputs/diagnostics/v1_real_replay_features.pt'],
            'libero': [self.root/'env/libero-smoke/bin/python',
                       self.root/'third_party/LIBERO/libero/libero/assets']}
        catalog = [dict(id=key, title=TITLES[key], available=all(p.exists() for p in paths),
                        missing=[str(p.relative_to(self.root)) for p in paths if not p.exists()])
                   for key, paths in needed.items()]
        tasks = {}
        task_file = self.root/'third_party/LIBERO/libero/libero/benchmark/libero_suite_task_map.py'
        if task_file.is_file():
            tree = ast.parse(task_file.read_text())
            tasks = ast.literal_eval(next(node.value for node in tree.body if isinstance(node, ast.Assign)))
        try:
            gpus, gpu_error = local_gpus(), None
        except (OSError, ValueError, subprocess.SubprocessError) as error:
            gpus, gpu_error = [], str(error)
        catalog += [dict(id=key,title=title,available=self.python.is_file(),missing=[])
                    for key,title in libero_dashboard.TITLES.items()]
        return dict(catalog=catalog, gpus=gpus, gpu_error=gpu_error, libero=libero_dashboard.catalog(self),
                    suites=[dict(id=k, title=v, tasks=[dict(id=i, name=name.replace('_', ' '))
                        for i, name in enumerate(tasks.get(k, []))]) for k, v in SUITES.items()],
                    fixture='固定 V1 夹具 · B1K Task0/34/40 · GR00T-N1.6-3B',
                    checkpoint=str(self.root/'checkpoints/groot_n1d6_base'))

    def prepare(self, request):
        if isinstance(request,dict) and isinstance(request.get('kind'),str) and request['kind'] in libero_dashboard.TITLES:
            return libero_dashboard.prepare(self,request)
        if not isinstance(request, dict) or set(request)-{'kind', 'gpu', 'steps', 'arm', 'suite', 'tasks', 'seed', 'notes'}:
            raise ValueError('不支持的快速验证参数')
        kind = request.get('kind', 'network')
        if not isinstance(kind, str) or kind not in TITLES:
            raise ValueError('请选择已接入的验证类型')
        gpu = request.get('gpu', 0)
        if type(gpu) is not int or gpu < 0:
            raise ValueError('GPU 编号不正确')
        notes = request.get('notes', '')
        if not isinstance(notes, str) or len(notes) > 500:
            raise ValueError('备注请控制在 500 字以内')
        options = self.options()
        entry = next(c for c in options['catalog'] if c['id'] == kind)
        if not entry['available']:
            raise ValueError('本机缺少运行文件：'+', '.join(entry['missing']))
        device = next((g for g in options['gpus'] if g['index'] == gpu), None)
        if device is None:
            raise ValueError(options['gpu_error'] or '本机不存在所选 GPU')
        min_mb = {'network': 3000, 'action': 28000, 'libero': 2000}[kind]
        config = dict(kind=kind, gpu=gpu, notes=notes.strip())
        py = str(self.python)
        replay = str(self.root/'outputs/diagnostics/v1_real_replay_features.pt')
        stages = []
        if kind == 'network':
            stages = [dict(name='代码回归', argv=[py, '-m', 'pytest', '-q',
                'src/v1_groot_memoryvla/tests', 'tests/test_groot_training_dashboard.py',
                '--junitxml={run}/unit.xml'], report='unit.json', unit=True),
                dict(name='真实 EOS 检索', argv=[py, '-m', 'v1_groot_memoryvla.audit.retrieval_benchmark',
                    '--weights', str(self.root/'outputs/diagnostics/v1_cognitive_16000_weights.pt'),
                    '--replay', replay, '--output', '{run}/retrieval.json'], report='retrieval.json')]
        elif kind == 'action':
            steps, arm = request.get('steps', 100), request.get('arm', 'all')
            if type(steps) is not int or not 1 <= steps <= 300:
                raise ValueError('短训练步数应在 1–300 之间')
            if arm not in ('all', 'baseline', 'direct_empty', 'memory'):
                raise ValueError('训练对照分支不正确')
            config.update(steps=steps, arm=arm)
            for branch in (['baseline', 'direct_empty', 'memory'] if arm == 'all' else [arm]):
                stages.append(dict(name={'baseline': 'Baseline', 'direct_empty': 'Direct · 关闭历史',
                    'memory': 'Memory · Q/K 归一化'}[branch], argv=[py, '-m',
                    'v1_groot_memoryvla.audit.action_smoke', '--checkpoint',
                    str(self.root/'checkpoints/groot_n1d6_base'), '--replay', replay,
                    '--steps', str(steps), '--arm', branch, '--output', '{run}/'+branch+'.json'],
                    report=branch+'.json', steps=steps))
        else:
            suite, tasks = request.get('suite', 'libero_spatial'), request.get('tasks', [0])
            steps, seed = request.get('steps', 20), request.get('seed', 42)
            if not isinstance(suite, str) or suite not in SUITES:
                raise ValueError('LIBERO 套件不正确')
            count = len(next(s for s in options['suites'] if s['id'] == suite)['tasks'])
            if (not isinstance(tasks, list) or not 1 <= len(tasks) <= 5 or len(set(map(str, tasks))) != len(tasks)
                    or any(type(t) is not int or not 0 <= t < count for t in tasks)):
                raise ValueError('请选择 1–5 个有效且不重复的 LIBERO 任务')
            if type(steps) is not int or not 1 <= steps <= 200 or type(seed) is not int or not 0 <= seed < 2**32:
                raise ValueError('环境检查步数为 1–200，seed 为非负 32 位整数')
            config.update(suite=suite, tasks=tasks, steps=steps, seed=seed)
            stages = [dict(name=f'{SUITES[suite]} · Task{task}', argv=[
                str(self.root/'env/libero-smoke/bin/python'), 'scripts/libero_smoke.py',
                '--suite', suite, '--tasks', str(task), '--steps', str(steps), '--seed', str(seed), '--save-frames',
                '--output', '{run}/task-'+str(task)+'.json'], report=f'task-{task}.json') for task in tasks]
        return dict(config=config, title=TITLES[kind], stages=stages, gpu=device,
                    required_free_mb=min_mb, ready=device['free_mb'] >= min_mb,
                    commands=[shlex.join(s['argv']) for s in stages])

    def start(self, request):
        with self.lock:
            plan = self.prepare(request)
            if not plan['ready']:
                raise ValueError(f"所选 GPU 可用显存不足，本检查需要约 {plan['required_free_mb']/1024:.0f} GiB")
            if any(r['status'] in ('running', 'starting', 'stopping') for r in self.snapshot()['jobs']):
                raise ValueError('已有快速验证正在运行，请完成或停止后再启动')
            run_id = time.strftime('%Y%m%d-%H%M%S')+'-'+uuid.uuid4().hex[:8]
            directory = self.runs/run_id
            directory.mkdir(parents=True)
            digest = hashlib.sha256()
            files = []
            for current, directories, names in os.walk(self.root/'src/v1_groot_memoryvla'):
                directories[:] = sorted(d for d in directories if d not in ('.venv', '__pycache__', '.git'))
                files.extend(Path(current)/name for name in names if name.endswith('.py'))
            files += [self.root/'scripts/libero_smoke.py', self.root/'scripts/libero_rollout.py',
                      Path(__file__), Path(__file__).with_name('benchmark_worker.py'), Path(libero_dashboard.__file__)]
            for path in sorted(p for p in files if not any(x in p.parts for x in ('.venv', '__pycache__'))):
                digest.update(str(path.relative_to(self.root)).encode())
                digest.update(path.read_bytes())
            plan.update(id=run_id, created_at=time.strftime('%Y-%m-%d %H:%M:%S'), source_digest=digest.hexdigest())
            atomic_json(directory/'plan.json', plan)
            atomic_json(directory/'state.json', dict(status='starting', completed=0, total=len(plan['stages'])))
            env = dict(os.environ, CUDA_VISIBLE_DEVICES=str(plan['config']['gpu']),
                PYTHONPATH=str(self.root/'src'), PYTHONNOUSERSITE='1', NO_ALBUMENTATIONS_UPDATE='1',
                HF_HUB_OFFLINE='1', TRANSFORMERS_OFFLINE='1', PYTHONUNBUFFERED='1')
            try:
                with (directory/'worker.log').open('ab') as log:
                    process = subprocess.Popen([sys.executable, str(Path(__file__).with_name('benchmark_worker.py')),
                        '--root', str(self.root), '--run', str(directory)], cwd=self.root, env=env,
                        stdin=subprocess.DEVNULL, stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
                atomic_json(directory/'process.json', dict(pid=process.pid, start=process_start(process.pid)))
                threading.Thread(target=process.wait, daemon=True).start()
            except OSError as error:
                atomic_json(directory/'state.json', dict(status='failed', error=str(error), completed=0, total=len(plan['stages'])))
                raise
            return self.detail({'run': run_id})

    def _directory(self, run_id):
        if not isinstance(run_id, str) or not re.fullmatch(r'\d{8}-\d{6}-[a-f0-9]{8}', run_id):
            raise ValueError('验证记录编号不正确')
        directory = (self.runs/run_id).resolve()
        if directory.parent != self.runs.resolve() or not (directory/'plan.json').is_file():
            raise ValueError('验证记录不存在')
        return directory

    def _row(self, directory):
        plan = read_json(directory/'plan.json')
        state = read_json(directory/'state.json', {})
        process = read_json(directory/'process.json', {})
        if state.get('status') in ('running', 'starting', 'stopping') and process:
            if process_start(process['pid']) != process.get('start'):
                state = dict(state, status='interrupted', error='检查进程已退出；查看日志定位原因')
        return dict(id=directory.name, title=plan['title'], kind=plan['config']['kind'],
                    config=plan['config'], created_at=plan['created_at'], legacy=False, **state)

    def snapshot(self):
        jobs = [self._row(p.parent) for p in self.runs.glob('*/plan.json')]
        for key, (kind, title) in LEGACY.items():
            path = self.output/(key+'.json')
            if path.is_file():
                report = read_json(path)
                jobs.append(dict(id='saved-'+key, title=title, kind=kind, legacy=True,
                    status='complete' if report.get('passed') else 'failed', completed=1, total=1,
                    created_at=time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(path.stat().st_mtime)),
                    config={k: report[k] for k in ('steps', 'suite', 'arm') if k in report},
                    elapsed_seconds=report.get('elapsed_seconds')))
        return dict(jobs=sorted(jobs, key=lambda r: r['created_at'], reverse=True))

    def detail(self, request):
        run_id = request.get('run')
        if isinstance(run_id, str) and run_id.startswith('saved-'):
            key = run_id[6:]
            if key not in LEGACY or not (self.output/(key+'.json')).is_file():
                raise ValueError('历史验证记录不存在')
            row = next(r for r in self.snapshot()['jobs'] if r['id'] == run_id)
            return dict(job=row, reports=[dict(name=row['title'], data=read_json(self.output/(key+'.json')))],
                        log='历史验收报告，未关联网页运行日志。', plan=None)
        directory = self._directory(run_id)
        plan = read_json(directory/'plan.json')
        reports = [dict(name=s['name'], data=read_json(directory/s['report'])) for s in plan['stages']
                   if (directory/s['report']).is_file()]
        log = directory/'run.log'
        if not log.is_file():
            log = directory/'worker.log'
        text = ''
        if log.is_file():
            with log.open('rb') as stream:
                stream.seek(max(0, log.stat().st_size-32000))
                text = stream.read().decode(errors='replace')
        return dict(job=self._row(directory), reports=reports, log=text, plan=plan)

    def stop(self, request):
        with self.lock:
            directory = self._directory(request.get('run'))
            process = read_json(directory/'process.json', {})
            row = self._row(directory)
            if row['status'] not in ('starting', 'running', 'stopping'):
                raise ValueError('该检查已结束')
            if not process.get('start') or process_start(process['pid']) != process['start']:
                raise ValueError('原检查进程已退出')
            # Only this job's dedicated session; no training/policy processes.
            (directory/'stop_requested').touch()
            os.killpg(process['pid'], signal.SIGTERM)
            return dict(id=directory.name, status='stopping')

    def image_path(self, request):
        directory = self._directory(request.get('run'))
        name = request.get('file', '')
        if not re.fullmatch(r'task-\d+-(frames|episode-\d+)/(start|end)-(agentview|robot0_eye_in_hand)\.png', name):
            raise ValueError('相机截图路径不正确')
        path = (directory/name).resolve()
        if not path.is_relative_to(directory) or not path.is_file():
            raise ValueError('相机截图不存在')
        return path

    def video_path(self, request):
        directory = self._directory(request.get('run'))
        name=request.get('file','')
        if not re.fullmatch(r'task-\d+-episode-\d+/rollout\.mp4',name):
            raise ValueError('录像路径不正确')
        path=(directory/name).resolve()
        if not path.is_relative_to(directory) or not path.is_file(): raise ValueError('录像不存在')
        return path
