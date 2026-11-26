"""Local quick-benchmark jobs and reports; no Torch or simulator imports."""
import ast
import hashlib
import json
import os
from pathlib import Path
import re
import shlex
import shutil
import signal
import subprocess
import sys
import threading
import time
import uuid

import libero_dashboard


TITLES = {'task40_scene_reset': 'Task40 · 桌椅跨回合重置检查',
          'network': '网络回归', 'action': '真实动作短训练', 'libero': 'LIBERO 环境检查',
          'v2_label_audit': 'V2 · 官方示教物理标签回放审计',
          'v2_state_decode': 'V2 · 官方物理状态反序列化审计',
          'v2_labels_export': 'V2 · 全部200条官方启动状态导出',
          'v2_train': 'V2 · 4090真实标签单轮验证训练',
          'v2_evaluate': 'V2 · 末尾权重集中评估',
          'v2_training_smoke': 'V2 · 4090 训练工程自检（人工状态测试值）'}
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
            'task40_scene_reset': [self.root/'scripts/b1k_python.sh',
                self.root/'evaluation/scripts/task40_scene_reset_check.py'],
            'v2_label_audit': [self.root/'scripts/b1k_python.sh',
                Path('/nvme3/envs/b1k2025-sim/bin/python'),
                self.root/'evaluation/scripts/v2_label_audit.py',
                Path('/nvme3/datasets/b1k2025_raw_task40_v2_audit/task-0040/episode_00400010.hdf5')],
            'network': [self.python, self.root/'outputs/diagnostics/v1_cognitive_16000_weights.pt',
                        self.root/'outputs/diagnostics/v1_real_replay_features.pt'],
            'action': [self.python, self.root/'checkpoints/groot_n1d6_base/config.json',
                       self.root/'outputs/diagnostics/v1_real_replay_features.pt'],
            'v2_training_smoke': [self.python, self.root/'src/v2_groot_memoryvla/training_smoke.py',
                       self.root/'checkpoints/v1__test01__exp11__step006000/model.safetensors.index.json',
                       self.root/'assets/datasets/b1k2025_groot_task40_v1_200/meta/info.json'],
            'libero': [self.root/'env/libero-smoke/bin/python',
                       self.root/'third_party/LIBERO/libero/libero/assets']}
        catalog = [dict(id=key, title=TITLES[key], available=all(p.exists() for p in paths),
                        missing=[str(p.relative_to(self.root)) for p in paths if not p.exists()])
                   for key, paths in needed.items()]
        catalog.append(dict(next(c for c in catalog if c['id']=='v2_label_audit'),
                            id='v2_state_decode', title=TITLES['v2_state_decode']))
        catalog.append(dict(next(c for c in catalog if c['id']=='v2_label_audit'),
                            id='v2_labels_export', title=TITLES['v2_labels_export']))
        training_files = [self.python,self.root/'src/v2_groot_memoryvla/task40_training.py',
            self.root/'evaluation/scripts/v2_certify_started.py',
            self.root/'checkpoints/v1__test01__exp11__step006000/config.json',
            self.root/'scripts/b1k_python.sh']
        catalog.append(dict(id='v2_train',title=TITLES['v2_train'],available=all(p.exists() for p in training_files),
            missing=[str(p.relative_to(self.root)) for p in training_files if not p.exists()]))
        evaluation_files = [self.python, self.root/'src/v2_groot_memoryvla/task40_evaluation.py',
            self.root/'evaluation/scripts/v2_task40_paired_evaluation.py',
            self.root/'Output/test_02/experiment_01/evidence/evaluation_protocol-v003.json']
        catalog.append(dict(id='v2_evaluate',title=TITLES['v2_evaluate'],
            available=all(p.is_file() for p in evaluation_files),
            missing=[str(p.relative_to(self.root)) for p in evaluation_files if not p.is_file()]))
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
        if isinstance(request,dict) and request.get('kind') in ('v3_preflight','v3_train','v3_evaluate'):
            from v3_workbench import prepare
            return prepare(self,request)
        if isinstance(request,dict) and isinstance(request.get('kind'),str) and request['kind'] in libero_dashboard.TITLES:
            return libero_dashboard.prepare(self,request)
        if not isinstance(request, dict) or set(request)-{'kind', 'gpu', 'steps', 'arm', 'suite', 'tasks', 'seed', 'notes', 'training_job', 'resume_evaluation_job', 'inject_reset_drift'}:
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
        min_mb = {'task40_scene_reset': 16000, 'network': 3000, 'action': 28000, 'libero': 2000, 'v2_training_smoke': 28000,
                  'v2_label_audit': 16000, 'v2_state_decode': 16000,
                  'v2_labels_export':16000,'v2_train':36000,'v2_evaluate':30000}[kind]
        config = dict(kind=kind, gpu=gpu, notes=notes.strip())
        py = str(self.python)
        replay = str(self.root/'outputs/diagnostics/v1_real_replay_features.pt')
        stages = []
        if kind == 'task40_scene_reset':
            arm = request.get('arm', 'original')
            if arm not in ('original', 'fixed'):
                raise ValueError('重置检查仅支持 original / fixed')
            config.update(arm=arm, category='reset_regression', model_version='v1',
                          test_id='test01', experiment_id='exp12')
            collision_steps = request.get('steps', 0)
            if type(collision_steps) is not int or collision_steps not in (0, 120):
                raise ValueError('家具重置检查仅支持位姿扰动0步或机器人碰撞120步')
            config['collision_steps'] = collision_steps
            inject = request.get('inject_reset_drift', False)
            if type(inject) is not bool:
                raise ValueError('稳定阶段故障注入必须是布尔值')
            config['inject_reset_drift'] = inject
            stages = [dict(name=TITLES[kind], argv=['env', '-u', 'CUDA_VISIBLE_DEVICES',
                f'OMNIGIBSON_GPU_ID={gpu}', 'OMNIGIBSON_APPDATA_PATH={run}/isaac_appdata',
                str(self.root/'scripts/b1k_python.sh'),
                str(self.root/'evaluation/scripts/task40_scene_reset_check.py'),
                '--arm', arm, '--collision-steps', str(collision_steps),
                '--output', '{run}/scene_reset.json'], report='scene_reset.json')]
            if inject:
                stages[0]['argv'].append('--inject-reset-drift')
        elif kind in ('v2_label_audit', 'v2_state_decode','v2_labels_export'):
            config.update(category='label_alignment_audit', model_version='v2', test_id='test02',
                          experiment_id='exp01', training_steps=0)
            stages = [dict(name=TITLES[kind], argv=['env', 'OMNIGIBSON_GPU_ID=0',
                'OMNIGIBSON_APPDATA_PATH={run}/isaac_appdata',
                str(self.root/'scripts/b1k_python.sh'),
                str(self.root/'evaluation/scripts/v2_label_audit.py'), '--output', '{run}/labels.json'],
                report='labels.json')]
            if kind in ('v2_state_decode','v2_labels_export'):
                stages[0]['argv'].append('--deserialize-only')
                if kind == 'v2_labels_export':
                    stages[0]['argv'].append('--all-episodes')
            else:
                base = stages[0]
                stages = [dict(base, name=f"{TITLES[kind]} · {eid}",
                    argv=base['argv'][:-1]+['{run}/labels-'+str(eid)+'.json', '--episode',str(eid)],
                    report=f'labels-{eid}.json') for eid in (400010,400020,403000)]
        elif kind == 'v2_evaluate':
            training = self._directory(request.get('training_job'))
            report = read_json(training/'v2_training.json', {})
            checkpoint = training/'v2__test02__exp01__run001__step001000'
            if (read_json(training/'plan.json')['config']['kind'] != 'v2_train'
                    or report.get('passed') is not True or report.get('steps') != 1000
                    or not (checkpoint/'COMPLETE.json').is_file()):
                raise ValueError('集中评估需要同一工作台1000步末尾完整重载通过的V2权重')
            experiment = self.root/'Output/test_02/experiment_01'
            protocol = experiment/'evidence/evaluation_protocol-v003.json'
            config.update(category='concentrated_evaluation',model_version='v2',test_id='test02',
                experiment_id='exp01',run_id='run001',training_job=training.name,
                checkpoint=str(checkpoint),protocol=str(protocol),
                protocol_sha256=hashlib.sha256(protocol.read_bytes()).hexdigest())
            stages = [dict(name='V2 · 留出集状态与100窗口机制诊断',
                argv=[py,'-m','v2_groot_memoryvla.task40_evaluation','--checkpoint',str(checkpoint),
                    '--dataset',str(self.root/'assets/datasets/b1k2025_groot_task40_v1_200'),
                    '--protocol',str(protocol),'--labels',str(experiment/'evidence/certified_started_labels.json'),
                    '--output','{run}/offline_evaluation.json'],report='offline_evaluation.json'),
                dict(name='V2 · 12组V1/V2同起点恢复配对',
                argv=[py,str(self.root/'evaluation/scripts/v2_task40_paired_evaluation.py'),
                    '--root',str(self.root),'--checkpoint',str(checkpoint),'--protocol',str(protocol),
                    '--output','{run}/paired_evaluation.json'],report='paired_evaluation.json')]
            resume_job = request.get('resume_evaluation_job')
            if resume_job is not None:
                previous = self._directory(resume_job)
                previous_plan = read_json(previous/'plan.json')
                previous_config = previous_plan.get('config', {})
                if (previous_config.get('kind') != 'v2_evaluate'
                        or previous_config.get('checkpoint') != str(checkpoint)
                        or previous_config.get('protocol_sha256') != config['protocol_sha256']
                        or not (previous/'paired_status.json').is_file()):
                    raise ValueError('续跑必须引用同权重、同锁定协议且已有配对状态的V2评估作业')
                stages = [dict(name='V2 · 续跑未完成的同起点恢复配对',
                    argv=[py,str(self.root/'evaluation/scripts/v2_task40_paired_evaluation.py'),
                        '--root',str(self.root),'--checkpoint',str(checkpoint),'--protocol',str(protocol),
                        '--resume-status',str(previous/'paired_status.json'),
                        '--output','{run}/paired_evaluation.json'],report='paired_evaluation.json')]
                config.update(resume_evaluation_job=previous.name,
                              category='paired_evaluation_continuation')
        elif kind == 'v2_train':
            steps = request.get('steps',1000)
            if type(steps) is not int or steps != 1000:
                raise ValueError('本轮V2验证训练已锁定1000 optimizer steps')
            checkpoint = str(self.root/'checkpoints/v1__test01__exp11__step006000')
            experiment = self.root/'Output/test_02/experiment_01'
            config.update(category='exploratory_validation',model_version='v2',test_id='test02',
                experiment_id='exp01',run_id='run001',steps=1000,checkpoint=checkpoint,
                labels='certified recorded ToggledOn; not synthetic',
                label_manifest=str(experiment/'evidence/certified_started_labels.json'))
            stages = [dict(name='V2 · 全部200条原始物理状态导出',argv=['env','OMNIGIBSON_GPU_ID=0',
                'OMNIGIBSON_APPDATA_PATH={run}/isaac_appdata',str(self.root/'scripts/b1k_python.sh'),
                str(self.root/'evaluation/scripts/v2_label_audit.py'),'--output','{run}/labels.json',
                '--deserialize-only','--all-episodes'],report='labels.json'),
                dict(name='V2 · 已审计started字段认证',argv=[py,str(self.root/'evaluation/scripts/v2_certify_started.py'),
                '--input','{run}/labels.json','--output',config['label_manifest']],report=config['label_manifest']),
                dict(name=TITLES[kind],argv=[py,'-m','v2_groot_memoryvla.task40_training',
                '--checkpoint',checkpoint,'--dataset',str(self.root/'assets/datasets/b1k2025_groot_task40_v1_200'),
                '--labels',config['label_manifest'],'--split',str(experiment/'evidence/data_split.json'),
                '--steps','1000','--output','{run}/v2_training.json'],report='v2_training.json',steps=1000)]
            reusable = self.runs/'20261123-232848-7ecd5d3e'
            if (reusable/'plan.json').is_file():
                stages[0]['argv'] += ['--reuse-dir',str(reusable)]
        elif kind == 'network':
            stages = [dict(name='代码回归', argv=[py, '-m', 'pytest', '-q',
                'src/v1_groot_memoryvla/tests', 'tests/test_groot_training_dashboard.py',
                '--junitxml={run}/unit.xml'], report='unit.json', unit=True),
                dict(name='真实 EOS 检索', argv=[py, '-m', 'v1_groot_memoryvla.audit.retrieval_benchmark',
                    '--weights', str(self.root/'outputs/diagnostics/v1_cognitive_16000_weights.pt'),
                    '--replay', replay, '--output', '{run}/retrieval.json'], report='retrieval.json')]
        elif kind == 'v2_training_smoke':
            steps = request.get('steps', 3)
            if type(steps) is not int or not 1 <= steps <= 100:
                raise ValueError('V2 工程自检步数应在 1–100 之间')
            checkpoint = str(self.root/'checkpoints/v1__test01__exp11__step006000')
            config.update(steps=steps, category='engineering_smoke', model_version='v2',
                          test_id='test02', experiment_id='exp01', checkpoint=checkpoint,
                          labels='artificial test values, NOT physical truth',
                          feature_source='fresh frozen checkpoint backbone', cached_backbone=True)
            stages = [dict(name=TITLES[kind], argv=[py, '-m', 'v2_groot_memoryvla.training_smoke',
                '--checkpoint', checkpoint, '--dataset',
                str(self.root/'assets/datasets/b1k2025_groot_task40_v1_200'), '--steps', str(steps),
                '--output', '{run}/v2_smoke.json'], report='v2_smoke.json', steps=steps)]
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
            is_v3=plan['config']['kind'] in ('v3_preflight','v3_train','v3_evaluate')
            if is_v3 and any(r['kind']==plan['config']['kind'] and r['status'] in ('running','starting','stopping')
                             for r in self.snapshot()['jobs']):
                raise ValueError('已有同类V3作业运行或排队，不能重复提交')
            if not is_v3 and any(r['status'] in ('running', 'starting', 'stopping')
                                and r.get('config', {}).get('gpu') == plan['config']['gpu']
                                for r in self.snapshot()['jobs']):
                raise ValueError('所选GPU已有快速验证正在运行，请选择另一张容量足够的GPU或等待原作业结束')
            run_id = time.strftime('%Y%m%d-%H%M%S')+'-'+uuid.uuid4().hex[:8]
            directory = self.runs/run_id
            directory.mkdir(parents=True)
            digest = hashlib.sha256()
            files = []
            for current, directories, names in os.walk(self.root/'src/v1_groot_memoryvla'):
                directories[:] = sorted(d for d in directories if d not in ('.venv', '__pycache__', '.git'))
                files.extend(Path(current)/name for name in names if name.endswith('.py'))
            if plan['config']['kind'] in ('v2_training_smoke','v2_train','v2_evaluate'):
                files.extend((self.root/'src/v2_groot_memoryvla').rglob('*.py'))
            files += [self.root/'scripts/libero_smoke.py', self.root/'scripts/libero_rollout.py',
                      Path(__file__), Path(__file__).with_name('benchmark_worker.py'), Path(libero_dashboard.__file__)]
            if plan['config']['kind'] == 'task40_scene_reset':
                files += [self.root/'evaluation/scripts/task40_scene_reset_check.py',
                          self.root/'scripts/b1k_python.sh']
                if plan['config']['arm'] == 'fixed':
                    files.append(self.root/'evaluation/scripts/groot_scene_reset.py')
            if plan['config']['kind'] in ('v2_label_audit', 'v2_state_decode','v2_labels_export','v2_train'):
                files += [self.root/'evaluation/scripts/v2_label_audit.py', self.root/'scripts/b1k_python.sh']
                files += [self.root/'evaluation/scripts/v2_recording_alignment.py']
            if plan['config']['kind'] == 'v2_train':
                files += [self.root/'evaluation/scripts/v2_certify_started.py']
            if plan['config']['kind'] == 'v2_evaluate':
                files += [self.root/'evaluation/scripts'/name for name in (
                    'v2_task40_policy.py','v2_task40_paired_evaluation.py','v2_lock_evaluation.py',
                    'task40_memory_instrumentation.py','task40_paired_sim.py')]
            if is_v3:
                files.extend((self.root/'src/v3_groot_role_memory').rglob('*.py'))
                files.extend((self.root/'evaluation/scripts').glob('v3_*.py'))
            for path in sorted(p for p in files if not any(x in p.parts for x in ('.venv', '__pycache__'))):
                digest.update(str(path.relative_to(self.root)).encode())
                digest.update(path.read_bytes())
            plan.update(id=run_id, created_at=time.strftime('%Y-%m-%d %H:%M:%S'), source_digest=digest.hexdigest())
            atomic_json(directory/'plan.json', plan)
            atomic_json(directory/'state.json', dict(status='starting', completed=0, total=len(plan['stages'])))
            env = dict(os.environ, CUDA_VISIBLE_DEVICES=str(plan['config']['gpu']),
                PYTHONPATH=os.pathsep.join((str(self.root/'src'),str(self.root/'evaluation/scripts'))),
                OMP_NUM_THREADS='4',OPENBLAS_NUM_THREADS='2',MKL_NUM_THREADS='4',
                PYTHONNOUSERSITE='1', NO_ALBUMENTATIONS_UPDATE='1',
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

    def delete(self, request):
        """Remove only a terminal V2 temporary smoke after its evidence is archived."""
        with self.lock:
            directory = self._directory(request.get('run'))
            row = self._row(directory)
            if row['kind'] not in ('v2_training_smoke','v3_preflight') or row['status'] in ('starting', 'running', 'stopping'):
                raise ValueError('仅可清理已结束的 V2/V3 临时工程自检')
            process = read_json(directory/'process.json', {})
            if process.get('start') and process_start(process['pid']) == process['start']:
                raise ValueError('等待原工程自检进程完全退出后再清理')
            test='test_03' if row['kind']=='v3_preflight' else 'test_02'
            evidence = (self.root/f'Output/{test}/experiment_01/evidence'/f"4090-{directory.name}.json")
            archived = read_json(evidence, {})
            if archived.get('job', {}).get('id') != directory.name or not archived.get('log'):
                raise ValueError('先归档作业 ID、配置、结果和日志到当前实验')
            shutil.rmtree(directory)
            return dict(id=row['id'], status='deleted', removed=str(directory), evidence=str(evidence))

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
