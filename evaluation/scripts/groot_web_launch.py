"""Small web adapter for the canonical evaluation CLI; no simulator imports."""

import json
import math
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
from urllib.parse import urlencode

from groot_tasks import TASK_NAMES
from groot_eval_control import EvaluationControl
sys.path.insert(0, str(Path(__file__).resolve().parents[2]/'src'))
from v1_2_groot_baseline import contracts as v12_contracts
sys.path.insert(0, str(Path(__file__).resolve().parents[2]/'train'))
from run_identity import canonical_version


TASK40_EXPERIMENT_KINDS = ('task40_collect', 'task40_forward')
TASK40_DEFAULT_EXPERIMENT = 'experiment_09'
TASK40_NEXT_EXPERIMENT = 'experiment_10'
TASK40_BASELINE_EXPERIMENT = 'experiment_11'
TASK40_OPENVLA_EXPERIMENT = 'experiment_12'


def task40_approach_distance(truth):
    """Pre-contact proximity of the nearest hand to the microwave door."""
    distances = truth['interaction']['door_distance_m']
    return min(distances.values(), default=math.inf)


def task40_artifacts(output):
    output = Path(output).resolve()
    files = {'job.json': '运行配置', 'results.json': '轨迹结果',
             'logs/simulator.log': '仿真日志', 'logs/forward.log': '前向日志', 'logs/failure.txt': '异常记录',
             'metrics/gpu.csv': 'GPU 采样', 'metrics/repeat_audit.json': '恢复重复性对照',
             'metrics/fixed_action_audit.json': '固定动作恢复对照',
             'metrics/accumulation_reset_audit.json': '清除渲染历史对照'}
    for path in sorted((output / 'metrics').glob('restore_*/lineage.json')):
        files[path.relative_to(output).as_posix()] = '恢复轨迹来源'
    for path in sorted((output / 'metrics').glob('episode_*/controlled_reclosure.json')):
        files[path.relative_to(output).as_posix()] = f'{path.parent.name} 受控重关记录'
    if (output / 'results.json').is_file():
        files.update({path.relative_to(output).as_posix(): path.name
                      for path in sorted((output / 'videos').glob('*.mp4'))})
    return [{'path': relative, 'label': label,
             'url': '/api/launch/artifact?' + urlencode({'run': output.name, 'path': relative})}
            for relative, label in files.items()
            if (output / relative).resolve().is_relative_to(output)
            and (output / relative).is_file()]


def v12_artifacts(output):
    """Expose native BEHAVIOR-1K JSON/video files without rewriting them."""
    output = Path(output).resolve()
    files = {}
    for path in sorted(output.glob("json/*.json")):
        files[path.relative_to(output).as_posix()] = "native rollout JSON"
    for path in sorted(output.glob("videos/*")):
        if path.is_file():
            files[path.relative_to(output).as_posix()] = "native rollout video"
    for name in ("job.json", "runner.json", "server.log", "evaluator.log"):
        if (output / name).is_file():
            files[name] = "V1.2 run metadata" if name in ("job.json", "runner.json") else "V1.2 process log"
    return [{'path': relative, 'label': label,
             'url': '/api/launch/artifact?' + urlencode({'run': output.name, 'path': relative})}
            for relative, label in files.items()]


class LaunchManager:
    def __init__(self, project_root, log_root=None):
        self.root = Path(project_root).resolve()
        self.evaluation = Path(os.environ.get('GROOT_EVAL_ROOT', str(self.root/'evaluation'))).resolve()
        self.script = self.root / "evaluation/scripts/run_groot_eval.sh"
        self.log_root = Path(log_root or "/nvme3/tmp/memory_vla/web_launch")
        self.jobs = []
        self.lock = threading.Lock()
        self.policy_resolver = None
        self.control = EvaluationControl(self.root, self.evaluation)
        self.task40_worker = self.root / 'evaluation/scripts/task40_restore_worker.py'
        self.task40_forward_worker = self.root / 'evaluation/scripts/task40_forward_worker.py'
        self.task40_runner = Path(os.environ.get('GROOT_TASK40_RUNNER', self.root / 'evaluation/scripts'))
        self.task40_output = self.root / 'Output/test_01/experiment_07/workbench_runs'
        self.task40_experiment = Path(os.environ.get('GROOT_TASK40_EXPERIMENT_ROOT',
            self.root / 'Output/test_01/experiment_09'))

    def pause(self, request):
        with self.lock:
            return self.control.pause(request)

    def finish(self, request):
        with self.lock:
            return self.control.finish(request)

    def task40_root(self, group=TASK40_DEFAULT_EXPERIMENT):
        if group == TASK40_DEFAULT_EXPERIMENT:
            return self.task40_experiment
        if group == TASK40_NEXT_EXPERIMENT:
            return self.task40_experiment.with_name(TASK40_NEXT_EXPERIMENT)
        if group == TASK40_BASELINE_EXPERIMENT:
            storage = os.environ.get('GROOT_TASK40_BASELINE_STORAGE')
            if storage:
                return Path(storage)/group
            return self.task40_experiment.with_name(TASK40_BASELINE_EXPERIMENT)
        if group == TASK40_OPENVLA_EXPERIMENT:
            return self.task40_experiment.with_name(TASK40_OPENVLA_EXPERIMENT)
        raise ValueError('不支持的 Task40 实验目录')

    def delete(self, request):
        with self.lock:
            result = self.control.delete(request)
            self.jobs = [job for job in self.jobs if self._status(job)['folder'] != request['run']]
            return result

    def options(self):
        evaluation = self.evaluation
        versions = {}
        for path in evaluation.iterdir():
            if path.is_dir() and re.fullmatch(r"V_\d+", path.name):
                versions[path.name] = sorted(p.name for p in path.iterdir()
                                             if p.is_dir() and not p.name.startswith("."))
        return {"tasks": [{"id": task, "name": name} for task, name in TASK_NAMES.items()],
                "versions": versions, "evaluation_root": str(evaluation),
                "model_versions": {"V_1_2": {"protocol": v12_contracts.B1K2026,
                                               "task": "turning_on_radio",
                                               "launch_ready": self._v12_paths_ready()}},
                "task40_collect": {"sim_slots": {str(gpu): 2 for gpu in (0, 1)
                    if (self.task40_experiment / 'metrics' / f'dual_gpu{gpu}_validated.json').is_file()}}}

    @staticmethod
    def _v12_config():
        """Resolve only explicit V1.2 paths; never reuse the 2025 runtime."""
        return {
            'python': Path(os.environ.get('V12_GROOT_PYTHON',
                '/nvme3/projects/GR00T/Isaac-GR00T-n1d7-ace36d9/.venv/bin/python')),
            'upstream': Path(os.environ.get('V12_GROOT_ROOT',
                '/nvme3/projects/GR00T/Isaac-GR00T-n1d7-ace36d9')),
            'behavior_python': Path(os.environ.get('V12_BEHAVIOR_PYTHON',
                '/nvme3/envs/b1k2026-sim/bin/python')),
            'behavior_root': Path(os.environ.get('V12_BEHAVIOR_ROOT',
                '/nvme3/projects/GR00T/BEHAVIOR-1K-2026-a8247a8')),
            'checkpoint': Path(os.environ.get('V12_CHECKPOINT',
                '/nvme3/checkpoints/memory_vla/official-n17-radio/turning_on_radio_GR00T-checkpoint-150000')),
            'checkpoint_sha256': os.environ.get('V12_CHECKPOINT_SHA256',
                'cd97a22ff74bd91838350e18fe94b48e1902d407f1256f5d48eb20befe277dc3'),
        }

    def _v12_paths_ready(self):
        config = self._v12_config()
        return all(path.is_file() if key.endswith('python') else path.is_dir()
                   for key, path in config.items()
                   if key in ('python', 'upstream', 'behavior_python', 'behavior_root', 'checkpoint'))

    def _v12_prepare(self, request):
        allowed = {'version', 'protocol', 'task', 'instance', 'rollouts', 'gpu_index',
                   'policy_host', 'policy_port', 'run_tag', 'notes', 'record_video',
                   'wait_for_gpu'}
        unknown = set(request) - allowed
        if unknown:
            raise ValueError('V1.2 不支持的启动参数：' + ', '.join(sorted(unknown)))
        if canonical_version(request.get('version')) != v12_contracts.VERSION:
            raise ValueError('V1.2 请求必须显式使用 V_1_2')
        if request.get('protocol') != v12_contracts.B1K2026:
            raise ValueError('V1.2 必须使用原生 B1K-2026 WebSocket 协议')
        if request.get('task', 'turning_on_radio') != 'turning_on_radio':
            raise ValueError('官方 radio 权重只允许 turning_on_radio')
        instance = request.get('instance', 0)
        rollouts = request.get('rollouts', 1)
        gpu = request.get('gpu_index', 0)
        if type(instance) is not int or not 0 <= instance < 20:
            raise ValueError('public-test instance 必须为 0–19 的整数')
        if type(rollouts) is not int or rollouts != 1:
            raise ValueError('V1.2 验收只允许一个 rollout')
        if type(gpu) is not int or gpu < 0:
            raise ValueError('GPU 编号必须为非负整数')
        config = self._v12_config()
        if not self._v12_paths_ready():
            raise ValueError('V1.2 N1.7/B1K2026 独立环境或源码尚未就绪')
        output = self.evaluation / 'V_1_2' / 'official' / f'public_test_instance_{instance:02d}'
        output.mkdir(parents=True, exist_ok=True)
        if any(output.iterdir()):
            raise ValueError('V1.2 验收输出必须是全新目录：' + str(output))
        host = request.get('policy_host', '127.0.0.1')
        port = request.get('policy_port', 8000)
        endpoint = dict(host=host, port=port, state='ready', version=v12_contracts.VERSION,
                        method='official', task_name='turning_on_radio', protocol=v12_contracts.B1K2026,
                        checkpoint_sha256=config['checkpoint_sha256'])
        spec = v12_contracts.official_serve_spec(
            python=config['python'], upstream=config['upstream'], checkpoint=config['checkpoint'],
            host=host, port=port, gpu_index=gpu,
            verified_radio_sha256=config['checkpoint_sha256'])
        eval_spec = v12_contracts.official_eval_spec(
            python=config['behavior_python'], behavior_root=config['behavior_root'], output=output,
            endpoint=endpoint, instance=instance, rollouts=rollouts)
        note = request.get('notes') or 'V1.2 / N1.7 / B1K2026：official radio, RTC off'
        if not isinstance(note, str) or not 1 <= len(note) <= 200:
            raise ValueError('V1.2 notes 必须是 1–200 字符')
        plan = dict(kind='b1k2026_websocket', version=v12_contracts.VERSION,
                    protocol=v12_contracts.B1K2026, task_id='turning_on_radio',
                    instance=instance, rollouts=rollouts, output=str(output),
                    gpus=[gpu], gpu_index=gpu, instances={str(gpu): [instance]}, notes=note,
                    policy_identity=spec, evaluator=eval_spec,
                    source_revision=spec['source']['revision'],
                    checkpoint_sha256=spec['checkpoint_sha256'], rtc=False,
                    official=True)
        runner = self.root / 'evaluation/scripts/run_v12_b1k2026.py'
        argv = [str(config['behavior_python']), '-u', str(runner),
                '--serve-spec', json.dumps(spec, ensure_ascii=False),
                '--eval-spec', json.dumps(eval_spec, ensure_ascii=False),
                '--output', str(output)]
        return {'argv': argv, 'command': shlex.join(argv), 'plan': plan}

    def argv(self, request):
        if not isinstance(request, dict):
            raise ValueError("启动参数必须是 JSON 对象")
        if request.get('task40_restore') is True:
            return self._task40_restore_argv(request)
        if request.get('task40_collect') is True:
            return self._task40_collect(request)
        if request.get('task40_forward') is True:
            return self._task40_forward(request)
        scalars = {"version", "group", "model_name", "mode", "gpu_index", "task", "preset",
                   "seed", "seed_mode", "policy_host", "policy_port", "run_tag",
                   "episodes_per_gpu", "repetitions", "additional_episodes", "debug_max_steps", "video_steps_per_render", "policy_checkpoint", "training_id", "action_steps"}
        booleans = {"record_video", "wait_for_gpu", "diagnostics"}
        arrays = {"instances_gpu0", "instances_gpu1"}
        unknown = set(request) - scalars - booleans - arrays - {"resume_from"}
        if unknown:
            raise ValueError("不支持的启动参数：" + ", ".join(sorted(unknown)))
        argv = ["bash", str(self.script)]
        resume = request.get("resume_from")
        if resume:
            base = self.evaluation
            path = Path(resume)
            if not path.is_absolute():
                path = base / (Path(*path.parts[1:]) if path.parts[0] == 'evaluation' else path)
            path = path.resolve()
            if not path.is_relative_to(base.resolve()):
                raise ValueError("续跑目录必须位于本项目 evaluation 下")
            argv += ["--resume-from", str(path)]
            # Preserve the saved protocol, GPU assignment, model endpoint and
            # video settings. The resume form sends no default overrides.
            if any(key not in {"resume_from", "policy_host", "policy_port", "additional_episodes"} for key in request):
                raise ValueError("继续评估沿用原配置，可指定本次回合数及推理地址和端口")
            if 'additional_episodes' in request and (type(request['additional_episodes']) is not int or request['additional_episodes'] < 1):
                raise ValueError('本次回合数必须为正整数')
            _, active = self.control.processes(path)
            if active:
                raise ValueError('评估仍在运行，请先暂停并等待停止后继续')
        for key in sorted(scalars):
            value = request.get(key)
            if key == 'version':
                value = canonical_version(value)
            if value is None or value == "":
                continue
            if isinstance(value, (list, dict, bool)) or len(str(value)) > 256:
                raise ValueError(f"参数 {key} 格式不正确")
            if key == "gpu_index" and str(request.get("mode", "single1")).startswith("dual"):
                continue
            if key == "policy_host" and not re.fullmatch(r"[A-Za-z0-9_.:-]+", str(value)):
                raise ValueError("推理主机格式不正确")
            argv += ["--" + key.replace("_", "-"), str(value)]
        for key in sorted(booleans):
            if key not in request:
                continue
            if not isinstance(request[key], bool):
                raise ValueError(f"参数 {key} 必须是开关")
            if key == "record_video":
                argv.append("--record-video" if request[key] else "--no-record-video")
            elif request[key]:
                argv.append("--" + key.replace("_", "-"))
        for key in sorted(arrays):
            if request.get(key) in (None, []):
                continue
            values = request[key]
            if not isinstance(values, list) or any(type(v) is not int for v in values):
                raise ValueError(f"参数 {key} 必须是实例编号列表")
            argv += ["--" + key.replace("_", "-"), *map(str, values)]
        return argv

    def _task40_path(self, value, label, root=None):
        root = self.root if root is None else root
        if not isinstance(value, str) or not value or len(value) > 512:
            raise ValueError(f'{label}路径不能为空')
        path = Path(value).expanduser()
        if not path.is_absolute():
            path = root / path
        path = path.resolve(strict=True)
        if not path.is_relative_to(root):
            raise ValueError(f'{label}必须位于当前项目目录内')
        return path

    def _task40_restore_argv(self, request):
        allowed = {'task40_restore', 'restore_checkpoint', 'restore_weights', 'sim_gpu',
                   'restore_seed', 'restore_cycles', 'restore_port', 'memory_mode'}
        if set(request) - allowed:
            raise ValueError('Task40 检查点恢复不接受普通评估参数')
        anchor = self._task40_path(request.get('restore_checkpoint'), '检查点')
        weights = self._task40_path(request.get('restore_weights'), '模型权重')
        required = ('start.json', 'anchor.json', 'sim_checkpoint.pt', 'memory_online.pt',
                    'anchor_live_observation.npz', 'prefix')
        missing = [name for name in required if not (anchor / name).exists()]
        if missing:
            raise ValueError('检查点目录不完整，缺少：' + ', '.join(missing))
        with (anchor / 'start.json').open() as stream:
            start = json.load(stream)
        with (anchor / 'sim_checkpoint.pt').open('rb') as stream:
            if not stream.read(1):
                raise ValueError('仿真检查点文件为空')
        prefix = anchor / 'prefix'
        prefix_decisions = sorted(int(path.name) for path in prefix.iterdir()
                                  if path.is_dir() and path.name.isdigit()
                                  and (path / 'decision.json').is_file())
        if not prefix_decisions or prefix_decisions != list(range(len(prefix_decisions))):
            raise ValueError('检查点缺少可回放的策略历史前缀')
        if not (weights / 'config.json').is_file() or not (weights / 'model.safetensors.index.json').is_file():
            raise ValueError('模型权重目录缺少 config.json 或 safetensors 索引')
        if not self.task40_worker.is_file() or not self.task40_runner.joinpath('paired_sim_diagnostic.py').is_file():
            raise ValueError('Task40 检查点恢复运行脚本缺失')
        sim_gpu = request.get('sim_gpu', 0)
        if type(sim_gpu) is not int or sim_gpu not in (0, 1):
            raise ValueError('仿真 GPU 必须是 0 或 1')
        policy_host = os.environ.get('GROOT_TASK40_POLICY_HOST', '172.28.62.17')
        seed = request.get('restore_seed', 73002)
        cycles = request.get('restore_cycles', 10)
        port = request.get('restore_port', 5586)
        mode = request.get('memory_mode', 'A_normal')
        if type(seed) is not int or not 0 <= seed < 2**32:
            raise ValueError('策略随机种子必须在 0 到 2^32-1 之间')
        if type(cycles) is not int or not 1 <= cycles <= 60:
            raise ValueError('连续决策数必须在 1 到 60 之间')
        if type(port) is not int or not 1024 <= port <= 65535:
            raise ValueError('策略端口必须在 1024 到 65535 之间')
        if mode not in ('A_normal', 'B_empty'):
            raise ValueError('记忆模式仅支持 A_normal 或 B_empty')
        run_id = time.strftime('%Y%m%d-%H%M%S') + '-' + uuid.uuid4().hex[:8]
        output = self.task40_output / run_id
        plan = {'kind': 'task40_restore', 'task_id': 40, 'version': 'Task40',
                'group': 'checkpoint_restore', 'output': str(output),
                'instances': {str(sim_gpu): [0]}, 'policy_host': policy_host,
                'restore_anchor': str(anchor), 'weights': str(weights), 'seed': seed,
                'cycles': cycles, 'memory_mode': mode, 'policy_port': port}
        argv = ['python3', '-u', str(self.task40_worker), '--root', str(self.root),
                '--output', str(output), '--anchor', str(anchor), '--weights', str(weights),
                '--sim-gpu', str(sim_gpu), '--policy-host', policy_host, '--seed', str(seed),
                '--cycles', str(cycles), '--memory-mode', mode, '--port', str(port)]
        return {'argv': argv, 'command': shlex.join(argv), 'plan': plan}

    def prepare(self, request):
        if not isinstance(request, dict):
            raise ValueError("启动参数必须是 JSON 对象")
        if request.get('task40_restore') is True:
            return self._task40_restore_argv(request)
        if request.get('task40_collect') is True:
            return self._task40_collect(request)
        if request.get('task40_forward') is True:
            return self._task40_forward(request)
        if canonical_version(request.get('version')) == 'V_1_2':
            return self._v12_prepare(request)
        argv = self.argv(request)
        identity=None
        if self.policy_resolver:
            host,port=request.get('policy_host','172.28.62.17'),request.get('policy_port',5556)
            if request.get('resume_from'):
                path=Path(argv[argv.index('--resume-from')+1])
                config=json.loads((path/'results.json').read_text())['configuration']
                saved_host,saved_port=config['policy_server'].rsplit(':',1)
                host,port=request.get('policy_host',saved_host),request.get('policy_port',int(saved_port))
            identity=self.policy_resolver(host,int(port))
            if identity:
                if canonical_version(identity.get('version'))=='V_1_2' or identity.get('protocol')=='b1k2026_websocket':
                    raise ValueError('N1.7 websocket service cannot be evaluated by the legacy B1K2025 runner')
                group=request.get('group')
                if not request.get('resume_from') and group in ('baseline','memory','official') and identity.get('method') and group!=identity['method']:
                    raise ValueError('结果方法 '+group+' 与推理服务实际方法 '+identity['method']+' 不一致，请选择对应分组')
                if not request.get('resume_from'):
                    if not group and identity.get('method'):
                        argv += ['--group',identity['method']]
                    identity_version = canonical_version(identity.get('version')) or ''
                    if (not request.get('version') and
                            (re.fullmatch(r'V_[0-9]+(?:_[0-9]+)?', identity_version) or
                             re.fullmatch(r'V[0-9]+_[0-9]+', identity_version))):
                        argv += ['--version',identity_version]
                argv += ['--policy-identity',json.dumps(identity,ensure_ascii=False)]
        result = subprocess.run(argv + ["--dry-run"], cwd=self.root, stdin=subprocess.DEVNULL,
                                capture_output=True, text=True, timeout=20)
        if result.returncode:
            raise ValueError((result.stderr or result.stdout).strip().splitlines()[-1])
        try:
            plan, _ = json.JSONDecoder().raw_decode(result.stdout.lstrip())
        except ValueError as error:
            raise ValueError("启动脚本未返回有效运行计划") from error
        if request.get("seed") == "random":
            argv[argv.index("--seed") + 1] = str(plan["base_seed"])
        return {"argv": argv, "command": shlex.join(argv), "plan": plan}

    def _task40_collect(self, request):
        allowed = {'task40_collect', 'sim_gpu', 'seed', 'cycles', 'instances', 'policy_host',
                   'policy_port', 'sim_slots', 'repetitions', 'restore_checkpoint', 'restore_repeats',
                   'controlled_reclosure', 'full_task_eval', 'capture_approach', 'experiment', 'renderer_aa',
                   'renderer_probe', 'renderer_gi_history', 'renderer_disable_indirect_diffuse'}
        if set(request) - allowed:
            raise ValueError('不支持的 Task40 采集参数')
        directed = None
        group = request.get('experiment', TASK40_DEFAULT_EXPERIMENT)
        experiment = self.task40_root(group)
        if 'restore_checkpoint' in request:
            archive = os.environ.get('GROOT_TASK40_ARCHIVE_ROOT')
            remote_archive = Path(archive) if archive and group == TASK40_BASELINE_EXPERIMENT else None
            if remote_archive is None and self.task40_experiment.resolve() != self.root / 'Output/test_01/experiment_09':
                raise ValueError('Task40 定向恢复仅支持本机评估')
            draft = self._task40_path(request['restore_checkpoint'], '失败检查点', root=remote_archive)
            directed = self._task40_failure_lineage(draft, archive_root=remote_archive)
        gpu = request.get('sim_gpu', 0)
        seed = request.get('seed', directed['source_seed'] if directed else None)
        cycles = request.get('cycles', 60 if directed else 300)
        slots = request.get('sim_slots', 1)
        repetitions = request.get('repetitions', 1)
        restore_repeats = request.get('restore_repeats', 1)
        renderer_aa = request.get('renderer_aa')
        renderer_probe = request.get('renderer_probe', False)
        gi_history = request.get('renderer_gi_history')
        disable_indirect_diffuse = request.get('renderer_disable_indirect_diffuse', False)
        controlled = request.get('controlled_reclosure', False)
        full_task_eval = request.get('full_task_eval', False)
        capture_approach = request.get('capture_approach', False)
        instances = request.get('instances', [directed['instance']] if directed else None)
        host = request.get('policy_host', '172.28.62.17')
        port = request.get('policy_port')
        if type(gpu) is not int or gpu not in (0, 1):
            raise ValueError('仿真 GPU 必须是 0 或 1')
        if type(slots) is not int or slots not in (1, 2):
            raise ValueError('Task40 单卡环境数只能为 1 或 2')
        if not isinstance(instances, list) or not 1 <= len(instances) <= 10 or any(type(i) is not int or i not in range(10) for i in instances) or len(set(instances)) != len(instances):
            raise ValueError('请选择 1–10 个不同的 Task40 实例')
        if type(repetitions) is not int or not 1 <= repetitions <= 10 // len(instances):
            raise ValueError('每个作业最多执行 10 条轨迹')
        if type(seed) is not int or not 0 <= seed <= 2**32 - len(instances) * repetitions:
            raise ValueError('采集种子超出范围')
        if type(cycles) is not int or not 1 <= cycles <= 300:
            raise ValueError('决策数必须在 1 到 300 之间')
        if 'restore_repeats' in request and (not directed or type(restore_repeats) is not int
                                             or restore_repeats not in (1, 2)):
            raise ValueError('Task40 恢复重复次数仅支持对同一检查点恢复 1 或 2 次')
        if 'renderer_aa' in request and (not directed or restore_repeats != 2
                                         or type(renderer_aa) is not int or renderer_aa not in (0, 3)):
            raise ValueError('渲染 AA 仅支持双次恢复诊断，取值为 0 或 3')
        if type(renderer_probe) is not bool or (renderer_probe and (
                not directed or restore_repeats != 2 or 'policy_port' in request
                or 'policy_host' in request or not 2 <= cycles <= 10)):
            raise ValueError('固定动作诊断要求恢复同一检查点两次、重放 2–10 个动作块，且不连接推理服务')
        if 'renderer_gi_history' in request and (not renderer_probe or type(gi_history) is not int or gi_history != 1):
            raise ValueError('间接光照历史长度 1 仅用于固定动作诊断')
        if 'renderer_disable_indirect_diffuse' in request and (
                not renderer_probe or type(disable_indirect_diffuse) is not bool):
            raise ValueError('关闭间接漫反射仅用于固定动作诊断')
        if type(controlled) is not bool or (controlled and directed):
            raise ValueError('Task40 受控重关必须从任务初始状态开始')
        if type(full_task_eval) is not bool or (full_task_eval and (
                directed or controlled or group not in (TASK40_NEXT_EXPERIMENT, TASK40_BASELINE_EXPERIMENT))):
            raise ValueError('完整任务评估只支持指定实验中的初始状态对照')
        if directed and (instances != [directed['instance']] or seed != directed['source_seed']
                         or repetitions != 1 or cycles > 60):
            raise ValueError('Task40 定向恢复限原父轨迹实例和种子、单环境、每次最多 60 个决策')
        if not renderer_probe and (not isinstance(host, str) or
                (host != '172.28.62.17' and group != TASK40_OPENVLA_EXPERIMENT)
                or type(port) is not int or not 1024 <= port <= 65535):
            raise ValueError('请选择已就绪的 Task40 诊断推理端口')
        identity = self.policy_resolver(host, port) if not renderer_probe and self.policy_resolver else None
        official = group in (TASK40_BASELINE_EXPERIMENT, TASK40_OPENVLA_EXPERIMENT) and (
            identity or {}).get('method') == 'official'
        rebuild = bool(directed and group in (TASK40_BASELINE_EXPERIMENT, TASK40_OPENVLA_EXPERIMENT) and not official)
        if not renderer_probe and (not identity or not (identity.get('task40_diagnostics') or official)):
            raise ValueError('该端口未运行 Task40 诊断推理服务')
        if type(capture_approach) is not bool or (capture_approach and (
                group != TASK40_BASELINE_EXPERIMENT or directed or controlled or full_task_eval
                or official or not identity.get('task40_trace'))):
            raise ValueError('首次门前检查点只从 Experiment 11 V1 的真实新轨迹采集')
        if not renderer_probe and (controlled or group == TASK40_NEXT_EXPERIMENT) and not identity.get('task40_trace'):
            raise ValueError('该 Task40 采集配置要求使用增强诊断推理端口')
        if directed and not renderer_probe and not official and (
                not identity.get('task40_trace') or (not rebuild and identity.get('checkpoint') != directed['policy_weights'])):
            raise ValueError('Task40 定向恢复要求使用与父轨迹权重一致的增强诊断推理端口')
        anchor = Path(os.environ.get('GROOT_TASK40_ANCHOR',
            self.root / 'Output/test_01/experiment_01/checkpoints')).resolve(strict=True)
        # The Experiment 12 worker uses RPC. The checkpoint path is policy
        # provenance on the inference host, including for initial collection.
        if group == TASK40_OPENVLA_EXPERIMENT:
            remote_weights = (identity or {}).get('checkpoint') or (directed or {}).get('policy_weights')
            if not remote_weights:
                raise ValueError('Experiment 12 需要推理服务登记实际 checkpoint')
            weights = Path(remote_weights)
        else:
            weights = Path(os.environ.get('GROOT_TASK40_WEIGHTS',
                self.root / 'Output/task40_memory_overnight/replay/original_step4000')).resolve(strict=True)
        output = experiment / 'runs' / uuid.uuid4().hex[:12]
        plan = {'kind': 'task40_collect', 'task_id': 40, 'version': (identity or {}).get('version'),
                'group': group, 'output': str(output), 'instances': {str(gpu): instances},
                'seed': seed, 'cycles': cycles, 'sim_slots': slots, 'repetitions': repetitions,
                'policy_host': host, 'policy_port': port,
                'policy_identity': identity}
        if rebuild:
            plan['rebuild_policy_history'] = True
        if official:
            plan['official_policy'] = True
        argv = ['python3', '-u', str(self.task40_worker), '--root', str(self.root),
                '--output', str(output), '--anchor', str(anchor), '--weights', str(weights),
                '--sim-gpu', str(gpu), '--policy-host', host, '--port', str(port or 0),
                '--seed', str(seed), '--cycles', str(cycles)]
        if directed:
            plan.update(phase='directed_restore', restore_checkpoint=str(draft),
                        lineage=directed, restore_repeats=restore_repeats)
            if group in (TASK40_BASELINE_EXPERIMENT, TASK40_OPENVLA_EXPERIMENT):
                plan['stop_at_first_open'] = True
            if restore_repeats == 2:
                argv += ['--restore-repeats', '2', '--renderer-audit']
                if renderer_aa is not None:
                    plan['renderer_aa'] = renderer_aa
                    argv += ['--renderer-aa', str(renderer_aa)]
            if renderer_probe:
                plan.update(phase='renderer_probe', policy_host=None,
                            policy_identity={'checkpoint': directed['policy_weights'], 'binding': 'archived_actions'})
                if gi_history is not None:
                    plan['renderer_gi_history'] = gi_history
                if disable_indirect_diffuse:
                    plan['renderer_disable_indirect_diffuse'] = True
        else:
            argv += ['--collect-instances', *map(str, instances), '--collect-repetitions', str(repetitions)]
            if controlled:
                plan.update(controlled_reclosure=True, phase='controlled_reclosure')
            if full_task_eval:
                plan.update(full_task_eval=True, phase='full_task_eval')
            if capture_approach:
                plan.update(capture_approach=True, phase='approach_capture')
        argv += ['--experiment-plan', json.dumps(plan, ensure_ascii=False)]
        return {'argv': argv, 'command': shlex.join(argv), 'plan': plan}

    def _task40_failure_lineage(self, checkpoint, archive_root=None):
        run = checkpoint.parent.parent
        exp = archive_root/'Output/test_01' if archive_root is not None else self.root/'Output/test_01'
        valid_runs = {exp/'experiment_07/episodes', exp/'experiment_09/runs', exp/'experiment_10/runs'}
        baseline_runs = exp/TASK40_BASELINE_EXPERIMENT/'runs' if archive_root is not None else self.task40_root(TASK40_BASELINE_EXPERIMENT)/'runs'
        valid_runs.add(baseline_runs)
        if archive_root is None:
            valid_runs.update((self.task40_experiment/'runs', self.task40_root(TASK40_NEXT_EXPERIMENT)/'runs'))
        if checkpoint.parent.name != 'checkpoints' or run.parent not in valid_runs:
            raise ValueError('失败检查点必须位于 Exp07 或 Exp09 的原始轨迹目录')

        def archive_file(folder, filename):
            file = folder / filename
            if not file.is_file() or file.resolve() != file or not file.stat().st_size:
                raise ValueError(f'Task40 归档文件缺失、为空或不在原目录：{file}')
            return file

        def read(folder, filename):
            return json.loads(archive_file(folder, filename).read_text())

        for filename in ('sim.pt', 'memory.pt', 'policy_rng_online.pt', 'observation.npz', 'physical.json'):
            archive_file(checkpoint, filename)
        try:
            source = read(checkpoint, 'source.json')
            branch, decision, step = (source[key] for key in ('branch', 'decision', 'step'))
            match = re.fullmatch(r'episode_i([0-9])_s([0-9]{1,10})_A_normal', branch)
            if not match or type(decision) is not int or decision < 1 or type(step) is not int or step != decision * 8:
                raise ValueError('失败检查点必须关联任务初始状态的完整策略决策及对应物理步')
            instance, seed = map(int, match.groups())
            parent = run / 'metrics' / branch
            trial = read(parent, 'trial.json')
            initial = read(parent, 'initial.json')
            if (trial['branch'], trial['instance'], trial['seed'], trial['start_step'], trial['mode'], trial['origin']) != (
                    branch, instance, seed, 0, 'A_normal', None) or not step < trial['end_step'] or (
                    trial['stop_reason'] == 'physx_crashed'):
                raise ValueError('失败检查点与父轨迹实例、种子或起点不一致')
            policy = initial['policy']
            if policy['seed'] != seed or not policy['policy_weights']:
                raise ValueError('父轨迹缺少匹配的策略身份')
            target = parent / f'{decision:04d}'
            anchor_decision = read(target, 'decision.json')
            session = anchor_decision['session']
            if session['episode_ids'] != [branch] or session['env_ids'] != [0]:
                raise ValueError('父轨迹的策略会话与失败检查点不一致')
            for index in range(decision + 1):
                folder = parent / f'{index:04d}'
                row = anchor_decision if index == decision else read(folder, 'decision.json')
                if (row['decision'], row['start_step'], row['instance'], row['seed'], row['session']) != (
                        index, index * 8, instance, seed, session):
                    raise ValueError('父轨迹的决策前缀不连续或跨会话')
                if index < decision:
                    if (row['end_step'], row['executed_steps']) != ((index + 1) * 8, 8):
                        raise ValueError('父轨迹的决策前缀缺少完整的动作块')
                    for filename in ('observation.npz', 'executed.npz'):
                        archive_file(folder, filename)
            if (anchor_decision['language'] != source['language'] or
                    archive_file(target, 'observation.npz').read_bytes() != (checkpoint / 'observation.npz').read_bytes()):
                raise ValueError('失败检查点与父轨迹锚点观测或指令不一致')

            baseline = initial['truth']
            evidence = source['evidence']
            approach = evidence.get('capture_stage') == 'approach'
            if approach and run.parent != baseline_runs:
                raise ValueError('首次门前检查点必须来自 Experiment 11')
            if (baseline['step'] != 0 or baseline['Open'] is not False or baseline['ToggledOn'] is not False or
                    evidence['final_open'] is not False or evidence['final_toggled_on'] is not False or
                    source['memory_replay']['online_snapshot_saved'] is not True):
                raise ValueError('失败检查点缺少闭门、未启动的在线历史')
            contacts, peak = [], 0.0
            with archive_file(parent, 'truth.jsonl').open() as stream:
                for index in range(1, step + 1):
                    truth = json.loads(next(stream, 'null'))
                    if not truth or truth['step'] != index or truth['Open'] is not False or truth['ToggledOn'] is not False:
                        raise ValueError('父轨迹物理前缀不连续或已开门、已启动')
                    interaction = truth['interaction']
                    if any(contact['body'] == interaction['door_body'] for arm in interaction['contacts'].values()
                           for contact in arm):
                        contacts.append(index)
                        peak = max(peak, abs(truth['hinge'] - baseline['hinge']))
            if approach:
                distance = task40_approach_distance(truth)
                if contacts or not math.isfinite(distance) or distance > 0.12 or abs(truth['hinge']-baseline['hinge']) > .005:
                    raise ValueError('首次门前检查点必须在接触门前且手指距离门体不超过 12 cm')
            elif len(contacts) < 3 or contacts != evidence['door_contact_steps'] or not math.isfinite(peak) or peak < 0.005:
                raise ValueError('父轨迹缺少与检查点一致的真实闭门拉动证据')
            saved_joints = read(checkpoint, 'physical.json')['joints']
            microwave, = truth['microwaves']
            if (truth['decision'] != decision - 1 or truth != anchor_decision['state_before'] or
                    saved_joints[microwave] != truth['microwaves'][microwave]['joint_positions'] or
                    saved_joints['robot_r1'] != truth['robots']['robot_r1']['joint_positions']):
                raise ValueError('失败检查点物理状态与父轨迹锚点不一致')
        except (KeyError, TypeError, AttributeError) as error:
            raise ValueError('失败检查点或父轨迹记录格式不完整') from error
        return {'source': 'directed_restore', 'prefix': str(parent), 'instance': instance, 'source_seed': seed,
                'origin_id': f'instance={instance}/seed={seed}', 'parent_branch': branch,
                'start_step': step, 'start_decision': decision,
                'anchor_stage': 'approach' if approach else 'after_attempt',
                'policy_weights': policy['policy_weights']}

    def _task40_forward(self, request):
        if set(request) - {'task40_forward', 'batch', 'policy_host', 'policy_port', 'experiment'}:
            raise ValueError('不支持的 Task40 冻结前向参数')
        group = request.get('experiment', TASK40_DEFAULT_EXPERIMENT)
        experiment = self.task40_root(group)
        batch = self._task40_path(request.get('batch'), '前向配置')
        if not batch.is_relative_to(experiment.resolve()) or batch.suffix != '.json' or not batch.is_file():
            raise ValueError('前向配置必须是所选实验下的 JSON 文件')
        host = request.get('policy_host', '172.28.62.17')
        port = request.get('policy_port')
        if not isinstance(host, str) or (host != '172.28.62.17' and group != TASK40_OPENVLA_EXPERIMENT
                ) or type(port) is not int or not 1024 <= port <= 65535:
            raise ValueError('请选择已就绪的 Task40 前向诊断端口')
        identity = self.policy_resolver(host, port) if self.policy_resolver else None
        if not identity or not identity.get('task40_trace'):
            raise ValueError('所选推理端口未启用 Task40 记忆前向诊断')
        output = experiment / 'runs' / uuid.uuid4().hex[:12]
        plan = {'kind': 'task40_forward', 'task_id': 40, 'group': group,
                'version': identity.get('version'), 'instances': {}, 'output': str(output),
                'batch': str(batch), 'policy_host': host, 'policy_port': port, 'policy_identity': identity}
        argv = ['bash', str(self.root / 'scripts/b1k_python.sh'), '-u', str(self.task40_forward_worker),
                '--root', str(self.root), '--output', str(output), '--policy-host', host,
                '--port', str(port), '--batch', str(batch), '--experiment-plan', json.dumps(plan, ensure_ascii=False)]
        return {'argv': argv, 'command': shlex.join(argv), 'plan': plan}

    @staticmethod
    def _policy_port(plan):
        if plan.get('policy_port') is not None:
            return plan['policy_port']
        server = plan.get('policy_server') or ''
        match = re.fullmatch(r'172\.28\.62\.17:(\d+)', server)
        return int(match[1]) if match else None

    def start(self, request):
        prepared = self.prepare(request)
        plan = prepared["plan"]
        # Legacy plans use instances={gpu: [instance]}; V1.2 also carries a
        # flat gpus list because it has one isolated service per GPU.
        gpus = set(plan.get("instances", {}))
        if not gpus:
            gpus = {str(gpu) for gpu in plan.get("gpus", [plan.get("gpu_index", 0)])}
        with self.lock:
            collect = plan.get('kind') == 'task40_collect'
            experiment = plan.get('kind') in TASK40_EXPERIMENT_KINDS
            active = [job for job in self.jobs if job['process'].poll() is None]
            same_gpu = [job for job in active if gpus & job['gpus']]
            port = self._policy_port(plan)
            if port is not None and any(
                    (experiment or job['plan'].get('kind') in TASK40_EXPERIMENT_KINDS)
                    and self._policy_port(job['plan']) == port for job in active):
                raise ValueError('该诊断推理端口已绑定另一作业')
            if same_gpu:
                if not (collect and plan['sim_slots'] == 2 and len(same_gpu) == 1
                        and same_gpu[0]['plan'].get('kind') == 'task40_collect'):
                    raise ValueError('该 GPU 已有工作台评估，请选择另一张 GPU 或等待结束')
                first_plan = same_gpu[0]['plan']
                first_output = Path(first_plan['output'])
                first_decision = (first_plan['lineage']['start_decision']
                                  if first_plan.get('phase') == 'directed_restore' else 0)
                if not any(first_output.glob(f'metrics/*/{first_decision:04d}/decision.json')):
                    raise ValueError('首个 worker 需完成第一轮决策后才能增加同卡 worker')
            job_id = uuid.uuid4().hex[:12]
            self.log_root.mkdir(parents=True, exist_ok=True)
            log_root = self.task40_root(plan['group']) / 'logs' if experiment else self.log_root
            log_root.mkdir(parents=True, exist_ok=True)
            log_path = log_root / f"{job_id}.log"
            if experiment:
                prepared['plan']['job_id'] = job_id
                prepared['plan']['started_at'] = time.strftime("%Y-%m-%d %H:%M:%S")
                prepared['argv'][-1] = json.dumps(prepared['plan'], ensure_ascii=False)
                prepared['command'] = shlex.join(prepared['argv'])
            elif plan.get('kind') == 'b1k2026_websocket':
                plan['job_id'] = job_id
                plan['started_at'] = time.strftime("%Y-%m-%d %H:%M:%S")
                (Path(plan['output']) / 'job.json').write_text(
                    json.dumps(plan, ensure_ascii=False, indent=2) + '\n')
            with log_path.open("ab") as log:
                process = subprocess.Popen(prepared["argv"], cwd=self.root, stdin=subprocess.DEVNULL,
                                           stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
            job = {"id": job_id, "process": process, "gpus": gpus, "log_path": log_path,
                   "command": prepared["command"], "plan": prepared["plan"],
                   "started_at": time.strftime("%Y-%m-%d %H:%M:%S"),
                   "process_start": self._process_start(process.pid)}
            self.jobs.append(job)
            return self._status(job)

    @staticmethod
    def _process_start(pid):
        try:
            fields = (Path('/proc') / str(pid) / 'stat').read_text().rsplit(')', 1)[1].split()
            return fields[19] if fields[0] != 'Z' else None
        except (OSError, IndexError):
            return None

    def stop_restore(self, request):
        if not isinstance(request, dict) or set(request) != {'job'} or not isinstance(request['job'], str):
            raise ValueError('请选择一个 Task40 恢复任务')
        with self.lock:
            job = next((item for item in self.jobs if item['id'] == request['job']), None)
            if job is None or job['plan'].get('kind') not in ('task40_restore', *TASK40_EXPERIMENT_KINDS):
                raise ValueError('找不到可停止的 Task40 恢复任务')
            pid = job['process'].pid
            expected = job.get('process_start')
            if self._process_start(pid) != expected or expected is None:
                raise ValueError('恢复进程已结束或发生变化，请刷新任务状态')
            arguments = (Path('/proc') / str(pid) / 'cmdline').read_bytes().decode(errors='replace')
            worker = self.task40_forward_worker if job['plan'].get('kind') == 'task40_forward' else self.task40_worker
            if str(worker) not in arguments:
                raise ValueError('进程身份不匹配，未发送停止信号')
            descriptor = os.pidfd_open(pid)
            try:
                signal.pidfd_send_signal(descriptor, signal.SIGTERM)
            finally:
                os.close(descriptor)
            return {'state': 'stopping', 'job': request['job'], 'message': '正在停止并清理 Task40 策略与仿真进程'}

    def _status(self, job):
        code = job["process"].poll()
        with job["log_path"].open("rb") as log:
            header = log.read(65536).decode(errors="replace")
            log.seek(max(0, job["log_path"].stat().st_size - 12000))
            tail = log.read().decode(errors="replace")
        plan = job["plan"]
        try:
            actual, _ = json.JSONDecoder().raw_decode(header.lstrip())
            if "output" in actual:
                plan = actual  # Actual CLI allocation wins over dry-run prediction.
        except ValueError:
            pass
        output = Path(plan["output"])
        evaluation = self.evaluation
        status = {"id": job["id"], "pid": job["process"].pid,
                "state": "running" if code is None else "complete" if code == 0 else "failed",
                "returncode": code, "started_at": job["started_at"], "output": str(output),
                "folder": output.relative_to(evaluation).as_posix() if output.is_relative_to(evaluation) else None,
                "result_ready": output.is_relative_to(evaluation) and (output / "results.json").is_file(),
                "command": job["command"], "log": str(job["log_path"]),
                "log_tail": "\n".join(tail.splitlines()[-20:]),
                "task_id": plan.get("task_id"), "gpus": sorted(job["gpus"]),
                "version": plan.get("version"), "group": plan.get("group"),
                "kind": plan.get("kind"), "seed": plan.get("seed"),
                "policy_identity": plan.get("policy_identity"), "policy_port": self._policy_port(plan),
                "process_start": job.get('process_start'), "batch": plan.get('batch')}
        if plan.get('kind') == 'b1k2026_websocket':
            status['artifacts'] = v12_artifacts(output)
            native_json = sorted(output.glob('json/*.json'))
            status['result_ready'] = bool(native_json)
            status['native_results'] = [str(path.relative_to(output)) for path in native_json]
            status['videos'] = [str(path.relative_to(output)) for path in sorted(output.glob('videos/*')) if path.is_file()]
        if plan.get('kind') in TASK40_EXPERIMENT_KINDS:
            if plan.get('kind') == 'task40_collect' and 'phase' not in plan:
                # Restored workbench records contain a reduced plan.
                try:
                    archived = json.loads((output / 'job.json').read_text())
                except (OSError, ValueError):
                    archived = {}
            else:
                archived = plan
            if archived.get('phase') in ('directed_restore', 'renderer_probe'):
                status.update(phase=archived['phase'], lineage=archived.get('lineage'))
                if 'renderer_gi_history' in archived:
                    status['renderer_gi_history'] = archived['renderer_gi_history']
                if archived.get('renderer_disable_indirect_diffuse'):
                    status['renderer_disable_indirect_diffuse'] = True
            elif archived.get('phase') in ('controlled_reclosure', 'full_task_eval', 'approach_capture'):
                status['phase'] = archived['phase']
            stage_file = output / 'metrics' / 'active_stage.json'
            try:
                progress = json.loads(stage_file.read_text())
            except (OSError, ValueError):
                progress = {'stage': 'initializing'} if code is None else {}
            log_file = 'forward.log' if plan['kind'] == 'task40_forward' else 'simulator.log'
            with_log = output / 'logs' / log_file
            sim_tail = ''
            if with_log.is_file():
                with with_log.open('rb') as log:
                    log.seek(max(0, with_log.stat().st_size - 8000))
                    sim_tail = '\n'.join(log.read().decode(errors='replace').splitlines()[-15:])
            status.update(result_ready=(output / 'results.json').is_file(), artifacts=task40_artifacts(output),
                          progress=progress)
            status['forward_log_tail' if plan['kind'] == 'task40_forward' else 'sim_log_tail'] = sim_tail
        if code is not None and status['result_ready']:
            try:
                result = json.loads((output/'results.json').read_text())
                if result.get('status') in ('cancelled', 'paused', 'interrupted'):
                    status['state'] = result['status']
            except (OSError, ValueError):
                pass
        return status

    def snapshot(self, *, running_only=False):
        with self.lock:
            return [self._status(job) for job in reversed(self.jobs)
                    if not running_only or job['process'].poll() is None]
