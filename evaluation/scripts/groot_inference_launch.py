"""Start the existing A100 policy launcher over the dashboard's SSH connection."""

import json
from pathlib import Path
import shlex
import subprocess
import threading
import time
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[2]/'train'))
from run_identity import identity_bundle


PROJECT = "/nvme3/projects/memory_vla"
DEFAULT_SOURCE = PROJECT + "/deployments/20261109_2100_task21_ddp_async_s42_resume_r4"
DEFAULT_PYTHON = PROJECT + "/envs/n1d6-p0/bin/python"

REMOTE = identity_bundle() + Path(__file__).with_name('groot_inference_remote.py').read_text()
CATALOG = REMOTE + "\ndispatch('catalog', None)\n"
STATUS = REMOTE + "\ndispatch('snapshot', JOBS)\n"


class InferenceManager:
    def __init__(self, project_root, host="172.28.62.17", device=None):
        self.host = host
        self.device=device or {}
        paths=self.device.get('paths',{})
        self.project=Path(paths.get('project', PROJECT))
        self.default_source=paths.get('project',PROJECT)+'/deployments/web_live' if paths else DEFAULT_SOURCE
        self.default_python=paths.get('python',DEFAULT_PYTHON)
        self.bind_host=host.split('@')[-1]
        self.remote='DEVICE_PATHS='+repr(paths)+'\n'+REMOTE
        self.script = Path(project_root) / "evaluation/scripts/run_groot_server.sh"
        self.jobs = []
        self.lock = threading.Lock()

    def _remote(self, command, code, timeout=12):
        result = subprocess.run(
            ["ssh", "-o", "BatchMode=yes", "-o", "ConnectTimeout=4", self.host, command],
            input=code, capture_output=True, text=True, timeout=timeout)
        if result.returncode:
            raise ValueError((result.stderr or result.stdout or "所选设备 SSH 连接失败").strip().splitlines()[-1])
        try:
            return json.loads(result.stdout)
        except ValueError as error:
            raise ValueError("所选设备未返回有效启动信息") from error

    def options(self):
        result = self._remote("python3 -", self.remote+"\ndispatch('catalog', None)\n")
        return dict(result, host=self.bind_host, source_dir=self.default_source, python_path=self.default_python)

    def argv(self, request):
        if not isinstance(request, dict) or set(request) - {"checkpoint", "gpu_index", "port", "seed", "source_dir", "python_path", "history_mode", "task40_diagnostics", "experiment", "inference_steps", "action_diagnostics"}:
            raise ValueError("不支持的推理启动参数")
        history_mode=request.get('history_mode', 'normal')
        if history_mode not in ('normal', 'empty', 'other_episode'):
            raise ValueError('历史对照须为normal、empty或other_episode')
        argv = []
        if 'inference_steps' in request:
            if type(request['inference_steps']) is not int or request['inference_steps'] not in (4, 8, 16):
                raise ValueError('inference_steps 必须是4、8或16')
            argv += ['--inference-steps', str(request['inference_steps'])]
        if 'action_diagnostics' in request:
            if type(request['action_diagnostics']) is not bool:
                raise ValueError('action_diagnostics 必须是开关')
            if request['action_diagnostics']:
                argv += ['--action-diagnostics']
        for key, default in (("checkpoint", None), ("gpu_index", None), ("port", None),
                             ("seed", 42), ("source_dir", self.default_source), ("python_path", self.default_python)):
            value = request.get(key, default)
            if key in ("gpu_index", "port", "seed"):
                if type(value) is not int:
                    raise ValueError(f"{key} 必须是整数")
                if (key == "gpu_index" and value < 0) or (key == "port" and not 1 <= value <= 65535) or (key == "seed" and not 0 <= value < 2**32):
                    raise ValueError(f"{key} 超出允许范围")
            elif not isinstance(value, str) or not value.startswith("/") or len(value) > 1024 or any(c in value for c in "\r\n\x00"):
                raise ValueError(f"{key} 请填写所选设备上的绝对路径")
            argv += ["--" + key.replace("_", "-"), str(value)]
        if history_mode != 'normal':
            argv += ['--history-mode', history_mode]
        if 'task40_diagnostics' in request:
            if type(request['task40_diagnostics']) is not bool:
                raise ValueError('task40_diagnostics 必须是开关')
            if request['task40_diagnostics']:
                argv += ['--task40-diagnostics']
        if 'experiment' in request:
            if request['experiment'] not in ('experiment_10', 'experiment_11', 'experiment_12') or request.get('task40_diagnostics') is not True:
                raise ValueError('所选 experiment 需要启用 Task40 诊断服务')
            argv += ['--log-dir', str(self.project / 'Output/test_01' / request['experiment'] / 'policy_services')]
        return argv + ["--host", self.bind_host, "--json"]

    def _run(self, request, check=False):
        argv = self.argv(request) + (["--check-only"] if check else [])
        # Send the same canonical script over stdin, so local and web startup
        # never depend on an outdated copy on the remote machine.
        return self._remote(shlex.join(["bash", "-s", "--", *argv]), self.script.read_text())

    def prepare(self, request):
        return self._run(request, check=True)

    def start(self, request):
        with self.lock:
            job = dict(self._run(request), host=self.bind_host, started_at=time.strftime("%Y-%m-%d %H:%M:%S"))
            self.jobs.append(job)
            return job

    def snapshot(self):
        with self.lock:
            code = "JOBS = " + repr(self.jobs) + "\n" + self.remote+"\ndispatch('snapshot', JOBS)\n"
            result = self._remote("python3 -", code)
            self.jobs = [dict(job, host=self.bind_host) for job in result["jobs"]]
            return {"jobs": list(reversed(self.jobs))}

    def stop(self, request):
        if not isinstance(request, dict) or set(request) != {"pid", "port", "checkpoint", "start_ticks"}:
            raise ValueError("请从服务列表选择要关闭的推理进程")
        if any(type(request[key]) is not int or request[key] <= 0 for key in ("pid", "port", "start_ticks")) or request["port"] > 65535:
            raise ValueError("服务进程或端口参数不正确")
        if not isinstance(request["checkpoint"], str) or not request["checkpoint"].startswith("/") or any(c in request["checkpoint"] for c in "\r\n\x00"):
            raise ValueError("服务权重路径不正确")
        with self.lock:
            code = self.remote + "\ndispatch('stop', " + repr(request) + ")\n"
            job = dict(self._remote("python3 -", code), host=self.bind_host)
            self.jobs = [j for j in self.jobs if not (j["pid"] == job["pid"] and j.get("start_ticks") == job.get("start_ticks"))]
            self.jobs.append(job)
            return job

    def resolve(self, host, port):
        """Read the actual remote process identity; never infer it from a label."""
        if host != self.bind_host:
            return None
        jobs=self.snapshot()['jobs']
        matches=[job for job in jobs if job['port']==int(port) and job['state']=='ready']
        if len(matches)!=1:
            raise ValueError('所选推理端口尚未就绪，请刷新并选择已就绪服务')
        keys=('version','method','training_id','train_number','checkpoint','checkpoint_step','source_dir',
              'source_digest','training_seed','policy_seed','history_mode','task_id','pid','start_ticks','host','port','naming_schema')
        keys += ('task40_diagnostics', 'task40_trace', 'gpu_index')
        result={key:matches[0].get(key) for key in keys}
        result['checkpoint']=matches[0].get('checkpoint_resolved') or matches[0]['checkpoint']
        return dict(result, binding='observed_process')
