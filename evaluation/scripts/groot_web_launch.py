"""Small web adapter for the canonical evaluation CLI; no simulator imports."""

import json
import os
from pathlib import Path
import re
import shlex
import subprocess
import threading
import time
import uuid

from groot_tasks import TASK_NAMES
from groot_eval_control import EvaluationControl


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

    def pause(self, request):
        with self.lock:
            return self.control.pause(request)

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
                "versions": versions, "evaluation_root": str(evaluation)}

    def argv(self, request):
        if not isinstance(request, dict):
            raise ValueError("启动参数必须是 JSON 对象")
        scalars = {"version", "group", "model_name", "mode", "gpu_index", "task", "preset",
                   "seed", "seed_mode", "policy_host", "policy_port", "run_tag",
                   "episodes_per_gpu", "debug_max_steps", "video_steps_per_render", "policy_checkpoint", "training_id"}
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
            if any(key not in {"resume_from", "policy_host", "policy_port"} for key in request):
                raise ValueError("续跑沿用原配置，只允许显式覆盖推理地址和端口")
        for key in sorted(scalars):
            value = request.get(key)
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

    def prepare(self, request):
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
                group=request.get('group')
                if not request.get('resume_from') and group in ('baseline','memory','official') and identity.get('method') and group!=identity['method']:
                    raise ValueError('结果方法 '+group+' 与推理服务实际方法 '+identity['method']+' 不一致，请选择对应分组')
                if not request.get('resume_from'):
                    if not group and identity.get('method'):
                        argv += ['--group',identity['method']]
                    if not request.get('version') and re.fullmatch(r'V_[0-9]+',identity.get('version') or ''):
                        argv += ['--version',identity['version']]
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

    def start(self, request):
        prepared = self.prepare(request)
        gpus = set(prepared["plan"]["instances"])
        with self.lock:
            for job in self.jobs:
                if job["process"].poll() is None and gpus & job["gpus"]:
                    raise ValueError("该 GPU 已有网页启动的评估，请选择另一张 GPU 或等待结束")
            job_id = uuid.uuid4().hex[:12]
            self.log_root.mkdir(parents=True, exist_ok=True)
            log_path = self.log_root / f"{job_id}.log"
            with log_path.open("ab") as log:
                process = subprocess.Popen(prepared["argv"], cwd=self.root, stdin=subprocess.DEVNULL,
                                           stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
            job = {"id": job_id, "process": process, "gpus": gpus, "log_path": log_path,
                   "command": prepared["command"], "plan": prepared["plan"],
                   "started_at": time.strftime("%Y-%m-%d %H:%M:%S")}
            self.jobs.append(job)
            return self._status(job)

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
        return {"id": job["id"], "pid": job["process"].pid,
                "state": "running" if code is None else "complete" if code == 0 else "failed",
                "returncode": code, "started_at": job["started_at"], "output": str(output),
                "folder": output.relative_to(evaluation).as_posix() if output.is_relative_to(evaluation) else None,
                "result_ready": (output / "results.json").is_file(),
                "command": job["command"], "log": str(job["log_path"]),
                "log_tail": "\n".join(tail.splitlines()[-20:]),
                "task_id": plan.get("task_id"), "gpus": sorted(job["gpus"]),
                "version": plan.get("version"), "group": plan.get("group")}

    def snapshot(self):
        with self.lock:
            return [self._status(job) for job in reversed(self.jobs)]
