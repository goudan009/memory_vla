"""Read-only Linux process/socket snapshots for the evaluation dashboard.

The same standalone stdlib collector runs over SSH on the policy machine.
Only selected environment fields are returned, never full process environments.
"""

import csv
import json
from pathlib import Path
import re
import socket
import subprocess
import threading
import time


def argument(argv, *names, default=None):
    for index, value in enumerate(argv):
        for name in names:
            if value == name and index + 1 < len(argv):
                return argv[index + 1]
            if value.startswith(name + "="):
                return value.split("=", 1)[1]
    return default


def classify(argv):
    if not argv or not Path(argv[0]).name.startswith("python"):
        return None
    module = argument(argv, "-m", default="")
    script = next((value for value in argv[1:] if value.endswith(".py")), "")
    code = argument(argv, "-c", default="")
    if (argument(argv, "--env_name", "--env-name") and
            (module in ("evaluation.rollout_policy", "gr00t.eval.rollout_policy") or
             script.endswith("rollout_policy.py") or "gr00t/eval/rollout_policy.py" in code)):
        return "evaluation"
    if (module in ("gr00t.eval.run_gr00t_server", "v1_groot_memoryvla.serve", "baselines.groot_n1d6.serve") or
            script.endswith("run_gr00t_server.py") or
            (script.endswith("serve.py") and ("groot" in script.lower() or "memoryvla" in script))):
        return "inference"
    if module == 'torch.distributed.run' and 'v1_groot_memoryvla.experiment.task21_ddp' in argv:
        return 'training'  # Count the torchrun controller once, not every rank.
    return None


def listening_ports(output):
    ports = {}
    for line in output.splitlines():
        fields = line.split()
        if len(fields) < 5:
            continue
        address = fields[3]  # ss -H -ltnp: state recv-q send-q local peer process
        try:
            port = int(address.rsplit(":", 1)[1])
        except (ValueError, IndexError):
            continue
        for pid in re.findall(r"pid=(\d+)", line):
            ports.setdefault(int(pid), set()).add(port)
    return ports


def collect_snapshot(proc_root=Path("/proc")):
    processes = []
    for path in proc_root.iterdir():
        if not path.name.isdigit():
            continue
        try:
            status = (path / "status").read_text()
            if re.search(r"^State:\s+Z", status, re.M):
                continue
            argv = (path / "cmdline").read_bytes().decode(errors="replace").rstrip("\0").split("\0")
            kind = classify(argv)
            if kind is None:
                continue
            try:
                environment = dict(item.split("=", 1) for item in
                                   (path / "environ").read_bytes().decode(errors="replace").split("\0") if "=" in item)
            except PermissionError:
                environment = {}
            row = {"pid": int(path.name), "kind": kind,
                   "gpu": environment.get("OMNIGIBSON_GPU_ID") if kind == "evaluation" else
                          environment.get("CUDA_VISIBLE_DEVICES", argument(argv, "--device"))}
            if kind == "evaluation":
                row.update(env_name=argument(argv, "--env_name", "--env-name"),
                           n_envs=int(argument(argv, "--n_envs", "--n-envs", default="1")),
                           policy_host=argument(argv, "--policy_client_host", "--policy-client-host"),
                           policy_port=argument(argv, "--policy_client_port", "--policy-client-port"))
                output = environment.get("GROOT_EVAL_OUTPUT_DIR")
                if output:
                    try:
                        metadata = json.loads((Path(output) / "run_meta.json").read_text())
                    except (OSError, ValueError):
                        metadata = {}
                    row.update(task_id=metadata.get("task_id"),
                               run_directory=metadata.get("public_output_dir") or output)
            elif kind == 'training':
                output=argument(argv,'--output',default='')
                row.update(run_directory=output, label=Path(output).name,
                           n_workers=argument(argv,'--nproc-per-node',default='1'))
            else:
                row.update(model=argument(argv, "--model-path", "--model_path"),
                           configured_port=argument(argv, "--port", default="5555"))
            processes.append(row)
        except (FileNotFoundError, ProcessLookupError, PermissionError):
            continue
    result = {"hostname": socket.gethostname(), "checked_at": time.strftime("%H:%M:%S"),
              "ok": True, "processes": sorted(processes, key=lambda row: row["pid"])}
    try:
        response = subprocess.run([
            'nvidia-smi', '--query-gpu=index,name,memory.total,memory.used,utilization.gpu',
            '--format=csv,noheader,nounits'], capture_output=True, text=True, timeout=2, check=True)
        result['gpus'] = [dict(index=int(index),name=name.strip(),total_mb=int(total),
                               used_mb=int(used),utilization=int(utilization))
                          for index,name,total,used,utilization in csv.reader(response.stdout.splitlines(),skipinitialspace=True)]
    except (OSError, subprocess.SubprocessError, ValueError) as error:
        result['gpus_error'] = str(error)
    try:
        response = subprocess.run(["ss", "-H", "-ltnp"], capture_output=True, text=True, timeout=2, check=True)
        ports = listening_ports(response.stdout)
        for row in processes:
            row["listening_ports"] = sorted(ports.get(row["pid"], ()))
    except (OSError, subprocess.SubprocessError) as error:
        result["ports_error"] = str(error)
        for row in processes:
            row["listening_ports"] = None
    return result


class ServiceMonitor:
    def __init__(self, policy_host="172.28.62.17", interval=5, additional_hosts=None):
        self.policy_host = policy_host
        self.interval = interval
        self._lock = threading.Lock()
        self._stop = threading.Event()
        self._value = {"local": {"ok": None}, "remote": {"ok": None, "host": policy_host}}
        self.additional_hosts = additional_hosts or {}
        self._value.update({key:{'ok':None,'host':host} for key,host in self.additional_hosts.items()})

    def snapshot(self):
        with self._lock:
            return self._value

    def poll(self):
        started = time.monotonic()
        try:
            local = collect_snapshot()
        except OSError as error:
            local = {"ok": False, "error": str(error), "checked_at": time.strftime("%H:%M:%S")}
        # Publish the local result even if SSH is slow/unavailable.
        with self._lock:
            self._value = {**self._value, "local": local}
        self.poll_remote('remote', self.policy_host)
        return max(0, self.interval - (time.monotonic() - started))

    def poll_remote(self, key, host):
        started=time.monotonic()
        with self._lock:
            previous=self._value.get(key,{})
        try:
            response = subprocess.run(
                ["ssh", "-o", "BatchMode=yes", "-o", "ConnectTimeout=3", "-o", "ConnectionAttempts=1",
                 host, "python3 - --snapshot"],
                input=Path(__file__).read_text(), capture_output=True, text=True, timeout=6, check=True,
            )
            remote = json.loads(response.stdout)
            remote["host"] = host
            remote['last_seen']=time.strftime('%Y-%m-%d %H:%M:%S')
        except (OSError, subprocess.SubprocessError, ValueError):
            remote = {"ok": False, "host": host, "error": "设备离线或 SSH 暂时不可达",
                      "checked_at": time.strftime("%H:%M:%S"), 'last_seen':previous.get('last_seen')}
        with self._lock:
            self._value = {**self._value, key: remote}
        return max(0, self.interval - (time.monotonic() - started))

    def start(self):
        def run():
            while not self._stop.is_set():
                self._stop.wait(self.poll())
        self._thread = threading.Thread(target=run, name="eval-service-monitor", daemon=True)
        self._thread.start()
        self._extra_threads=[]
        for key,host in self.additional_hosts.items():
            def run_extra(key=key,host=host):
                while not self._stop.is_set():
                    self._stop.wait(self.poll_remote(key,host))
            thread=threading.Thread(target=run_extra,name='device-monitor-'+key,daemon=True)
            self._extra_threads.append(thread)
            thread.start()

    def close(self):
        self._stop.set()
        self._thread.join(timeout=7)
        for thread in self._extra_threads:
            thread.join(timeout=7)


if __name__ == "__main__":
    print(json.dumps(collect_snapshot(), ensure_ascii=False))
