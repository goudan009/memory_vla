#!/usr/bin/env python3
"""Run one official GR00T N1.7/B1K-2026 evaluation under workbench control."""

import argparse
import json
import os
from pathlib import Path
import signal
import subprocess
import time
from urllib.error import URLError
from urllib.request import Request, urlopen


def _json(value):
    return json.loads(value)


def _write(path, value):
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n")


def _health(spec, timeout=180):
    url = f"http://{spec['host']}:{spec['port']}{spec.get('health_path', '/healthz')}"
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        try:
            with urlopen(Request(url, method="GET"), timeout=3) as response:
                if response.status == 200:
                    return
        except (OSError, URLError):
            pass
        time.sleep(1)
    raise RuntimeError(f"official policy server health check timed out: {url}")


def _terminate(process):
    if process is None or process.poll() is not None:
        return
    try:
        os.killpg(process.pid, signal.SIGTERM)
        process.wait(timeout=20)
    except (OSError, subprocess.TimeoutExpired):
        try:
            os.killpg(process.pid, signal.SIGKILL)
        except OSError:
            pass
        process.wait(timeout=10)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--serve-spec", required=True, type=_json)
    parser.add_argument("--eval-spec", required=True, type=_json)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    serve = args.serve_spec
    evaluate = args.eval_spec
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    if Path(evaluate["output"]).resolve() != output:
        raise ValueError("evaluator output does not match workbench allocation")
    if serve.get("protocol") != "b1k2026_websocket" or evaluate.get("protocol") != "b1k2026_websocket":
        raise ValueError("V1.2 runner only accepts the native B1K-2026 protocol")

    (output / "server.log").touch()
    (output / "evaluator.log").touch()
    _write(output / "runner.json", {
        "version": serve.get("version"),
        "protocol": serve.get("protocol"),
        "task": evaluate.get("task_name"),
        "checkpoint_sha256": serve.get("checkpoint_sha256"),
        "source_revision": serve.get("source", {}).get("revision"),
        "instance_indices": evaluate.get("instance_indices"),
        "num_rollouts": evaluate.get("num_rollouts"),
        "state": "starting",
    })

    env = os.environ.copy()
    env.update(serve.get("env", {}))
    behavior_root = Path(evaluate["cwd"]).resolve()
    # Keep the official N1.7 server's upstream environment separate from the
    # Python 3.11 B1K-2026 evaluator compatibility paths.
    eval_env = env.copy()
    eval_paths = [
        str(behavior_root / "OmniGibson"),
        str(behavior_root),
        str(behavior_root / "bddl3"),
        str(behavior_root / "joylo"),
        "/opt/isaac-sim-5.1/extscache/omni.warp.core-1.8.2+lx64",
        "/nvme3/projects/memory_vla/evaluation/V_1_2/runtime_compat",
        "/nvme3/projects/GR00T/lerobot-b1k-py311/src",
        "/nvme3/envs/b1k2025-policy/lib/python3.11/site-packages",
        "/nvme3/projects/SERF-VLA/venvs/serf-official-clean/lib/python3.11/site-packages",
    ]
    if eval_env.get("PYTHONPATH"):
        eval_paths.append(eval_env["PYTHONPATH"])
    eval_env["PYTHONPATH"] = os.pathsep.join(eval_paths)
    eval_env["OMNIGIBSON_DATA_PATH"] = os.environ.get(
        "OMNIGIBSON_DATA_PATH", "/nvme3/datasets/b1k2026-assets/data"
    )
    server_log = (output / "server.log").open("ab")
    evaluator_log = (output / "evaluator.log").open("ab")
    server = None
    try:
        server = subprocess.Popen(
            serve["argv"], cwd=serve["cwd"], env=env, stdin=subprocess.DEVNULL,
            stdout=server_log, stderr=subprocess.STDOUT, start_new_session=True,
        )
        _health(serve)
        _write(output / "runner.json", {
            "version": serve.get("version"), "protocol": serve.get("protocol"),
            "task": evaluate.get("task_name"), "checkpoint_sha256": serve.get("checkpoint_sha256"),
            "source_revision": serve.get("source", {}).get("revision"),
            "instance_indices": evaluate.get("instance_indices"), "num_rollouts": evaluate.get("num_rollouts"),
            "state": "ready", "server_pid": server.pid,
        })
        completed = subprocess.run(
            evaluate["argv"], cwd=evaluate["cwd"], env=eval_env, stdin=subprocess.DEVNULL,
            stdout=evaluator_log, stderr=subprocess.STDOUT, check=False,
        )
        native = sorted((output / "json").glob("*.json"))
        if completed.returncode == 0 and not native:
            raise RuntimeError("official evaluator exited without native JSON rollout output")
        _write(output / "runner.json", {
            "version": serve.get("version"), "protocol": serve.get("protocol"),
            "task": evaluate.get("task_name"), "checkpoint_sha256": serve.get("checkpoint_sha256"),
            "source_revision": serve.get("source", {}).get("revision"),
            "instance_indices": evaluate.get("instance_indices"), "num_rollouts": evaluate.get("num_rollouts"),
            "state": "complete" if completed.returncode == 0 else "failed",
            "evaluator_returncode": completed.returncode,
            "native_json": [str(path.relative_to(output)) for path in native],
            "native_video": [str(path.relative_to(output)) for path in sorted((output / "videos").glob("*")) if path.is_file()],
        })
        return completed.returncode
    finally:
        _terminate(server)
        server_log.close()
        evaluator_log.close()


if __name__ == "__main__":
    raise SystemExit(main())
