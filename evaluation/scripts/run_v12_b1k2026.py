#!/usr/bin/env python3
"""Run one official GR00T N1.7/B1K-2026 evaluation under workbench control."""

import argparse
import json
import os
from pathlib import Path
import shutil
import signal
import subprocess
import sys
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


def _prepare_isolated_isaac_apps(behavior_root, eval_env):
    """Stage OmniGibson's app files outside the read-only Isaac installation."""
    # Do not inherit the legacy 2025/DeepMap Isaac path from the workbench
    # process. V1.2 is pinned to the installed Isaac Sim 5.1 tree.
    isaac_path = Path("/opt/isaac-sim-5.1").resolve()
    if not (isaac_path / "VERSION").is_file():
        raise FileNotFoundError(f"pinned Isaac Sim installation is unavailable: {isaac_path}")
    eval_env["ISAAC_PATH"] = str(isaac_path)
    app_root = Path(__file__).resolve().parents[1] / "V_1_2" / "runtime" / "isaac"
    apps_path = app_root / "apps"
    app_root.mkdir(parents=True, exist_ok=True)
    apps_path.mkdir(parents=True, exist_ok=True)
    for name in ("exts", "extscache", "extsPhysics", "extsUser", "extsDeprecated"):
        target = app_root / name
        source = isaac_path / name
        if target.is_symlink():
            if target.resolve() != source.resolve():
                raise RuntimeError(f"unexpected isolated Isaac extension link: {target}")
        elif target.exists():
            if source.exists() or not target.is_dir() or any(target.iterdir()):
                raise RuntimeError(f"isolated Isaac extension path is not a symlink: {target}")
        elif source.exists():
            target.symlink_to(source, target_is_directory=True)

    source_kit = behavior_root / "OmniGibson" / "omnigibson" / "omnigibson_5_1_0.kit"
    source_icon = behavior_root / "docs" / "assets" / "OmniGibson_logo.png"
    if not source_kit.is_file() or not source_icon.is_file():
        raise FileNotFoundError("pinned OmniGibson app kit or icon is missing")
    shutil.copy2(source_kit, apps_path / source_kit.name)
    shutil.copy2(source_icon, apps_path / source_icon.name)
    eval_env["EXP_PATH"] = str(apps_path)

    # Isaac Sim's bundled python.sh sources this setup file before importing
    # isaacsim. Reuse the official environment setup while keeping the B1K
    # evaluator interpreter and its Python packages.
    setup = isaac_path / "setup_python_env.sh"
    if not setup.is_file():
        raise FileNotFoundError(f"Isaac Sim Python setup is unavailable: {setup}")
    command = ["/bin/bash", "-c", "source \"$1\"; env -0", "v12-isaac-env", str(setup)]
    result = subprocess.run(command, env=eval_env, stdout=subprocess.PIPE, check=True)
    for item in result.stdout.split(b"\0"):
        if not item or b"=" not in item:
            continue
        key, value = item.split(b"=", 1)
        eval_env[key.decode()] = value.decode()
    eval_env["CARB_APP_PATH"] = str(isaac_path / "kit")
    eval_env["ISAAC_PATH"] = str(isaac_path)
    eval_env["EXP_PATH"] = str(apps_path)
    eval_env["LD_PRELOAD"] = str(isaac_path / "kit" / "libcarb.so")


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
    env.setdefault("HF_HUB_CACHE", "/nvme3/models/.hf-cache/hub")
    env.setdefault("GROOT_HF_LOCAL_FIRST", "1")
    env.setdefault("GROOT_PATCH_MISTRAL", "1")
    behavior_root = Path(evaluate["cwd"]).resolve()
    # Keep the official N1.7 server's upstream environment separate from the
    # Python 3.11 B1K-2026 evaluator compatibility paths.
    eval_env = env.copy()
    # B1K-2026 pins warp-lang==1.12.0. Put the isolated evaluator's site
    # packages first so Isaac Sim's bundled 1.8.x Warp extension cannot shadow
    # the required API (while Isaac's omni.* extensions remain available).
    isolated_site = str(Path(sys.prefix) / "lib" / "python3.11" / "site-packages")
    eval_paths = [
        isolated_site,
        str(behavior_root / "OmniGibson"),
        str(behavior_root),
        str(behavior_root / "bddl3"),
        str(behavior_root / "joylo"),
        "/nvme3/projects/memory_vla/evaluation/V_1_2/runtime_compat",
        "/nvme3/projects/GR00T/lerobot-b1k-py311/src",
        "/nvme3/envs/b1k2025-policy/lib/python3.11/site-packages",
        "/nvme3/projects/SERF-VLA/venvs/serf-official-clean/lib/python3.11/site-packages",
    ]
    if eval_env.get("PYTHONPATH"):
        eval_paths.append(eval_env["PYTHONPATH"])
    eval_env["PYTHONPATH"] = os.pathsep.join(eval_paths)
    # The workbench service may itself inherit the legacy 2025 simulator
    # environment.  V1.2 must never resolve assets through that process-wide
    # value: bind the evaluator to the pinned B1K-2026 data root explicitly.
    eval_env["OMNIGIBSON_DATA_PATH"] = "/nvme3/datasets/b1k2026-assets/data"
    # Before SimulationApp starts, post2 has no lazy omni.log binding. Use the
    # upstream stdout/stderr buffering context until the app is initialized.
    eval_env["OMNIGIBSON_NO_OMNI_LOGS"] = "True"
    _prepare_isolated_isaac_apps(behavior_root, eval_env)
    evaluator_argv = evaluate["argv"]
    if evaluator_argv[1:3] != ["-m", "omnigibson.eval.eval"]:
        raise ValueError("unexpected V1.2 evaluator module")
    evaluator_argv = [
        evaluator_argv[0], str(Path(__file__).with_name("v12_b1k2026_evaluator.py")),
        *evaluator_argv[3:],
    ]
    server_log = (output / "server.log").open("ab")
    evaluator_log = (output / "evaluator.log").open("ab")
    server = None
    try:
        server_argv = list(serve['argv'])
        language = serve.get('language', {})
        if language.get('mode') == 'task_description':
            script = str(Path(serve['cwd']) / 'scripts/b1k/serve_b1k.py')
            if server_argv[1:3] != ['-u', script]:
                raise ValueError('unexpected V1.2 official server script')
            server_argv = [server_argv[0], '-u', str(Path(__file__).with_name('v12_b1k2026_server.py')),
                           '--official-script', script, '--text-prompt', language['text_prompt'],
                           *server_argv[3:]]
        server = subprocess.Popen(
            server_argv, cwd=serve["cwd"], env=env, stdin=subprocess.DEVNULL,
            stdout=server_log, stderr=subprocess.STDOUT, start_new_session=True,
        )
        _health(serve)
        _write(output / "runner.json", {
            "version": serve.get("version"), "protocol": serve.get("protocol"),
            "task": evaluate.get("task_name"), "checkpoint_sha256": serve.get("checkpoint_sha256"),
            "source_revision": serve.get("source", {}).get("revision"),
            "instance_indices": evaluate.get("instance_indices"), "num_rollouts": evaluate.get("num_rollouts"),
            "state": "ready", "server_pid": server.pid, "evaluator_argv": evaluator_argv,
            "server_argv": server_argv, "language": language,
        })
        completed = subprocess.run(
            evaluator_argv, cwd=evaluate["cwd"], env=eval_env, stdin=subprocess.DEVNULL,
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
            "evaluator_argv": evaluator_argv,
            "server_argv": server_argv, "language": language,
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
