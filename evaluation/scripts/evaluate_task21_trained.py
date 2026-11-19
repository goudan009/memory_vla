"""Wait for paired training, serve the selected weights, then queue official B1K runs."""

import argparse
import json
import os
from pathlib import Path
import shlex
import subprocess
import sys
import time

import msgpack
import zmq


def remote(host, command):
    return subprocess.check_output(["ssh", "-o", "BatchMode=yes", "-o", "ConnectTimeout=10",
        host, command], text=True)


def write(path, data):
    path.write_text(json.dumps(data, indent=2, ensure_ascii=False)+"\n")


def policy_request(host, port, endpoint, timeout_ms=2000):
    context = zmq.Context()
    socket = context.socket(zmq.REQ)
    socket.setsockopt(zmq.RCVTIMEO, timeout_ms)
    socket.setsockopt(zmq.SNDTIMEO, timeout_ms)
    socket.setsockopt(zmq.LINGER, 0)
    socket.connect(f"tcp://{host}:{port}")
    try:
        socket.send(msgpack.packb({"endpoint": endpoint}))
        return socket.recv() != b"ERROR"
    except zmq.Again:
        return False
    finally:
        socket.close()
        context.term()


def run(args):
    args.output.mkdir(parents=True, exist_ok=True)
    project = Path(__file__).resolve().parents[2]
    write(args.output / "status.json", {"status": "waiting_training", "remote_run": args.remote_run})
    while True:
        # Keep lightweight logs/curves/metrics available in the local output tree;
        # full model and optimizer files remain on the A100 data volume.
        subprocess.run(["rsync", "-az", "--exclude=checkpoints/", "--exclude=*.pt",
            "--exclude=*.safetensors", "--exclude=*.partial*",
            args.host+":"+args.remote_run+"/", str(args.output.parent)+"/"], check=True)
        command = shlex.join(["test", "-f", args.remote_run+"/status.json"])
        command += " && " + shlex.join(["cat", args.remote_run+"/status.json"])
        response = subprocess.run(["ssh", "-o", "BatchMode=yes", args.host, command],
                                  text=True, capture_output=True)
        state = json.loads(response.stdout) if response.returncode == 0 else {}
        if state.get("status") == "failed":
            raise RuntimeError(f"Training failed; closed-loop not launched: {state}")
        if state.get("status") == "completed":
            break
        time.sleep(30)
    selected = state["selected_checkpoints"]
    records = []
    for arm, gpu, port, label in (("baseline", 2, 5562, "groot"), ("memory_v1", 1, 5561, "grootmeory")):
        checkpoint = selected[arm]["path"]
        log = args.remote_run+f"/logs/policy_{arm}.log"
        command = [args.remote_python, "-u", "-m", "v1_groot_memoryvla.serve", "--model-path", checkpoint,
                   "--embodiment-tag", "BEHAVIOR_R1_PRO", "--port", str(port),
                   "--use-sim-policy-wrapper", "--seed", "42"]
        code = ("import os,subprocess,json; "
            f"env=dict(os.environ,CUDA_VISIBLE_DEVICES={str(gpu)!r},PYTHONPATH='src',PYTHONNOUSERSITE='1',"
            "PYTHONDONTWRITEBYTECODE='1',NO_ALBUMENTATIONS_UPDATE='1',HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1'); "
            f"p=subprocess.Popen({command!r},cwd={args.remote_source!r},env=env,"
            f"stdout=open({log!r},'a'),stderr=subprocess.STDOUT,stdin=subprocess.DEVNULL,start_new_session=True); "
            "print(json.dumps({'pid':p.pid}))")
        process = json.loads(remote(args.host, shlex.join([args.remote_python, "-c", code])))
        record = {"arm": arm, "checkpoint": selected[arm], "policy_port": port,
                  "remote_gpu": gpu, "policy_pid": process["pid"], "policy_log": log}
        records.append(record)
        write(args.output / "selected_policies.json", records)
        ready = False
        for _ in range(150):
            if policy_request(args.host, port, "ping"):
                ready = True
                break
            time.sleep(2)
        if not ready:
            raise RuntimeError(f"Selected policy did not start: {record}")
        invocation = ["bash", str(project / "evaluation/scripts/run_groot_eval.sh"),
            "--mode", "single1", "--preset", "official-eval", "--gpu-index", "1",
            "--task", "21", "--episodes-per-gpu", "10", "--instances-gpu1", *map(str, range(10)),
            "--seed", "0", "--seed-mode", "sequence", "--policy-host", args.host,
            "--policy-port", str(port), "--model-name", label, "--version", "experiment_1",
            "--run-tag", Path(args.remote_run).name+"_"+arm, "--wait-for-gpu", "--gpu-wait-interval", "30"]
        record["invocation"] = invocation
        write(args.output / "status.json", {"status": "closed_loop_queued_or_running", **record})
        with (args.output / f"{arm}.log").open("a") as stream:
            environment = dict(os.environ, PROJECT_ROOT=str(project),
                GROOT_REPO="/nvme3/projects/GR00T/Isaac-GR00T-n1d6-9b37aa1")
            completed = subprocess.run(invocation, cwd=project, env=environment,
                                       stdout=stream, stderr=subprocess.STDOUT)
        record["evaluation_returncode"] = completed.returncode
        # Only stop the dedicated policy started above, never an existing service.
        policy_request(args.host, port, "kill")
        write(args.output / "selected_policies.json", records)
        if completed.returncode:
            raise RuntimeError(f"Closed-loop evaluation failed for {arm}; see its log")
    write(args.output / "status.json", {"status": "completed", "policies": records,
            "evaluation_results": "evaluation/experiment_1; tag identifies this paired training run"})


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--host", default="172.28.62.17")
    p.add_argument("--remote-run", required=True)
    p.add_argument("--remote-source", required=True)
    p.add_argument("--remote-python", default="/nvme3/projects/memory_vla/envs/n1d6-p0/bin/python")
    p.add_argument("--output", type=Path, required=True)
    a = p.parse_args()
    try:
        run(a)
    except Exception as error:
        a.output.mkdir(parents=True, exist_ok=True)
        write(a.output / "status.json", {"status": "failed", "error": repr(error)})
        raise
