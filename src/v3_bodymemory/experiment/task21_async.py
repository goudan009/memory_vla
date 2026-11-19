"""Independent checkpoint validation for single-arm or paired B1K training."""

import argparse
import gc
import json
import os
from pathlib import Path
from .memory_training_options import add_memory_training_options, memory_training_argv
import signal
import subprocess
import sys
import time

import torch

from .task21_formal import atomic_json, initialize, validate


def validate_checkpoints(args):
    root = args.output / args.arm
    for name in ("metrics", "diagnostics/actions", "checkpoints", "logs"):
        (root / name).mkdir(parents=True, exist_ok=True)
    processed, best, stale = set(), float("inf"), 0
    for path in sorted((root / "metrics").glob("round-*.json")):
        result = json.loads(path.read_text())
        processed.add(result["step"])
        best, stale = result["best_loss"], result["stale_rounds"]
    print(json.dumps({"kind": "validator_ready", "arm": args.arm,
        "processed_checkpoints": sorted(processed), "pid": os.getpid()}), flush=True)
    while True:
        pending = sorted((root / "validation_queue").glob("step-*.json"))
        for path in pending:
            entry = json.loads(path.read_text())
            step = entry["step"]
            if step in processed:
                continue
            atomic_json(root / "validation_status.json", {"status": "validating", "checkpoint_step": step,
                "pid": os.getpid(), "training_continues": True})
            processor, train, validation, model = initialize(args, Path(entry["checkpoint"]))
            with torch.no_grad():
                result = validate(model, processor, train, validation, args, root, step)
            if result["validation_loss"] < best:
                atomic_json(root / "checkpoints/best.json", {"step": step, "path": entry["checkpoint"],
                    "validation_loss": result["validation_loss"]})
            improved = result["validation_loss"] < best*(1-.001)
            stale = 0 if improved else stale+1
            best = min(best, result["validation_loss"])
            result.update(stale_rounds=stale, best_loss=best, checkpoint=entry["checkpoint"],
                          asynchronous=True)
            atomic_json(root / f"metrics/round-{step:06d}.json", result)
            processed.add(step)
            del model, processor, train, validation
            gc.collect()
            torch.cuda.empty_cache()
        status_path = root / "status.json"
        if status_path.exists():
            status = json.loads(status_path.read_text())
            if status.get("training_complete"):
                final_step = status["step"]
                queued = {json.loads(p.read_text())["step"] for p in (root / "validation_queue").glob("step-*.json")}
                if final_step in processed and queued <= processed:
                    atomic_json(root / "validation_status.json", {"status": "completed",
                        "validated_steps": sorted(processed), "best_loss": best})
                    return
        time.sleep(3)


def shared_stop_target(current_steps, interval):
    """Leave room for a rank already entering its next optimizer update."""
    return (max(current_steps)//interval+2)*interval


class ExistingProcess:
    """Observe an already running child through its completion record."""

    def __init__(self, pid, status_path):
        self.pid, self.status_path = pid, status_path

    def poll(self):
        try:
            state = Path(f"/proc/{self.pid}/stat").read_text().split()[2]
            if state != "Z":
                return None
        except FileNotFoundError:
            pass
        status = json.loads(self.status_path.read_text()) if self.status_path.exists() else {}
        return 0 if status.get("status") == "completed" else 1


def take_over_async(args, groups, plan):
    """Replace only the baseline while existing M1 ranks keep their exact state."""
    from .launch_task21_ddp import busy_gpus
    root = args.output
    previous = json.loads((root / "status.json").read_text())
    assert previous["mode"] == "ddp_async" and previous["status"] == "training"
    pids = previous["processes"]
    expected = {previous["pid"]: "launch_task21_ddp",
        pids["train_memory_v1"]: "torch.distributed.run",
        pids["train_baseline"]: "torch.distributed.run",
        pids["validate_memory_v1"]: "task21_async",
        pids["validate_baseline"]: "task21_async"}
    for pid, token in expected.items():
        cmd = Path(f"/proc/{pid}/cmdline").read_bytes().replace(b"\0", b" ").decode()
        assert token in cmd and str(root) in cmd, (pid, cmd)
    assert not (root / "baseline/checkpoints/latest.json").exists(), "Switch before the first baseline checkpoint"
    assert not (root / "metrics/stop_training.json").exists()
    # GPU0 may only contain the two explicitly allowed original services.
    prior_plan = json.loads((root / "config/ddp_launch.json").read_text())
    extra = set(groups["baseline"]) - {int(v) for v in
        prior_plan["commands"]["baseline"]["CUDA_VISIBLE_DEVICES"].split(",")}
    assert not extra & busy_gpus(args.allow_service_pids), "Additional GPU has another job"
    os.kill(previous["pid"], signal.SIGTERM)
    os.kill(pids["validate_baseline"], signal.SIGTERM)
    os.kill(pids["train_baseline"], signal.SIGTERM)
    old = ExistingProcess(pids["train_baseline"], root / "baseline/status.json")
    while old.poll() is None:
        time.sleep(1)
    subprocess.run(["gio", "trash", str(root / "baseline")], check=True)
    atomic_json(root / "config/transition_2plus2.json", {"previous": previous,
        "new_baseline_gpus": groups["baseline"], "baseline_restarts_from_official_step": 0,
        "memory_training_continues": True, "original_service_pids_unchanged": args.allow_service_pids,
        "new_source": str(Path(__file__).resolve().parents[3]), "pid": os.getpid()})
    plan["commands"]["memory_v1"] = prior_plan["commands"]["memory_v1"]
    plan["memory_process_adopted"] = True
    atomic_json(root / "config/ddp_launch.json", plan)
    adopted = {"train_memory_v1": ExistingProcess(pids["train_memory_v1"], root / "memory_v1/status.json"),
        "validate_memory_v1": ExistingProcess(pids["validate_memory_v1"], root / "memory_v1/validation_status.json")}
    return launch_async(args, groups, adopted)


def launch_async(args, groups, adopted=None):
    from .launch_task21_ddp import command
    from .launch_task21_formal import plot

    children, streams, train = dict(adopted or {}), [], {}
    root = args.output
    try:
        for arm, gpus in groups.items():
            env = dict(os.environ, CUDA_VISIBLE_DEVICES=",".join(map(str, gpus)),
                PYTHONNOUSERSITE="1", PYTHONDONTWRITEBYTECODE="1", OMP_NUM_THREADS="4",
                HF_HUB_OFFLINE="1", TRANSFORMERS_OFFLINE="1", NO_ALBUMENTATIONS_UPDATE="1")
            if "train_"+arm in children:
                train[arm] = children["train_"+arm]
            else:
                # Root logs from a restarted baseline belong with its discarded run.
                if adopted and arm == "baseline":
                    for old_name in ("baseline_torchrun.log", "baseline_validation.log"):
                        old_log = root / "logs" / old_name
                        if old_log.exists():
                            subprocess.run(["gio", "trash", str(old_log)], check=True)
                stream = (root / f"logs/{arm}_torchrun.log").open("a")
                streams.append(stream)
                if args.resume_step:
                    # A completed run's validator must wait for the new budget.
                    atomic_json(root / arm / "status.json", {"status": "resuming",
                        "step": args.resume_step, "total_steps": args.steps,
                        "training_complete": False})
                    atomic_json(root / arm / "validation_status.json", {"status": "waiting_for_checkpoint",
                        "resume_step": args.resume_step, "total_steps": args.steps})
                train[arm] = subprocess.Popen(command(args, arm, gpus), env=env, start_new_session=True,
                    stdin=subprocess.DEVNULL, stdout=stream, stderr=subprocess.STDOUT)
                children["train_"+arm] = train[arm]
            if "validate_"+arm in children:
                continue
            argv = [sys.executable, "-u", "-m", __name__, "--output", str(root),
                "--dataset", str(args.dataset), "--checkpoint", str(args.checkpoint), "--arm", arm,
                "--seed", str(args.seed), "--microbatch", str(args.microbatch)]
            argv += memory_training_argv(args)
            stream = (root / f"logs/{arm}_validation.log").open("a")
            streams.append(stream)
            validation_gpu = args.baseline_validation_gpu if arm == "baseline" and args.baseline_validation_gpu is not None else args.async_validation_gpu
            env = dict(env, CUDA_VISIBLE_DEVICES=str(validation_gpu))
            children["validate_"+arm] = subprocess.Popen(argv, env=env, start_new_session=True,
                stdin=subprocess.DEVNULL, stdout=stream, stderr=subprocess.STDOUT)
        rounds, stop_target = set(), None
        while True:
            for name, child in children.items():
                if child.poll() not in (None, 0):
                    raise RuntimeError(f"{name} exited with {child.returncode}; see logs")
            states = {arm: json.loads((root / arm / "status.json").read_text())
                      for arm in groups if (root / arm / "status.json").exists()}
            validation_states = {arm: json.loads((root / arm / "validation_status.json").read_text())
                      for arm in groups if (root / arm / "validation_status.json").exists()}
            for step in range(1000, args.steps+1, 1000):
                if step in rounds:
                    continue
                paths = {arm: root / arm / f"metrics/round-{step:06d}.json" for arm in groups}
                if not all(p.exists() for p in paths.values()):
                    continue
                results = {arm: json.loads(path.read_text()) for arm, path in paths.items()}
                stop = (not getattr(args, "disable_early_stopping", False) and step >= 2000
                        and all(v["stale_rounds"] >= 4 for v in results.values()))
                if stop and stop_target is None:
                    # Both arms keep exactly the same update count despite validation lag.
                    stop_target = min(args.steps, shared_stop_target(
                        [s.get("step", 0) for s in states.values()], 1000))
                    atomic_json(root / "metrics/stop_training.json", {"stop_after_step": stop_target,
                        "based_on_validation_step": step, "reason": "both arms stale for four validation rounds"})
                atomic_json(root / f"metrics/decision-{step:06d}.json",
                    {"step": step, "stop": stop, "stop_after_step": stop_target, "arms": results})
                rounds.add(step)
                plot(root)
            atomic_json(root / "status.json", {"status": "training" if any(p.poll() is None for p in train.values())
                else "waiting_for_async_validation", "mode": "ddp_async", "pid": os.getpid(),
                "max_steps": args.steps, "train": states, "validation": validation_states,
                "processes": {name: p.pid for name, p in children.items()}, "stop_after_step": stop_target})
            if all(child.poll() == 0 for child in children.values()):
                break
            time.sleep(5)
        steps = [json.loads((root / arm / "status.json").read_text())["step"] for arm in groups]
        assert len(set(steps)) == 1, "Paired training budgets differ"
        selected = {arm: json.loads((root / arm / "checkpoints/best.json").read_text()) for arm in groups}
        atomic_json(root / "selected_checkpoints.json", selected)
        plot(root)
        atomic_json(root / "status.json", {"status": "completed", "steps_each": steps[0],
            "selected_checkpoints": selected})
    except Exception as error:
        atomic_json(root / "status.json", {"status": "failed", "error": repr(error)})
        raise
    finally:
        for child in children.values():
            if child.poll() is None:
                os.killpg(child.pid, signal.SIGTERM)
        for stream in streams:
            stream.close()


def main():
    p = argparse.ArgumentParser(description=__doc__)
    add_memory_training_options(p)
    for name in ("dataset", "checkpoint", "output"):
        p.add_argument("--"+name, type=Path, required=True)
    p.add_argument("--arm", choices=("baseline", "memory_v1"), required=True)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--microbatch", type=int, default=4)
    args = p.parse_args()
    from .output import capture_console
    with capture_console(args.output / args.arm / "logs/validation_detail.log"):
        validate_checkpoints(args)


if __name__ == "__main__":
    main()
