"""Launch single-arm or paired B1K DDP training from an explicit Episode split."""

import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import time

from .launch_task21_formal import plot
from .task21_formal import atomic_json
from .memory_training_options import add_memory_training_options, memory_training_argv, TRAINING_PROTOCOL
from v3_bodymemory.data.episode_split import validate_split


def gpu_list(text):
    result = [int(v) for v in text.split(",")]
    if not result or len(result) != len(set(result)) or min(result) < 0:
        raise argparse.ArgumentTypeError("Supply unique GPU indices, e.g. 2,3")
    return result


def busy_gpus(allowed_pids=()):
    indices = subprocess.check_output(["nvidia-smi", "--query-gpu=index,uuid",
                                      "--format=csv,noheader,nounits"], text=True)
    owners = {u.strip(): int(i) for i, u in (line.split(",") for line in indices.splitlines())}
    apps = subprocess.check_output(["nvidia-smi", "--query-compute-apps=gpu_uuid,pid",
                                   "--format=csv,noheader,nounits"], text=True)
    return {owners[u.strip()] for line in apps.splitlines() if line.strip()
            for u, pid in [line.split(",")] if int(pid.strip()) not in allowed_pids}


def command(args, arm, gpus):
    result = [sys.executable, "-u", "-m", "torch.distributed.run", "--standalone",
        "--nnodes=1", f"--nproc-per-node={len(gpus)}", "--module",
        "v3_bodymemory.experiment.task21_ddp", "--dataset", str(args.dataset),
        "--checkpoint", str(args.checkpoint), "--output", str(args.output), "--arm", arm,
        "--steps", str(args.steps), "--global-windows-per-step", str(args.global_windows_per_step),
        "--microbatch", str(args.microbatch), "--seed", str(args.seed),
        "--learning-rate", str(args.learning_rate), "--eval-interval",
        str(getattr(args, "validation_interval", 1000)) if args.async_validation_gpu is not None else "500", "--checkpoint-interval", "1000"]
    result += memory_training_argv(args)
    if args.async_validation_gpu is not None:
        result.append("--async-validation")
    if getattr(args, "no_validation", False):
        result.append("--no-validation")
    if args.train_only:
        result.append("--train-only")
    if args.paired:
        result.append("--paired")
    if getattr(args, "resume_step", 0):
        result += ["--resume-step", str(args.resume_step)]
    return result


def launch(args):
    groups = {args.arm: args.gpus}
    if args.paired:
        assert args.arm == "memory_v1" and args.baseline_gpus
        assert not set(args.gpus) & set(args.baseline_gpus)
        groups["baseline"] = args.baseline_gpus
    assert all(args.global_windows_per_step % len(gpus) == 0 for gpus in groups.values())
    split = json.loads(args.split.read_text())
    validate_split(split)
    if getattr(args, "no_validation", False):
        assert not args.paired and args.async_validation_gpu is None
        assert not split['validation'], 'No-validation runs must use an explicit training-only split'
    else:
        assert split['validation'], 'Enable validation only with independent validation Episodes'
    plan = {"mode": "paired" if args.paired else "single_arm", "output": str(args.output),
        "global_windows_per_step": args.global_windows_per_step,
        "global_decisions_per_step": 32*args.global_windows_per_step,
        "steps_each": args.steps, "validation_interval": None if getattr(args, "no_validation", False) else
            getattr(args, "validation_interval", 1000) if args.async_validation_gpu is not None else 500,
        "checkpoint_interval": 1000, "async_validation_gpu": args.async_validation_gpu,
        "validation_enabled": not getattr(args, "no_validation", False),
        "baseline_validation_gpu": args.baseline_validation_gpu,
        "training_protocol": TRAINING_PROTOCOL,
        "memory_conditioning": getattr(args, "memory_conditioning", "direct"),
        "memory_length": getattr(args, "memory_length", 64),
        "image_augmentation": getattr(args, "image_augmentation", True),
        "validation_parallel_with_training": args.async_validation_gpu is not None,
        "commands": {arm: {"CUDA_VISIBLE_DEVICES": ",".join(map(str, gpus)),
                           "argv": command(args, arm, gpus)} for arm, gpus in groups.items()}}
    print(json.dumps(plan, indent=2), flush=True)
    if args.dry_run:
        return
    selected = {gpu for group in groups.values() for gpu in group}
    if args.async_validation_gpu is not None:
        selected.add(args.async_validation_gpu)
        if args.baseline_validation_gpu is not None:
            selected.add(args.baseline_validation_gpu)
    if args.adopt_running_memory:
        from .task21_async import take_over_async
        return take_over_async(args, groups, plan)
    assert not selected & busy_gpus(args.allow_service_pids), "Selected GPUs already have compute processes; no jobs were stopped"
    if args.resume_step:
        assert args.async_validation_gpu is not None, "Resume requires asynchronous validation"
        from .task21_ddp import resume_manifest
        for arm, gpus in groups.items():
            resume_manifest(argparse.Namespace(**dict(vars(args), arm=arm)), args.output / arm, len(gpus))
    else:
        assert not (args.output / "status.json").exists(), "Use a fresh output directory"
    for directory in ("config", "logs", "metrics", "plots"):
        (args.output / directory).mkdir(parents=True, exist_ok=True)
    if args.resume_step:
        assert split == json.loads((args.output / "config/data_split.json").read_text())
        plan["resume_step"] = args.resume_step
        atomic_json(args.output / f"config/ddp_resume_step-{args.resume_step:06d}.json", plan)
    else:
        atomic_json(args.output / "config/data_split.json", split)
        atomic_json(args.output / "config/ddp_launch.json", plan)
    if args.async_validation_gpu is not None:
        from .task21_async import launch_async
        return launch_async(args, groups)
    children, streams = {}, []
    try:
        for arm, gpus in groups.items():
            env = dict(os.environ, CUDA_VISIBLE_DEVICES=",".join(map(str, gpus)), PYTHONNOUSERSITE="1",
                PYTHONDONTWRITEBYTECODE="1", NO_ALBUMENTATIONS_UPDATE="1", OMP_NUM_THREADS="4",
                HF_HUB_OFFLINE="1", TRANSFORMERS_OFFLINE="1")
            stream = (args.output / f"logs/{arm}_torchrun.log").open("a")
            streams.append(stream)
            children[arm] = subprocess.Popen(command(args, arm, gpus), env=env,
                stdin=subprocess.DEVNULL, stdout=stream, stderr=subprocess.STDOUT)
        atomic_json(args.output / "status.json", {"status": "training", "mode": "ddp",
            "pid": os.getpid(), "torchrun": {arm: p.pid for arm, p in children.items()}})
        if args.paired:
            for step in range(500, args.steps+1, 500):
                paths = {arm: args.output / arm / f"metrics/round-{step:06d}.json" for arm in groups}
                while not all(path.exists() for path in paths.values()):
                    for arm, child in children.items():
                        if child.poll() is not None:
                            raise RuntimeError(f"{arm} torchrun exited before validation: {child.returncode}")
                    time.sleep(5)
                results = {arm: json.loads(path.read_text()) for arm, path in paths.items()}
                stop = step == args.steps or (step >= 2000 and all(v["stale_rounds"] >= 4 for v in results.values()))
                atomic_json(args.output / f"metrics/decision-{step:06d}.json",
                    {"step": step, "stop": stop, "arms": results})
                plot(args.output)
                if stop:
                    break
        for arm, child in children.items():
            assert child.wait() == 0, f"{arm} torchrun failed"
        selected = {arm: json.loads((args.output / arm / "checkpoints/best.json").read_text()) for arm in groups} if not args.train_only and not getattr(args, "no_validation", False) else {}
        atomic_json(args.output / "selected_checkpoints.json", selected)
        atomic_json(args.output / "status.json", {"status": "completed", "selected_checkpoints": selected})
        if getattr(args, "no_validation", False):
            plot(args.output)
    except Exception as error:
        atomic_json(args.output / "status.json", {"status": "failed", "error": repr(error)})
        raise
    finally:
        for child in children.values():
            if child.poll() is None:
                child.terminate()
        for stream in streams:
            stream.close()


def main():
    p = argparse.ArgumentParser(description=__doc__)
    add_memory_training_options(p)
    for name in ("dataset", "checkpoint", "output", "split"):
        p.add_argument("--"+name, type=Path, required=True)
    p.add_argument("--gpus", type=gpu_list, required=True)
    p.add_argument("--arm", choices=("memory_v1", "baseline"), default="memory_v1")
    p.add_argument("--paired", action="store_true")
    p.add_argument("--baseline-gpus", type=gpu_list)
    p.add_argument("--global-windows-per-step", type=int, default=2)
    p.add_argument("--steps", type=int, default=10000)
    p.add_argument("--microbatch", type=int, default=4)
    p.add_argument("--learning-rate", type=float, default=1e-5)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--dry-run", action="store_true")
    p.add_argument("--disable-early-stopping", action="store_true",
                   help="Use the full matched optimizer budget regardless of validation plateaus")
    p.add_argument("--validation-interval", type=int, default=1000,
                   help="Async validation cadence; checkpoints still save every 1000 steps")
    p.add_argument("--async-validation-gpu", type=int, help="Independent validation GPU (may share a training GPU); no validation barriers")
    p.add_argument("--baseline-validation-gpu", type=int, help="Optionally put the baseline validator on a different GPU")
    p.add_argument("--allow-service-pids", nargs="*", type=int, default=[], help="Explicit existing services permitted to share selected GPUs")
    p.add_argument("--adopt-running-memory", action="store_true", help="Keep this run's M1 training and validator; restart only baseline on the requested GPUs")
    p.add_argument("--train-only", action="store_true")
    p.add_argument("--no-validation", action="store_true", help="Train and save every 1000 steps without a validator; distinct from --train-only smoke checks")
    p.add_argument("--resume-step", type=int, default=0)
    args = p.parse_args()
    assert (args.train_only and args.steps > 0 and not args.paired) or (args.steps >= 1000 and args.steps % 1000 == 0)
    assert args.global_windows_per_step >= 1
    assert args.memory_length >= 1
    assert args.validation_interval >= 1000 and args.validation_interval % 1000 == 0
    launch(args)


if __name__ == "__main__":
    main()
