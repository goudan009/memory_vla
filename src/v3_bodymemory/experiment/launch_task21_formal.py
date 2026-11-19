"""Persistent paired training controller; one check, shared budgets and early stop."""

import argparse
import hashlib
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time

from .task21_formal import atomic_json


def takeover_at_checkpoint(args):
    """Let the original workers finish validation, then resume their exact state."""
    root = args.output
    previous = json.loads((root / "status.json").read_text())
    assert previous["pid"] == args.takeover_pid and previous["status"] == "training"
    expected = {args.takeover_pid: "experiment.launch_task21_formal",
                **{pid: "experiment.task21_formal" for pid in previous["workers"].values()}}
    for pid, module in expected.items():
        command = Path(f"/proc/{pid}/cmdline").read_bytes().replace(b"\0", b" ").decode()
        assert module in command and str(root) in command
    assert not (root / f"metrics/decision-{args.resume_step:06d}.json").exists()
    args.takeover_committed = False
    record = {"status": "waiting_for_checkpoint", "resume_step": args.resume_step,
              "checkpoint_interval": args.checkpoint_interval, "validation_interval": 500,
              "old_processes": expected, "pid": os.getpid(), "source": str(Path(__file__).resolve())}
    path = root / "config/checkpoint_interval_change.json"
    os.kill(args.takeover_pid, signal.SIGSTOP)
    try:
        atomic_json(path, record)
        rounds = [root / arm / f"metrics/round-{args.resume_step:06d}.json"
                  for arm in previous["workers"]]
        while not all(p.exists() for p in rounds):
            for pid in previous["workers"].values():
                assert Path(f"/proc/{pid}").exists(), f"Worker {pid} exited before checkpoint"
            time.sleep(5)
        assert not (root / f"metrics/decision-{args.resume_step:06d}.json").exists()
        for arm in previous["workers"]:
            assert (root / arm / "checkpoints/latest_training_state.pt").exists()
        for pid in previous["workers"].values():
            os.kill(pid, signal.SIGTERM)
        os.kill(args.takeover_pid, signal.SIGTERM)
        os.kill(args.takeover_pid, signal.SIGCONT)
        args.takeover_committed = True
        for pid in expected:
            while True:
                try:
                    state = Path(f"/proc/{pid}/stat").read_text().split()[2]
                except FileNotFoundError:
                    break
                if state == "Z":
                    break
                time.sleep(1)
        record.update(status="resuming", completed_checkpoint=args.resume_step)
        atomic_json(path, record)
    finally:
        if not args.takeover_committed:
            os.kill(args.takeover_pid, signal.SIGCONT)


def plot(root):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    for arm in ("baseline", "memory_v1"):
        path = root / arm / "metrics/metrics.jsonl"
        if not path.exists():
            continue
        rows = [json.loads(s) for s in path.read_text().splitlines()]
        tr, ev = ([r for r in rows if r["kind"] == k] for k in ("train", "eval"))
        # Aggregate training windows in blocks of 50; raw values remain in JSONL.
        xs, ys = [], []
        for i in range(0, len(tr), 50):
            block = tr[i:i+50]
            xs.append(block[-1]["step"])
            ys.append(sum(r["loss"] for r in block)/len(block))
        axes[0, 0].plot(xs, ys, label=arm+" train")
        axes[0, 0].plot([r["step"] for r in ev], [r["validation_loss"] for r in ev], "o-", label=arm+" val")
        for c in ("normal", "empty", "wrong"):
            axes[0, 1].plot([r["step"] for r in ev], [r["action_mae"][c] for r in ev], label=arm+" "+c)
        axes[1, 0].plot([r["step"] for r in tr], [r["decisions_per_second_end_to_end"] for r in tr], label=arm)
        axes[1, 1].plot([r["step"] for r in tr], [r["peak_allocated_gib"] for r in tr], label=arm)
    for ax, title in zip(axes.flat, ("Action loss", "Normalized action MAE (fixed panel)",
                                    "End-to-end decisions / second", "Peak allocated GiB")):
        ax.set_title(title)
        ax.set_xlabel("Optimizer steps")
        ax.legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(root / "plots/training_curves.png", dpi=160)
    plt.close(fig)
    tasks = sorted({r["task"] for r in tr if r.get("task") is not None})
    if len(tasks) > 1:
        fig, axes = plt.subplots(len(tasks), 2, figsize=(12, 3*len(tasks)), squeeze=False)
        for task, pair in zip(tasks, axes):
            task_rows = [r for r in tr if r.get("task") == task]
            blocks = [task_rows[i:i+50] for i in range(0, len(task_rows), 50)]
            pair[0].plot([b[-1]["step"] for b in blocks],
                         [sum(r["loss"] for r in b)/len(b) for b in blocks], label="train (50 task updates)")
            selected = [r for r in ev if str(task) in r.get("validation_by_task", {})]
            values = [r["validation_by_task"][str(task)] for r in selected]
            pair[0].plot([r["step"] for r in selected], [v["loss"] for v in values], "o-", label="independent val")
            pair[0].axhline(.0025, linestyle="--", color="gray", label="observation target")
            for condition in ("normal", "empty", "wrong"):
                pair[1].plot([r["step"] for r in selected],
                    [v["action_mae"][condition] for v in values], "o-", label=condition)
            for axis, title in zip(pair, (f"Task {task} action loss", f"Task {task} action MAE (fixed panel)")):
                axis.set_title(title); axis.set_xlabel("Optimizer steps (all tasks)"); axis.legend(fontsize=7)
        fig.tight_layout()
        fig.savefig(root / "plots/per_task_curves.png", dpi=160)
        plt.close(fig)


def launch(args):
    root = args.output
    for name in ("config", "logs", "metrics", "plots", "diagnostics"):
        (root / name).mkdir(parents=True, exist_ok=True)
    assert args.resume_step or not (root / "status.json").exists(), "Use a fresh run directory"
    if args.takeover_pid:
        takeover_at_checkpoint(args)
    config = dict(vars(args))
    config.update(seed=42, action_horizon=32, execution_horizon=8, gradient_window=32,
        eval_interval=500, checkpoint_interval=args.checkpoint_interval,
        memory_capacity=16, perceptual_dim=256, retrieval_layers=2, fusion="gate",
        consolidate="tome", update_fused=True, frozen_eagle=True,
        cross_window_memory="independent Episode keys, detached fused writes",
        optimizer="AdamW", weight_decay=.01, max_gradient_norm=1., lr_schedule="constant",
        validation_scope="all complete windows in all 20 independent Episodes",
        action_mae_scope="one fixed middle 32-decision window per validation Episode, full preceding context",
        wrong_history="other training Episode, relative progress matched, exact post-ToMe timestamps matched",
        early_stop="both arms stale for 4 rounds, relative threshold0.1%, minimum2000 steps; stop together",
        checkpoint_selection="each arm minimum normal-history independent validation Episode mean loss")
    atomic_json(root / "config/run_config.json", {k: str(v) if isinstance(v, Path) else v for k, v in config.items()})
    package = Path(__file__).resolve().parents[1]
    hashes = {str(p.relative_to(package)): hashlib.sha256(p.read_bytes()).hexdigest()
              for p in package.rglob("*.py") if ".venv" not in p.parts}
    hash_name = f"source_sha256_resume_{args.resume_step}.json" if args.resume_step else "source_sha256.json"
    atomic_json(root / "config" / hash_name, hashes)
    atomic_json(root / "status.json", {"status": "resuming" if args.resume_step else "long_sequence_check",
                                      "step": args.resume_step, "pid": os.getpid()})
    common = ["--dataset", str(args.dataset), "--checkpoint", str(args.checkpoint),
              "--output", str(root), "--steps", str(args.steps), "--eval-interval", "500",
              "--microbatch", str(args.microbatch), "--seed", "42", "--learning-rate", str(args.learning_rate),
              "--checkpoint-interval", str(args.checkpoint_interval), "--resume-step", str(args.resume_step)]
    env = os.environ.copy()
    env.update(CUDA_VISIBLE_DEVICES=str(args.memory_gpu), PYTHONNOUSERSITE="1",
               PYTHONDONTWRITEBYTECODE="1", NO_ALBUMENTATIONS_UPDATE="1", OMP_NUM_THREADS="4")
    command = [sys.executable, "-u", "-m", "v3_bodymemory.experiment.task21_formal", *common]
    if not args.resume_step:
        with (root / "logs/long_sequence_check.log").open("a") as log:
            subprocess.run([*command, "--check-only"], env=env, stdout=log, stderr=subprocess.STDOUT, check=True)
    assert json.loads((root / "diagnostics/long_sequence_check.json").read_text())["passed"]
    if args.resume_step:
        rounds = {arm: json.loads((root / arm / f"metrics/round-{args.resume_step:06d}.json").read_text())
                  for arm in ("memory_v1", "baseline")}
        atomic_json(root / f"metrics/decision-{args.resume_step:06d}.json",
                    {"step": args.resume_step, "stop": False, "reason": "resume_after_checkpoint_interval_change",
                     "arms": rounds})
        plot(root)
    children, logs = {}, []
    try:
        for arm, gpu in (("memory_v1", args.memory_gpu), ("baseline", args.baseline_gpu)):
            child_env = dict(env, CUDA_VISIBLE_DEVICES=str(gpu))
            log = (root / f"logs/{arm}.log").open("a")
            logs.append(log)
            children[arm] = subprocess.Popen([*command, "--arm", arm], env=child_env,
                                             stdout=log, stderr=subprocess.STDOUT)
        atomic_json(root / "status.json", {"status": "training", "pid": os.getpid(),
                    "workers": {a: p.pid for a, p in children.items()}})
        for step in range(args.resume_step+500, args.steps+1, 500):
            paths = {a: root / a / f"metrics/round-{step:06d}.json" for a in children}
            while not all(p.exists() for p in paths.values()):
                for arm, child in children.items():
                    if child.poll() is not None:
                        raise RuntimeError(f"{arm} exited before paired validation: {child.returncode}")
                time.sleep(5)
            result = {a: json.loads(p.read_text()) for a, p in paths.items()}
            stop = step == args.steps or (step >= 2000 and all(r["stale_rounds"] >= 4 for r in result.values()))
            decision = {"step": step, "stop": stop, "reason": "budget_limit" if step == args.steps else
                        "both_validation_curves_plateau" if stop else "continue", "arms": result}
            atomic_json(root / f"metrics/decision-{step:06d}.json", decision)
            plot(root)
            print(json.dumps(decision), flush=True)
            if stop:
                break
        for child in children.values():
            assert child.wait() == 0
        selected = {a: json.loads((root / a / "checkpoints/best.json").read_text()) for a in children}
        atomic_json(root / "selected_checkpoints.json", selected)
        atomic_json(root / "status.json", {"status": "completed", "optimizer_steps_each": step,
                    "selected_checkpoints": selected, "pid": os.getpid()})
    finally:
        for child in children.values():
            if child.poll() is None:
                child.terminate()
        for log in logs:
            log.close()


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for name in ("dataset", "checkpoint", "output"):
        p.add_argument("--"+name, type=Path, required=True)
    p.add_argument("--memory-gpu", type=int, default=1)
    p.add_argument("--baseline-gpu", type=int, default=2)
    p.add_argument("--steps", type=int, default=10000)
    p.add_argument("--checkpoint-interval", type=int, default=1000)
    p.add_argument("--resume-step", type=int, default=0)
    p.add_argument("--takeover-pid", type=int, default=0)
    p.add_argument("--microbatch", type=int, default=4)
    p.add_argument("--learning-rate", type=float, default=1e-5)
    a = p.parse_args()
    assert a.steps >= 500 and a.steps % 500 == 0
    assert a.checkpoint_interval > 0 and a.checkpoint_interval % 500 == 0
    assert 0 <= a.resume_step < a.steps and a.resume_step % 500 == 0
    assert not a.takeover_pid or a.resume_step
    assert a.memory_gpu != a.baseline_gpu and 0 not in (a.memory_gpu, a.baseline_gpu)
    try:
        launch(a)
    except Exception as error:
        if not a.takeover_pid or getattr(a, "takeover_committed", False):
            atomic_json(a.output / "status.json", {"status": "failed", "error": repr(error), "pid": os.getpid()})
        raise


if __name__ == "__main__":
    main()
