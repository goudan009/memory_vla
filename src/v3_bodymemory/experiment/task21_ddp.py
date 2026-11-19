"""Episode-affine DDP training; separate entry, never takes over an existing run."""

import argparse
import hashlib
from contextlib import nullcontext
from datetime import timedelta
import json
import os
from pathlib import Path
import time
import traceback

import numpy as np
import torch
import torch.distributed as dist
from torch import nn
from torch.nn.parallel import DistributedDataParallel

from .first_training import cache_sequence, gradient_norms, seeded, slice_features
from .output import capture_console
from .memory_training_options import add_memory_training_options, TRAINING_PROTOCOL
from .task21_diagnostic import append_record, head_hash
from .task21_formal import (atomic_json, initialize, memory_runtime,
                            restore_memory_runtime, save_model, validate, window_context, supervision_count)


def resume_manifest(args, root, world):
    manifest = json.loads((root / "checkpoints/latest.json").read_text())
    original = json.loads((root / "config/ddp.json").read_text())
    if hasattr(args, "training_protocol"):
        assert original.get("training_protocol") == TRAINING_PROTOCOL, (
            "Training sampling/augmentation changed: start a fresh run; resume older runs with their saved source snapshot")
        for key in ("memory_length", "memory_conditioning", "image_augmentation"):
            assert original.get(key) == getattr(args, key), key
    assert manifest["step"] == args.resume_step
    for key, value in (("world_size", world), ("seed", args.seed),
                       ("global_windows_per_step", args.global_windows_per_step)):
        assert manifest[key] == original[key] == value, key
    for key in ("dataset", "checkpoint", "arm", "microbatch", "learning_rate"):
        assert original[key] == (str(getattr(args, key)) if isinstance(getattr(args, key), Path)
                                  else getattr(args, key)), key
    assert args.steps >= original["steps"] and args.steps > manifest["step"], "steps"
    assert Path(manifest["path"]).is_dir()
    return manifest


def restore_rank(model, optimizer, root, manifest, rank, device):
    state = torch.load(root / f"checkpoints/latest_rank-{rank:03d}.pt",
                       map_location="cpu", weights_only=True, mmap=True)
    for key in ("step", "epoch", "position"):
        assert state[key] == manifest[key], (rank, key)
    optimizer.load_state_dict(torch.load(root / "checkpoints/latest_optimizer.pt",
        map_location="cpu", weights_only=True, mmap=True))
    # Adam advances a parameter only when it has a gradient. A one-frame
    # episode can update the action head without using either retrieval bank.
    assert all(0 < int(value["step"]) <= manifest["step"] for value in optimizer.state.values())
    restore_memory_runtime(model, state["memory_runtime"], device)
    return state


class EpisodePlan:
    """Virtual slots fix global batches independently of physical GPU count."""

    def __init__(self, dataset, slots, seed):
        assert 1 <= slots <= len(dataset.by_episode)
        episodes = np.array(sorted(dataset.by_episode))
        np.random.default_rng(seed).shuffle(episodes)
        # Greedy length balancing reduces empty tail slots without dropping data.
        load = [0] * slots
        self.owner = {}
        for eid in sorted(episodes.tolist(), key=lambda e: -len(dataset.by_episode[e])):
            slot = min(range(slots), key=lambda i: (load[i], i))
            self.owner[eid] = slot
            load[slot] += len(dataset.by_episode[eid])
        self.orders = [[i for eid in episodes if self.owner[int(eid)] == slot
                        for i in dataset.by_episode[int(eid)]]
                       for slot in range(slots)]
        self.steps = max(map(len, self.orders))

    def batch(self, position):
        return [indices[position] if position < len(indices) else None for indices in self.orders]


class TaskBalancedPlan:
    """Cycle tasks equally; each task keeps its own chronological Episode cursor.

    Position alone reconstructs every task pass and slot owner on resume. A
    completed task pass may reshuffle only its own fully reset Episodes, while
    other tasks' detached banks remain live. No Episode is repeated mid-pass.
    """

    steps = None  # Task passes have different lengths; there is no shared epoch.

    def __init__(self, dataset, slots, seed):
        from types import SimpleNamespace
        self.dataset, self.slots, self.seed = dataset, slots, seed
        self.tasks = sorted(set(dataset.task_by_episode.values()))
        np.random.default_rng(seed).shuffle(self.tasks)
        self.views, self.plans, self.owner = {}, {}, {}
        self.task_steps = {}
        for task in self.tasks:
            episodes = {e: indices for e, indices in dataset.by_episode.items()
                        if dataset.task_by_episode[e] == task}
            view = SimpleNamespace(by_episode=episodes, groups=dataset.groups,
                order=lambda s, episodes=episodes: [i for i in dataset.order(s)
                    if dataset.groups[i][1] in episodes])
            self.views[task] = view
            plan = self._plan(task, 0)
            self.task_steps[task] = plan.steps

    def _plan(self, task, epoch):
        cached = self.plans.get(task)
        if cached is None or cached[0] != epoch:
            plan = EpisodePlan(self.views[task], self.slots, self.seed + task * 1000003 + epoch)
            self.plans[task] = epoch, plan
            self.owner.update(plan.owner)
        return self.plans[task][1]

    def progress(self, position):
        task = self.tasks[position % len(self.tasks)]
        epoch, cursor = divmod(position // len(self.tasks), self.task_steps[task])
        return {"task": task, "task_epoch": epoch, "task_position": cursor}

    def batch(self, position):
        current = self.progress(position)
        return self._plan(current["task"], current["task_epoch"]).batch(current["task_position"])


class TrainableStep(nn.Module):
    """DDP forward must contain both memory fusion and the original action loss."""

    def __init__(self, model):
        super().__init__()
        self.memory, self.action_head = model.memory, model.action_head

    def forward(self, features, ids, context, actions, noise):
        if features is None:
            # An exhausted slot joins the same reductions, without writing memory.
            return next(p for p in self.parameters() if p.requires_grad).reshape(-1)[0] * 0
        if self.memory is not None:
            features = self.memory(features, ids, context)
        seeded(noise)
        return self.action_head(features, actions)["loss"]


def backward_windows(ddp, sequences, seeds, global_count, microbatch=4, device="cuda"):
    """Match a single global masked loss, with equal collective counts per rank."""
    rank_loss = 0.
    units = torch.tensor(sum(supervision_count(seq) for seq in sequences),
                         dtype=torch.float64, device=device)
    dist.all_reduce(units)
    global_units = float(units)
    if global_units <= 0:
        raise ValueError("A training step needs valid action labels")
    chunks = (32 + microbatch-1)//microbatch
    for slot, (seq, seed) in enumerate(zip(sequences, seeds)):
        for chunk, start in enumerate(range(0, 32, microbatch)):
            active = seq is not None and start < len(seq["timesteps"])
            end = min(start+microbatch, len(seq["timesteps"])) if active else start
            final = slot == len(sequences)-1 and chunk == chunks-1
            sync = nullcontext() if final else ddp.no_sync()
            with sync, torch.autocast("cuda", dtype=torch.bfloat16, enabled=device != "cpu"):
                loss = ddp(slice_features(seq["features"], start, end, device) if active else None,
                           seq["input_ids"][start:end].to(device) if active else None,
                           window_context(seq, start, end) if active else None,
                           slice_features(seq["actions"], start, end, device) if active else None, seed+start)
                if not torch.isfinite(loss):
                    raise RuntimeError("Non-finite distributed action loss")
                weight = supervision_count(seq, start, end)/global_units if active else 0.
                rank_loss += float(loss.detach()) * weight
                (loss * weight * dist.get_world_size()).backward()
    return rank_loss


def checkpoint(model, processor, optimizer, root, args, step, epoch, position, control_group=None):
    rank, world = dist.get_rank(), dist.get_world_size()
    directory = root / f"checkpoints/step-{step:06d}"
    tick = time.monotonic()
    print(json.dumps({"kind": "checkpoint", "stage": "saving", "step": step, "rank": rank}), flush=True)
    if rank == 0:
        save_model(model, processor, directory)
        temp = root / "checkpoints/latest_optimizer.partial.pt"
        torch.save(optimizer.state_dict(), temp)
        temp.replace(root / "checkpoints/latest_optimizer.pt")
    temp = root / f"checkpoints/latest_rank-{rank:03d}.partial.pt"
    torch.save({"step": step, "epoch": epoch, "position": position,
                "memory_runtime": memory_runtime(model)}, temp)
    temp.replace(root / f"checkpoints/latest_rank-{rank:03d}.pt")
    # These files are a host-side transaction. An early rank must not launch a
    # GPU barrier kernel while the writer is copying model/optimizer to CPU.
    if control_group is None:
        dist.barrier()
    else:
        dist.monitored_barrier(group=control_group, timeout=timedelta(minutes=10), wait_all_ranks=True)
    if rank == 0:
        atomic_json(root / "checkpoints/latest.json", {"step": step, "epoch": epoch,
            "position": position, "path": str(directory), "world_size": world,
            "global_windows_per_step": args.global_windows_per_step, "seed": args.seed})
    # All rank files are complete. Rank 1 can start its next window while rank 0
    # publishes latest.json / the immutable validation queue entry; no second
    # GPU collective is needed to publish CPU files.
    print(json.dumps({"kind": "checkpoint", "stage": "saved", "step": step,
        "rank": rank, "seconds": time.monotonic()-tick}), flush=True)
    return directory


def distributed_validation(model, processor, train, validation, args, root, rank_root, step):
    # Each complete validation Episode is evaluated by exactly one rank.
    episode_map = validation.by_episode
    rank, world = dist.get_rank(), dist.get_world_size()
    validation.by_episode = {e: episode_map[e] for i, e in enumerate(sorted(episode_map)) if i % world == rank}
    try:
        result = validate(model, processor, train, validation, args, rank_root, step)
    finally:
        validation.by_episode = episode_map
    parts = [None] * world
    dist.all_gather_object(parts, result)
    episodes = {e: loss for p in parts for e, loss in p["validation_by_episode"].items()}
    assert len(episodes) == len(episode_map)
    result = {"kind": "eval", "arm": args.arm, "step": step,
        "validation_loss": float(np.mean(list(episodes.values()))), "validation_by_episode": episodes,
        "action_mae": {c: sum(p["action_mae"][c]*len(p["validation_by_episode"]) for p in parts)/len(episode_map)
                       for c in ("normal", "empty", "wrong")},
        "seconds": max(p["seconds"] for p in parts), "world_size": world,
        "action_mae_scope": parts[0]["action_mae_scope"]}
    if rank == 0:
        append_record(root / "metrics/metrics.jsonl", result)
        # Retain a common per-Episode table; detailed actions stay under each rank.
        for r in range(world):
            for name in ("action_by_episode", "validation_by_episode"):
                source = root / f"ranks/rank-{r:03d}/metrics/{name}.jsonl"
                for line in source.read_text().splitlines():
                    row = json.loads(line)
                    if row["step"] == step:
                        append_record(root / f"metrics/{name}.jsonl", row)
    return result


def train(args, control_group=None):
    rank, world = dist.get_rank(), dist.get_world_size()
    assert args.global_windows_per_step % world == 0
    assert world <= 20
    root = args.output / args.arm
    rank_root = root / f"ranks/rank-{rank:03d}"
    for base in (root, rank_root):
        for directory in ("config", "metrics", "checkpoints", "diagnostics/actions"):
            (base / directory).mkdir(parents=True, exist_ok=True)
    manifest = resume_manifest(args, root, world) if args.resume_step else None
    if not manifest:
        assert not (root / "metrics/metrics.jsonl").exists(), "Use --resume-step for an existing run"
    processor, dataset, validation, model = initialize(args, Path(manifest["path"]) if manifest else None)
    core = TrainableStep(model)
    ddp = DistributedDataParallel(core, device_ids=[int(os.environ["LOCAL_RANK"])],
        broadcast_buffers=False, find_unused_parameters=True, gradient_as_bucket_view=True)
    parameters = [p for p in core.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(parameters, lr=args.learning_rate, weight_decay=.01, foreach=False)
    start, epoch, position, best, stale = 0, 0, 0, float("inf"), 0
    if manifest:
        state = restore_rank(model, optimizer, root, manifest, rank, "cuda")
        start, epoch, position = (state[k] for k in ("step", "epoch", "position"))
        print(json.dumps({"kind": "resumed", "rank": rank, "step": start,
            "epoch": epoch, "position": position, "memory_entries":
            sum(len(v) for b in (model.memory.perceptual, model.memory.cognitive)
                for v in b.bank.values()) if model.memory else 0}), flush=True)
        del state
    multitask = getattr(dataset, "is_multitask", False)
    plan = (TaskBalancedPlan(dataset, args.global_windows_per_step, args.seed) if multitask
            else EpisodePlan(dataset, args.global_windows_per_step, args.seed+epoch))
    if rank == 0 and not manifest:
        config = {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()}
        config.update(world_size=world, decisions_per_window=32,
            global_decisions_per_step=32*args.global_windows_per_step, frozen_eagle=True,
            trainable_parameters=sum(p.numel() for p in parameters), action_head_sha256=head_hash(model),
            steps_per_epoch=plan.steps, windows_per_epoch=len(dataset), episode_slot_owner=plan.owner,
            rank_owner="slot modulo world_size", validation=f"{len(validation.by_episode)} whole independent Episodes",
            budget_note="Keep global_windows_per_step fixed to compare GPU counts; old single-card run used 1")
        if multitask:
            config.update(task_sampling="equal task cycle; chronological complete Episode passes per task",
                          task_steps_per_pass=plan.task_steps, tasks=sorted(plan.tasks))
        atomic_json(root / "config/ddp.json", config)
    if args.paired:
        peer = args.output / ("baseline" if args.arm == "memory_v1" else "memory_v1") / "config/ddp.json"
        while not peer.exists():
            time.sleep(2)
        paired = json.loads(peer.read_text())
        if not manifest:
            assert paired["action_head_sha256"] == head_hash(model)
        assert paired["global_windows_per_step"] == args.global_windows_per_step
        assert paired["seed"] == args.seed
    if control_group is None:
        dist.barrier()
    else:
        dist.monitored_barrier(group=control_group, timeout=timedelta(minutes=10), wait_all_ranks=True)
    if manifest and rank == 0 and args.async_validation:
        atomic_json(root / f"validation_queue/step-{start:06d}.json",
                    {"step": start, "checkpoint": manifest["path"]})
        atomic_json(root / f"config/resume-step-{start:06d}.json", {
            "checkpoint": manifest, "source": str(Path(__file__).resolve()),
            "optimizer_restored": True, "rank_memory_and_cursor_restored": True,
            "total_steps": args.steps})
    slots = list(range(rank, args.global_windows_per_step, world))
    for step in range(start+1, args.steps+1):
        if position == plan.steps:
            if model.memory:
                assert not model.memory.cognitive.bank and not model.memory.perceptual.bank
            epoch, position = epoch+1, 0
            plan = EpisodePlan(dataset, args.global_windows_per_step, args.seed+epoch)
        batch = plan.batch(position)
        task_progress = plan.progress(position) if multitask else {}
        count = sum(i is not None for i in batch)
        model.train()
        model.backbone.eval()
        optimizer.zero_grad(set_to_none=True)
        torch.cuda.reset_peak_memory_stats()
        torch.cuda.synchronize()
        tick = time.monotonic()
        seeds = [args.seed+100000*(step-1)+1000*s for s in slots]
        sequences = [cache_sequence(model, processor, dataset, batch[s], args.microbatch,
                                   augmentation_seed=seed)
                     if batch[s] is not None else None for s, seed in zip(slots, seeds)]
        torch.cuda.synchronize()
        encode_seconds = time.monotonic()-tick
        # Same virtual slot gets the same RNG stream under one or multiple GPUs.
        loss = backward_windows(ddp, sequences, seeds, count, args.microbatch)
        gradients = gradient_norms(model) if model.memory else {}
        norm = torch.nn.utils.clip_grad_norm_(parameters, 1.)
        assert torch.isfinite(norm)
        optimizer.step()
        torch.cuda.synchronize()
        seconds = time.monotonic()-tick
        timing = torch.tensor([seconds, torch.cuda.max_memory_allocated()/2**30, encode_seconds,
                               seconds-encode_seconds], device="cuda")
        dist.all_reduce(timing, op=dist.ReduceOp.MAX)
        total = torch.tensor(loss, device="cuda", dtype=torch.float64)
        dist.all_reduce(total)
        decisions = torch.tensor(sum(len(s["timesteps"]) for s in sequences if s), device="cuda")
        dist.all_reduce(decisions)
        position += 1
        row = {"kind": "train", "step": step, "total_steps": args.steps, "progress_percent": 100*step/args.steps,
            "arm": args.arm, "rank": rank, "epoch": epoch,
            "loss": float(total), "seconds": float(timing[0]), "world_size": world,
            "decode_processor_backbone_seconds_max_rank": float(timing[2]),
            "train_and_gradient_sync_seconds_max_rank": float(timing[3]),
            "global_windows": count, "global_decisions": int(decisions),
            "decisions_per_second_end_to_end": int(decisions)/float(timing[0]),
            "peak_allocated_gib": float(timing[1]), "gradient_norm": float(norm),
            "memory_gradients": gradients, "episodes": [s["episode"] for s in sequences if s],
            **task_progress}
        append_record(rank_root / "metrics/metrics.jsonl", row)
        atomic_json(rank_root / "status.json", {"status": "training", "step": step, "pid": os.getpid()})
        if rank == 0:
            append_record(root / "metrics/metrics.jsonl", row)
            print(json.dumps({k: v for k, v in row.items() if k != "memory_gradients"}), flush=True)
            atomic_json(root / "status.json", {"status": "training", "step": step, "total_steps": args.steps,
                "world_size": world, "pid": os.getpid(), "progress_percent": 100*step/args.steps})
        if args.train_only:
            continue
        if args.async_validation or getattr(args, "no_validation", False):
            stop_path = args.output / "metrics/stop_training.json"
            stop_at = json.loads(stop_path.read_text())["stop_after_step"] if stop_path.exists() else args.steps
            if step % args.checkpoint_interval == 0 or step == args.steps or step >= stop_at:
                optimizer.zero_grad(set_to_none=True)
                if rank == 0:
                    atomic_json(root / "status.json", {"status": "saving_checkpoint", "step": step,
                        "total_steps": args.steps, "world_size": world, "pid": os.getpid()})
                path = checkpoint(model, processor, optimizer, root, args, step, epoch, position, control_group)
                if rank == 0 and not getattr(args, "no_validation", False) and (step % args.eval_interval == 0 or step == args.steps or step >= stop_at):
                    atomic_json(root / f"validation_queue/step-{step:06d}.json",
                                {"step": step, "checkpoint": str(path)})
            if step >= stop_at:
                break
            continue
        if step % args.eval_interval == 0 or step == args.steps:
            optimizer.zero_grad(set_to_none=True)
            atomic_json(rank_root / "status.json", {"status": "validating", "step": step, "pid": os.getpid()})
            if rank == 0:
                atomic_json(root / "status.json", {"status": "validating", "step": step, "world_size": world})
            evaluation = distributed_validation(model, processor, dataset, validation, args, root, rank_root, step)
            periodic = step % args.checkpoint_interval == 0 or step == args.steps
            path = checkpoint(model, processor, optimizer, root, args, step, epoch, position, control_group) if periodic else None
            if evaluation["validation_loss"] < best:
                if path is None:
                    path = root / "checkpoints/best"
                    if rank == 0:
                        save_model(model, processor, path)
                if rank == 0:
                    atomic_json(root / "checkpoints/best.json", {"step": step, "path": str(path),
                                "validation_loss": evaluation["validation_loss"]})
            improved = evaluation["validation_loss"] < best*(1-.001)
            stale = 0 if improved else stale+1
            best = min(best, evaluation["validation_loss"])
            evaluation.update(stale_rounds=stale, best_loss=best, checkpoint=str(path) if path else None)
            stop = step == args.steps
            if rank == 0:
                atomic_json(root / f"metrics/round-{step:06d}.json", evaluation)
            dist.barrier()
            if args.paired:
                if rank == 0:
                    atomic_json(root / "status.json", {"status": "paired_validation_barrier", "step": step})
                decision = args.output / f"metrics/decision-{step:06d}.json"
                while not decision.exists():
                    time.sleep(2)
                stop = json.loads(decision.read_text())["stop"]
            elif args.patience and step >= 2000:
                stop = stop or stale >= args.patience
            if stop:
                if not periodic:
                    checkpoint(model, processor, optimizer, root, args, step, epoch, position, control_group)
                break
    if args.train_only:
        digest = hashlib.sha256()
        for name, parameter in core.named_parameters():
            digest.update(name.encode())
            digest.update(parameter.detach().cpu().contiguous().numpy().tobytes())
        digests = [None] * world
        dist.all_gather_object(digests, digest.hexdigest())
        assert len(set(digests)) == 1, "DDP parameters diverged between ranks"
        if rank == 0:
            atomic_json(root / "config/real_ddp_check.json", {"world_size": world, "steps": step,
                "backend": dist.get_backend(), "all_parameter_sha256": digests,
                "rank_parameters_bitwise_equal": True})
    if rank == 0:
        atomic_json(root / "status.json", {"status": "completed", "step": step,
            "training_complete": True,
            "best": json.loads((root / "checkpoints/best.json").read_text())
                    if (root / "checkpoints/best.json").exists() else None})


def main():
    p = argparse.ArgumentParser(description=__doc__)
    add_memory_training_options(p)
    for name in ("dataset", "checkpoint", "output"):
        p.add_argument("--"+name, type=Path, required=True)
    p.add_argument("--arm", choices=("baseline", "memory_v1"), default="memory_v1")
    p.add_argument("--global-windows-per-step", type=int, default=2)
    p.add_argument("--steps", type=int, default=10000)
    p.add_argument("--eval-interval", type=int, default=500)
    p.add_argument("--checkpoint-interval", type=int, default=1000)
    p.add_argument("--microbatch", type=int, default=4)
    p.add_argument("--learning-rate", type=float, default=1e-5)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--patience", type=int, default=0, help="0: fixed budget; paired mode uses shared early stop")
    p.add_argument("--paired", action="store_true", help="Wait for a paired controller's validation decisions")
    p.add_argument("--async-validation", action="store_true", help="Publish immutable checkpoints; independent validators consume them")
    p.add_argument("--train-only", action="store_true", help="Bounded real-model DDP check: no validation or checkpoint, compare rank weights")
    p.add_argument("--no-validation", action="store_true", help="Skip validation while retaining periodic and final model/optimizer checkpoints")
    p.add_argument("--resume-step", type=int, default=0, help="Restore matching latest model, optimizer, and per-rank Episode state")
    args = p.parse_args()
    assert args.steps > 0 and args.eval_interval > 0 and args.microbatch in (1, 2, 4, 8, 16, 32)
    assert args.checkpoint_interval > 0
    assert args.memory_length >= 1
    assert (args.async_validation and args.eval_interval % args.checkpoint_interval == 0) or (
        args.checkpoint_interval % args.eval_interval == 0)
    rank = int(os.environ["RANK"])
    torch.cuda.set_device(int(os.environ["LOCAL_RANK"]))
    torch.set_num_threads(4)
    dist.init_process_group("nccl", timeout=timedelta(minutes=10),
                            device_id=torch.device("cuda", int(os.environ["LOCAL_RANK"])))
    control_group = dist.new_group(backend="gloo", timeout=timedelta(minutes=10))
    root = args.output / args.arm
    (root / "logs").mkdir(parents=True, exist_ok=True)
    try:
        with capture_console(root / f"logs/rank-{rank:03d}.log"):
            train(args, control_group)
    except Exception as error:
        failure = {"status": "failed", "error": repr(error), "pid": os.getpid(),
                   "rank": rank, "traceback": traceback.format_exc()}
        atomic_json(root / f"status-failed-rank-{rank:03d}.json", failure)
        if rank == 0:
            atomic_json(root / "status.json", failure)
        traceback.print_exc()
        raise
    else:
        # A failed rank must exit immediately so torchrun can stop its peers.
        # Collective teardown here can hang while a peer is still in backward,
        # hiding the original Python exception for the entire NCCL timeout.
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
