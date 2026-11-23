"""Episode-affine DDP training; separate entry, never takes over an existing run."""

import argparse
import hashlib
from contextlib import nullcontext
from datetime import timedelta
from functools import partial
import json
import os
from pathlib import Path
import random
import re
import time
import traceback

import numpy as np
import torch
import torch.distributed as dist
from torch import nn
from torch.nn.parallel import DistributedDataParallel
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
from torch.distributed.fsdp import FullStateDictConfig, MixedPrecision, ShardingStrategy, StateDictType
from torch.distributed.fsdp.wrap import transformer_auto_wrap_policy
from torch.distributed.algorithms._checkpoint.checkpoint_wrapper import (
    CheckpointImpl, apply_activation_checkpointing, checkpoint_wrapper,
)

from .first_training import cache_sequence, get_training_group, gradient_norms, seeded, slice_features
from .output import capture_console
from .memory_training_options import add_memory_training_options, OPENVLA_PROTOCOL, TRAINING_PROTOCOL
from .finetuning_options import add_finetuning_options, resolve_finetuning
from .task21_diagnostic import append_record, head_hash
from .task21_formal import (atomic_json, initialize, memory_runtime,
                            restore_memory_runtime, save_model, validate, window_context, supervision_count, ACTION_SLICES)
from .task40_fusion_check import check_fusion
from .task40_state_training import LABEL_NAMES, Task40StateHead, forward_state_only, load_state_panel
from .openvla_preflight import forward_panel

# Values indicate availability of verified-state supervision. Both action-only
# paths run a real-checkpoint preflight before entering the optimization loop.
EXP10_FUSION_TRAINING = {"gated_eos": True, "dit_attention": True}


def resume_manifest(args, root, world):
    manifest = json.loads((root / "checkpoints/latest.json").read_text())
    original = json.loads((root / "config/ddp.json").read_text())
    assert resolve_finetuning(original) == resolve_finetuning(args), 'Fine-tuning/freeze/LoRA configuration changed'
    for key, default in [('weight_decay', .01), ('max_grad_norm', 1.), ('optimizer_eps', 1e-8)]:
        assert original.get(key, default) == getattr(args, key, default), key
    if hasattr(args, "training_protocol"):
        assert original.get("training_protocol") == args.training_protocol, (
            "Training sampling/augmentation changed: start a fresh run; resume older runs with their saved source snapshot")
        for key in ("memory_length", "memory_conditioning", "memory_cognitive_qk_norm", "image_augmentation"):
            assert original.get(key) == getattr(args, key), key
        assert original.get("memory_cognitive_residual_projection", False) == getattr(args, "memory_cognitive_residual_projection", False)
        assert original.get("memory_cognitive_residual_scope", "eos") == getattr(args, "memory_cognitive_residual_scope", "eos")
        assert original.get("memory_residual_readout", "delta") == getattr(args, "memory_residual_readout", "delta"), "memory_residual_readout"
    assert manifest["step"] == args.resume_step
    for key, value in (("world_size", world), ("seed", args.seed),
                       ("global_windows_per_step", args.global_windows_per_step)):
        assert manifest[key] == original[key] == value, key
    for key in ("dataset", "checkpoint", "arm", "microbatch", "learning_rate"):
        assert original[key] == (str(getattr(args, key)) if isinstance(getattr(args, key), Path)
                                  else getattr(args, key)), key
    assert args.steps >= original["steps"] and args.steps > manifest["step"], "steps"
    if original.get("training_protocol") == OPENVLA_PROTOCOL:
        assert args.steps == original["steps"], "Keep the original Experiment 12 training budget"
    assert Path(manifest["path"]).is_dir()
    return manifest


def restore_rank(model, optimizer, root, manifest, rank, device, scheduler=None, sharded=False):
    state_dir = Path(manifest["path"]) / "training_state" if scheduler else root / "checkpoints"
    state = torch.load(state_dir / f"latest_rank-{rank:03d}.pt",
                       map_location="cpu", weights_only=True, mmap=True)
    for key in ("step", "epoch", "position"):
        assert state[key] == manifest[key], (rank, key)
    optimizer_name = f"optimizer_rank-{rank:03d}.pt" if sharded else "latest_optimizer.pt"
    optimizer.load_state_dict(torch.load(state_dir / optimizer_name,
        map_location="cpu", weights_only=True, mmap=True))
    # Adam advances a parameter only when it has a gradient. A one-frame
    # episode can update the action head without using either retrieval bank.
    assert all(0 < int(value["step"]) <= manifest["step"] for value in optimizer.state.values())
    restore_memory_runtime(model, state["memory_runtime"], device)
    if scheduler:
        scheduler.load_state_dict(json.loads((state_dir / "scheduler.json").read_text()))
        assert scheduler.last_epoch == manifest["step"]
        rng = state["rng"]
        random.setstate(rng["python"])
        np.random.set_state((rng["numpy_generator"], rng["numpy_keys"].numpy().astype(np.uint32),
                             *rng["numpy_position"]))
        torch.set_rng_state(rng["torch"])
        torch.cuda.set_rng_state(rng["cuda"], device)
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


class FullGroupPlan:
    """Shuffle intact upstream-sized groups. Give every rank two real groups."""

    def __init__(self, dataset, global_groups, seed):
        self.global_groups = global_groups
        span = (dataset.group_size - 1) * dataset.history_stride
        self.groups = np.array([i for i, (source, episode, start) in enumerate(dataset.groups)
            if start + span < dataset.episode_lengths[source, episode]])
        np.random.default_rng(seed).shuffle(self.groups)
        self.steps = len(self.groups) // global_groups
        if self.steps == 0:
            raise ValueError("Insufficient complete groups for one global optimizer step")
        self.owner = {}

    def batch(self, position):
        start = position * self.global_groups
        return self.groups[start:start + self.global_groups].tolist()

    def rank_groups(self, position, rank, world, first_step, seed):
        for offset in range(position, self.steps):
            step = first_step + offset - position
            batch = self.batch(offset)
            for slot in range(rank, self.global_groups, world):
                yield batch[slot], seed + 100000 * (step - 1) + 1000 * slot


class SeededTrainingGroups(torch.utils.data.Dataset):
    def __init__(self, dataset, requests):
        self.dataset = dataset
        self.requests = list(requests)

    def __len__(self):
        return len(self.requests)

    def __getitem__(self, index):
        group, seed = self.requests[index]
        return get_training_group(self.dataset, group, seed)


class TrainableStep(nn.Module):
    """DDP forward must contain both memory fusion and the original action loss."""

    def __init__(self, model, state_processor=None):
        super().__init__()
        self.memory, self.action_head = model.memory, model.action_head
        backbone = getattr(model, 'backbone', None)
        self.backbone = backbone if backbone is not None and any(p.requires_grad for p in backbone.parameters()) else None
        self.fusion = (getattr(self.action_head, "memory_fusion_mode", "legacy") != "legacy"
                       or getattr(self.action_head, "openvla_memory", False))
        self.state_head = None
        if state_processor is not None:
            output_dim = model.action_head.model.proj_out_2.out_features
            self.state_head = Task40StateHead(output_dim).cuda()
            object.__setattr__(self, "_state_model", model)
            self.state_processor = state_processor

    def forward(self, features, ids, context, actions, noise, state_sample=None):
        if state_sample is not None:
            return forward_state_only(self._state_model, self.state_processor, self.state_head,
                                      state_sample, noise_seed=noise)
        if features is None:
            # An exhausted slot joins the same reductions, without writing memory.
            loss = next(p for p in self.parameters() if p.requires_grad).reshape(-1)[0] * 0
            return (loss, None) if self.fusion else loss
        if self.backbone is not None:
            seeded(noise)  # LoRA/backbone dropout follows the virtual slot, not physical rank.
            features = self.backbone(features)
        if self.memory is not None:
            features = self.memory(features, ids, context)
        seeded(noise)
        result = self.action_head(features, actions)
        if not self.fusion:
            return result["loss"]
        grouped = [(result["action_loss"][..., start:end].detach().sum(),
                    result["action_mask"][..., start:end].detach().sum())
                   for start, end in ACTION_SLICES.values()]
        return result["loss"], torch.stack([torch.stack(pair) for pair in grouped])


def backward_windows(ddp, sequences, seeds, global_count, microbatch=4, device="cuda", grouped_losses=None,
                     window_size=32, shard_gradients=False):
    """Match a single global masked loss, with equal collective counts per rank."""
    rank_loss = 0.
    units = torch.tensor(sum(supervision_count(seq) for seq in sequences),
                         dtype=torch.float64, device=device)
    dist.all_reduce(units)
    global_units = float(units)
    if global_units <= 0:
        raise ValueError("A training step needs valid action labels")
    chunks = (window_size + microbatch-1)//microbatch
    for slot, (seq, seed) in enumerate(zip(sequences, seeds)):
        for chunk, start in enumerate(range(0, window_size, microbatch)):
            active = seq is not None and start < len(seq["timesteps"])
            end = min(start+microbatch, len(seq["timesteps"])) if active else start
            final = slot == len(sequences)-1 and chunk == chunks-1
            # Reduce-scatter each FSDP microbatch so accumulation retains
            # only local gradient shards. Optimizer cadence remains unchanged.
            sync = nullcontext() if final or shard_gradients else ddp.no_sync()
            with sync, torch.autocast("cuda", dtype=torch.bfloat16, enabled=device != "cpu"):
                result = ddp(slice_features(seq["features"], start, end, device) if active else None,
                           seq["input_ids"][start:end].to(device) if active else None,
                           window_context(seq, start, end) if active else None,
                           slice_features(seq["actions"], start, end, device) if active else None, seed+start)
                if grouped_losses is not None:
                    loss, groups = result
                    if groups is not None:
                        grouped_losses.add_(groups)
                else:
                    loss = result
                if not torch.isfinite(loss):
                    raise RuntimeError("Non-finite distributed action loss")
                weight = supervision_count(seq, start, end)/global_units if active else 0.
                rank_loss += float(loss.detach()) * weight
                (loss * weight * dist.get_world_size()).backward()
    return rank_loss


def checkpoint(model, processor, optimizer, root, args, step, epoch, position, control_group=None,
               state_head=None, scheduler=None, sharded=None, before_save=None):
    rank, world = dist.get_rank(), dist.get_world_size()
    run = root.parent.name
    canonical = (getattr(args, "training_protocol", None) == OPENVLA_PROTOCOL
                 and re.fullmatch(r"v1__test01__exp12__run(?!000)[0-9]{3}", run))
    name = f"{run}__step{step:06d}" if canonical else f"step-{step:06d}"
    directory = root / "checkpoints" / name
    tick = time.monotonic()
    print(json.dumps({"kind": "checkpoint", "stage": "saving", "step": step, "rank": rank}), flush=True)
    state_dir = directory / "training_state" if scheduler else root / "checkpoints"
    state_dir.mkdir(parents=True, exist_ok=True)
    if sharded is not None:
        with FSDP.state_dict_type(sharded, StateDictType.FULL_STATE_DICT,
                                  FullStateDictConfig(offload_to_cpu=True, rank0_only=True)):
            full = sharded.state_dict()
        if rank == 0:
            if before_save is not None:
                before_save(full)
            save_model(model, processor, directory, state_dict=full)
            del full
    if rank == 0:
        if sharded is None:
            save_model(model, processor, directory)
        if state_head is not None:
            torch.save(state_head.state_dict(), directory / "task40_state_head.pt")
        if scheduler:
            atomic_json(state_dir / "scheduler.json", scheduler.state_dict())
    if sharded is not None or rank == 0:
        optimizer_name = f"optimizer_rank-{rank:03d}" if sharded is not None else "latest_optimizer"
        temp = state_dir / f"{optimizer_name}.partial.pt"
        torch.save(optimizer.state_dict(), temp)
        temp.replace(state_dir / f"{optimizer_name}.pt")
    temp = state_dir / f"latest_rank-{rank:03d}.partial.pt"
    rank_state = {"step": step, "epoch": epoch, "position": position,
                  "memory_runtime": memory_runtime(model)}
    if scheduler:
        numpy_state = np.random.get_state()
        rank_state["rng"] = dict(python=random.getstate(), numpy_generator=numpy_state[0],
                                 numpy_keys=torch.from_numpy(numpy_state[1].astype(np.int64)),
                                 numpy_position=numpy_state[2:], torch=torch.get_rng_state(),
                                 cuda=torch.cuda.get_rng_state())
    torch.save(rank_state, temp)
    temp.replace(state_dir / f"latest_rank-{rank:03d}.pt")
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
    openvla = args.memory_conditioning == "openvla"
    if openvla:
        assert world == 4 and args.global_windows_per_step == 8 and args.microbatch in (1, 2, 4)
        assert args.no_validation and not args.paired and not args.fusion_variant
        assert args.learning_rate == 2e-5 and args.weight_decay == 0.
        args.training_protocol = OPENVLA_PROTOCOL
    root = args.output / args.arm
    rank_root = root / f"ranks/rank-{rank:03d}"
    for base in (root, rank_root):
        for directory in ("config", "metrics", "checkpoints", "diagnostics/actions"):
            (base / directory).mkdir(parents=True, exist_ok=True)
    manifest = resume_manifest(args, root, world) if args.resume_step else None
    if not manifest:
        assert not (root / "metrics/metrics.jsonl").exists(), "Use --resume-step for an existing run"
    processor, dataset, validation, model = initialize(args, Path(manifest["path"]) if manifest else None)
    if getattr(args, "fusion_variant", None) and not manifest:
        evidence = check_fusion(model, processor, dataset, args)
        atomic_json(rank_root / "metrics/fusion_preflight.json", evidence)
        if rank == 0:
            atomic_json(root / "metrics/fusion_preflight.json", evidence)
            print(json.dumps({"kind": "fusion_preflight", **evidence}), flush=True)
    state_panel = None
    if getattr(args, "fusion_state_lambda", 0.):
        if world != 1:
            raise ValueError("Task40 auxiliary small runs require one training GPU")
        state_panel = load_state_panel(args.fusion_state_panel)
        for split in ("train", "validation"):
            if not state_panel["splits"][split]["samples"]:
                raise ValueError(f"Empty Task40 state panel split: {split}")
    preflight = openvla and args.steps == 2
    if preflight and rank == 0:
        initial_plan = FullGroupPlan(dataset, args.global_windows_per_step, args.seed)
        target = int(initial_plan.groups[0])
        donor = next(int(i) for i in initial_plan.groups if dataset.groups[i][:2] != dataset.groups[target][:2])
        panel_indices = (target, donor)
        forward_panel(model, processor, dataset, panel_indices, root / "metrics/openvla_initial.json")
    if openvla:
        dist.monitored_barrier(group=control_group, timeout=timedelta(minutes=10), wait_all_ranks=True)
    core = TrainableStep(model, validation.processor if state_panel else None)
    named_parameters = [(n, p) for n, p in core.named_parameters() if p.requires_grad]
    parameters = [p for _, p in named_parameters]
    trainable_count = sum(p.numel() for p in parameters)
    initial_head_hash = head_hash(model) if rank == 0 and not manifest else None
    groups = parameters
    if openvla:
        # Classify tensors before sharding flattens the original parameters.
        groups = [
            {"params": [p for n, p in named_parameters if
                        (p.ndim <= 1 or n.endswith(".bias"))], "weight_decay": 0.},
            {"params": [p for n, p in named_parameters if
                        p.ndim > 1 and not n.endswith(".bias")]},
        ]
        torch.backends.cuda.matmul.allow_tf32 = True
        language_layer = type(model.backbone.model.language_model.model.layers[0])
        block_types = {language_layer, type(model.action_head.model.transformer_blocks[0]),
                       type(model.memory.cognitive.retrieval_blocks[0])}
        block_types.update(type(m) for m in model.backbone.model.vision_model.modules()
                           if type(m).__name__ == "Siglip2EncoderLayer")
        ddp = FSDP(core, use_orig_params=True, sharding_strategy=ShardingStrategy.FULL_SHARD,
                   auto_wrap_policy=partial(transformer_auto_wrap_policy, transformer_layer_cls=block_types),
                   mixed_precision=MixedPrecision(param_dtype=torch.bfloat16, reduce_dtype=torch.float32,
                                                  buffer_dtype=torch.float32),
                   device_id=torch.cuda.current_device(), limit_all_gathers=True)
        if args.gradient_checkpointing:
            apply_activation_checkpointing(ddp, check_fn=lambda m: isinstance(m, language_layer),
                                           checkpoint_wrapper_fn=partial(checkpoint_wrapper,
                                               checkpoint_impl=CheckpointImpl.NO_REENTRANT))
    else:
        ddp = DistributedDataParallel(core, device_ids=[int(os.environ["LOCAL_RANK"])],
            broadcast_buffers=False, find_unused_parameters=True, gradient_as_bucket_view=True)
    optimizer = torch.optim.AdamW(groups, lr=args.learning_rate, weight_decay=getattr(args, 'weight_decay', .01),
                                  eps=getattr(args, 'optimizer_eps', 1e-8),
                                  fused=True if openvla else None, foreach=None if openvla else False)
    scheduler = None
    if openvla:
        from transformers import get_constant_schedule
        scheduler = get_constant_schedule(optimizer)
    start, epoch, position, best, stale = 0, 0, 0, float("inf"), 0
    if manifest:
        state = restore_rank(model, optimizer, root, manifest, rank, "cuda", scheduler, sharded=openvla)
        start, epoch, position = (state[k] for k in ("step", "epoch", "position"))
        print(json.dumps({"kind": "resumed", "rank": rank, "step": start,
            "epoch": epoch, "position": position, "memory_entries":
            sum(len(v) for b in (model.memory.perceptual, model.memory.cognitive)
                for v in b.bank.values()) if model.memory else 0}), flush=True)
        del state
    multitask = getattr(dataset, "is_multitask", False)
    plan = (FullGroupPlan(dataset, args.global_windows_per_step, args.seed + epoch) if openvla else
            TaskBalancedPlan(dataset, args.global_windows_per_step, args.seed) if multitask else
            EpisodePlan(dataset, args.global_windows_per_step, args.seed + epoch))
    window_size = 16 if openvla else 32
    group_reader = None
    reference = None

    def verify_before_save(state_dict):
        nonlocal reference
        _, _, _, gathered = initialize(args)
        gathered.load_state_dict(state_dict, strict=True)
        reference = forward_panel(gathered, processor, dataset, panel_indices, root / "metrics/openvla_step2.json")
        del gathered
        torch.cuda.empty_cache()
    if rank == 0 and not manifest:
        config = {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()}
        config.update(world_size=world, decisions_per_window=window_size,
            global_decisions_per_step=window_size*args.global_windows_per_step,
            frozen_eagle=not any(p.requires_grad for p in model.backbone.parameters()),
            trainable_parameters=trainable_count, action_head_sha256=initial_head_hash,
            steps_per_epoch=plan.steps, windows_per_epoch=len(dataset), episode_slot_owner=plan.owner,
            rank_owner="slot modulo world_size", validation=f"{len(validation.by_episode)} whole independent Episodes",
            budget_note="Keep global_windows_per_step fixed to compare GPU counts; old single-card run used 1")
        if openvla:
            config.update(gradient_accumulation_steps=32 // args.microbatch,
                          dataloader_workers=2, scheduler="constant", warmup_steps=0,
                          sharding="FULL_SHARD", upstream_commit="d732ea9072bc063399ccc817aed74ab172eb50be",
                          complete_groups=len(plan.groups), gpu_name=torch.cuda.get_device_name())
        if multitask:
            config.update(task_sampling="equal task cycle; chronological complete Episode passes per task",
                          task_steps_per_pass=plan.task_steps, tasks=sorted(plan.tasks))
        config.update(resolve_finetuning(args))
        atomic_json(root / 'config/finetuning.json', model.finetuning_summary)
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
        assert resolve_finetuning(paired) == resolve_finetuning(args)
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
    state_fusion_gradient_seen = False
    for step in range(start+1, args.steps+1):
        if position == plan.steps:
            if model.memory:
                assert not model.memory.cognitive.bank and not model.memory.perceptual.bank
            epoch, position = epoch+1, 0
            plan = (FullGroupPlan(dataset, args.global_windows_per_step, args.seed + epoch) if openvla
                    else EpisodePlan(dataset, args.global_windows_per_step, args.seed + epoch))
            group_reader = None
        batch = plan.batch(position)
        task_progress = plan.progress(position) if multitask else {}
        count = sum(i is not None for i in batch)
        model.train()
        if not any(p.requires_grad for p in model.backbone.parameters()):
            model.backbone.eval()
        optimizer.zero_grad(set_to_none=True)
        torch.cuda.reset_peak_memory_stats()
        torch.cuda.synchronize()
        tick = time.monotonic()
        seeds = [args.seed+100000*(step-1)+1000*s for s in slots]
        if openvla:
            if group_reader is None:
                prefetch = SeededTrainingGroups(dataset, plan.rank_groups(position, rank, world, step, args.seed))
                # Workers decode and transform CPU observations. A dedicated
                # generator keeps worker initialization out of policy RNG.
                group_reader = iter(torch.utils.data.DataLoader(prefetch, batch_size=None, num_workers=2,
                    generator=torch.Generator().manual_seed(args.seed + epoch)))
            sequences = [cache_sequence(model, processor, dataset, batch[s], args.microbatch,
                                        group=next(group_reader)) for s in slots]
            if any(len(sequence["timesteps"]) != window_size for sequence in sequences):
                raise RuntimeError("Incomplete decision group in a fixed 128-sample batch")
            for sequence in sequences:
                # Reproduce upstream group reset across the accumulated chunks.
                sequence["episode_start"] = sequence["episode_end"] = True
        else:
            sequences = [cache_sequence(model, processor, dataset, batch[s], args.microbatch,
                                       augmentation_seed=seed)
                         if batch[s] is not None else None for s, seed in zip(slots, seeds)]
        torch.cuda.synchronize()
        encode_seconds = time.monotonic()-tick
        # Same virtual slot gets the same RNG stream under one or multiple GPUs.
        grouped = torch.zeros((len(ACTION_SLICES), 2), device="cuda", dtype=torch.float64) if core.fusion else None
        loss = backward_windows(ddp, sequences, seeds, count, args.microbatch, grouped_losses=grouped,
                                window_size=window_size, shard_gradients=openvla)
        monitor = not openvla or step == 1 or step % 10 == 0 or step == args.steps
        gradients = gradient_norms(model) if model.memory and monitor else {}
        training_monitor = {}
        if openvla and monitor:
            components = ("backbone", "action_head.model", "action_head.memory_", "memory.cognitive", "memory.perceptual")
            component_parameters = {
                prefix: [(name, p) for name, p in core.named_parameters()
                         if name.startswith(prefix) and p.grad is not None]
                for prefix in components}
            with torch.no_grad():
                norms_sq = torch.stack([
                    torch.stack([p.grad.float().norm().square() for _, p in found]).sum()
                    if found else torch.zeros((), device="cuda")
                    for found in component_parameters.values()])
                dist.all_reduce(norms_sq)
                training_monitor["component_grad_norms"] = dict(zip(components, norms_sq.sqrt().tolist()))
            if preflight and step == 2:
                if not all(value > 0 for value in training_monitor["component_grad_norms"].values()):
                    raise RuntimeError("Missing gradients in an OpenVLA component")
                tracked = []
                for found in component_parameters.values():
                    candidates = [(p, p.grad.detach().abs().flatten()[0]) for _, p in found if p.numel()]
                    if not candidates:
                        tracked.append(None)
                        continue
                    param, _ = max(candidates, key=lambda item: float(item[1]))
                    index = int(param.grad.detach().abs().flatten().argmax())
                    tracked.append((param, index, param.detach().flatten()[index].clone()))
            training_monitor["learning_rate"] = optimizer.param_groups[0]["lr"]
            training_monitor["microbatch"] = args.microbatch
            training_monitor["gradient_accumulation_steps"] = 32 // args.microbatch
        fusion_gradients = {name: float(parameter.grad.detach().norm()) if parameter.grad is not None else None
                            for name, parameter in core.named_parameters() if parameter.requires_grad} if core.fusion and not openvla else {}
        state_evidence = {}
        if state_panel:
            samples = state_panel["splits"]["train"]["samples"]
            index = (step - 1) % len(samples)
            action_grads = [p.grad.detach().clone() if p.grad is not None else None for p in parameters]
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast("cuda", dtype=torch.bfloat16):
                state = ddp(None, None, None, None, args.seed + 10000000 + step, state_sample=samples[index])
            if not torch.isfinite(state["loss"]):
                raise RuntimeError("Non-finite Task40 state loss")
            (args.fusion_state_lambda * state["loss"]).backward()
            state_gradients = {name: float(p.grad.detach().float().norm()) if p.grad is not None else None
                               for name, p in core.named_parameters() if p.requires_grad}
            state_fusion_gradient_seen |= any(
                p is not None and p > 0 for name, p in state_gradients.items() if "cognitive_fusion" in name
            )
            if step >= min(len(samples), args.steps) and not state_fusion_gradient_seen:
                raise RuntimeError("The Task40 state loss did not reach the fusion module")
            for parameter, action_grad in zip(parameters, action_grads):
                if action_grad is not None:
                    if parameter.grad is None:
                        parameter.grad = action_grad
                    else:
                        parameter.grad.add_(action_grad)
            state_evidence = {"state_panel_index": index, "state_loss": float(state["loss"].detach()),
                              "state_loss_weight": args.fusion_state_lambda,
                              "weighted_state_grad_norms": state_gradients}
            loss += args.fusion_state_lambda * state_evidence["state_loss"]
        max_norm = getattr(args, "max_grad_norm", 1.)
        norm = ddp.clip_grad_norm_(max_norm) if openvla else torch.nn.utils.clip_grad_norm_(parameters, max_norm)
        assert torch.isfinite(norm)
        optimizer.step()
        if scheduler:
            scheduler.step()
        if preflight and step == 2:
            updates = torch.stack([
                (sample[0].detach().flatten()[sample[1]] - sample[2]).abs() if sample is not None
                else torch.zeros((), device="cuda") for sample in tracked
            ])
            dist.all_reduce(updates, op=dist.ReduceOp.MAX)
            training_monitor["component_sample_update_abs"] = dict(zip(components, updates.tolist()))
            if not all(value > 0 for value in training_monitor["component_sample_update_abs"].values()):
                raise RuntimeError("OpenVLA preflight component did not update")
            del tracked
        torch.cuda.synchronize()
        seconds = time.monotonic()-tick
        timing = torch.tensor([seconds, torch.cuda.max_memory_allocated()/2**30, encode_seconds,
                               seconds-encode_seconds], device="cuda")
        dist.all_reduce(timing, op=dist.ReduceOp.MAX)
        total = torch.tensor(loss, device="cuda", dtype=torch.float64)
        dist.all_reduce(total)
        decisions = torch.tensor(sum(len(s["timesteps"]) for s in sequences if s), device="cuda")
        local_decisions = int(decisions)
        dist.all_reduce(decisions)
        if openvla:
            assert local_decisions == 32 and int(decisions) == 128
            training_monitor.update(gpu_name=torch.cuda.get_device_name(),
                gpu_peak_reserved_gib=torch.cuda.max_memory_reserved()/2**30,
                gradient_norm_after_clip=min(float(norm), max_norm))
        position += 1
        row = {"kind": "train", "step": step, "total_steps": args.steps, "progress_percent": 100*step/args.steps,
            "arm": args.arm, "rank": rank, "epoch": epoch,
            "loss": float(total), "seconds": float(timing[0]), "world_size": world,
            "decode_processor_backbone_seconds_max_rank": float(timing[2]),
            "train_and_gradient_sync_seconds_max_rank": float(timing[3]),
            "global_windows": count, "global_decisions": int(decisions), "local_decisions": local_decisions,
            "decisions_per_second_end_to_end": int(decisions)/float(timing[0]),
            "peak_allocated_gib": float(timing[1]), "gradient_norm": float(norm),
            "memory_gradients": gradients, "episodes": [s["episode"] for s in sequences if s],
            **task_progress, **state_evidence, **training_monitor}
        if grouped is not None:
            dist.all_reduce(grouped)
            row["body_flow_mse"] = {group: float(error / units) if units > 0 else None
                                    for group, (error, units) in zip(ACTION_SLICES, grouped.tolist())}
            if not openvla:
                row["fusion_action_grad_norms"] = fusion_gradients
        append_record(rank_root / "metrics/metrics.jsonl", row)
        atomic_json(rank_root / "status.json", {"status": "training", "step": step, "pid": os.getpid()})
        if rank == 0:
            append_record(root / "metrics/metrics.jsonl", row)
            if monitor:
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
                path = checkpoint(model, processor, optimizer, root, args, step, epoch, position, control_group,
                                  core.state_head, scheduler, sharded=ddp if openvla else None,
                                  before_save=verify_before_save if preflight and step == 2 else None)
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
            if state_panel:
                state_rows = []
                with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
                    for i, sample in enumerate(state_panel["splits"]["validation"]["samples"]):
                        value = forward_state_only(model, validation.processor, core.state_head, sample,
                                                   noise_seed=args.seed + 19000000 + i)
                        row_state = {key: value[key][0].detach().cpu().tolist()
                                     for key in ("logits", "targets", "target_mask")}
                        state_rows.append(row_state)
                atomic_json(root / f"metrics/state_validation-{step:06d}.json",
                            {"step": step, "labels": LABEL_NAMES, "rows": state_rows})
            periodic = step % args.checkpoint_interval == 0 or step == args.steps
            path = checkpoint(model, processor, optimizer, root, args, step, epoch, position, control_group,
                              core.state_head) if periodic else None
            if evaluation["validation_loss"] < best:
                if path is None:
                    path = root / "checkpoints/best"
                    if rank == 0:
                        save_model(model, processor, path)
                        if core.state_head is not None:
                            torch.save(core.state_head.state_dict(), path / "task40_state_head.pt")
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
                    checkpoint(model, processor, optimizer, root, args, step, epoch, position, control_group,
                               core.state_head)
                break
    if preflight and step == 2:
        saved = json.loads((root / "checkpoints/latest.json").read_text()) if rank == 0 else None
        packet = [saved]
        dist.broadcast_object_list(packet, group=control_group)
        restore_rank(model, optimizer, root, packet[0], rank, "cuda", scheduler, sharded=True)
        if rank == 0:
            _, _, _, reloaded = initialize(args, path)
            after = forward_panel(reloaded, processor, dataset, panel_indices, root / "metrics/openvla_reload.json")
            assert torch.equal(reference, after), "Strict reload changed actions for identical input and RNG"
            atomic_json(root / "metrics/openvla_reload_proof.json",
                        {"action_bitwise_equal": True, "restored_ranks": world, "checkpoint": str(path)})
    if args.train_only:
        digest = hashlib.sha256()
        for name, parameter in core.named_parameters():
            digest.update(name.encode())
            digest.update(parameter.detach().cpu().contiguous().view(torch.uint8).numpy().tobytes())
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
    add_finetuning_options(p)
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
    p.add_argument("--fusion-variant", choices=("gated_eos", "dit_attention"))
    p.add_argument("--fusion-dit-layers", nargs="+", type=int)
    p.add_argument("--fusion-state-lambda", type=float)
    p.add_argument("--fusion-state-panel", type=Path)
    args = p.parse_args()
    if args.fusion_variant and args.fusion_state_lambda is None:
        p.error("Specify an explicit Task40 state-loss weight (0 for action only)")
    if args.fusion_state_lambda and (not args.fusion_state_panel or args.resume_step):
        p.error("Task40 state-supervised training requires a state panel and a fresh job")
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
