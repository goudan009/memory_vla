"""Matched A100 training; detached Episode memory persists across 32-decision steps."""

import argparse
from copy import deepcopy
from contextlib import contextmanager
import json
import os
from pathlib import Path
import time

import numpy as np
import torch
from transformers.feature_extraction_utils import BatchFeature

from .first_training import (cache_sequence, slice_features, seeded, write_json,
                             gradient_norms, bank_trace)
from .task21_diagnostic import load_model, head_hash, append_record, action_predictions
from .memory_training_options import add_memory_training_options
from v1_groot_memoryvla.data.dataset.episode_windows import EpisodeWindows
from v1_groot_memoryvla.data.episode_split import validate_split
from v1_groot_memoryvla.model.gr00t_n1d6.processing_gr00t_n1d6 import Gr00tN1d6Processor


def atomic_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".partial.json")
    write_json(temporary, value)
    temporary.replace(path)


def window_context(seq, start, end):
    return {"mode": "train_sequence", "sequence_start": seq["episode_start"] and start == 0,
            "sequence_end": seq["episode_end"] and end == len(seq["timesteps"]),
            "episode_keys": [seq["episode_key"]] * (end-start),
            "timesteps": seq["timesteps"][start:end]}


def supervision_count(seq, start=0, end=None):
    """Loss units for exact microbatch/DDP weighting, including action tail masks."""
    if seq is None:
        return 0.
    end = len(seq["timesteps"]) if end is None else end
    mask = seq["actions"].get("action_mask")
    return float(mask[start:end].sum()) if mask is not None else float(end-start)


@contextmanager
def isolated_validation(model):
    """Swap runtime dictionaries, retaining the exact live training bank objects."""
    state = []
    if model.memory is not None:
        for bank in (model.memory.perceptual, model.memory.cognitive):
            state.append((bank, bank.bank, bank.last_written, bank.merge_count))
            bank.bank, bank.last_written, bank.merge_count = {}, {}, {}
    try:
        yield
    finally:
        for bank, history, last, merges in state:
            bank.bank, bank.last_written, bank.merge_count = history, last, merges


def fused_window(model, seq, noise, backward=False, microbatch=4, collect=False):
    total, outputs, trace = 0., [], []
    count, units = len(seq["timesteps"]), supervision_count(seq)
    for start in range(0, count, microbatch):
        end = min(start+microbatch, count)
        features = slice_features(seq["features"], start, end)
        actions = slice_features(seq["actions"], start, end)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            if 'input_ids' in features:
                seeded(noise + start)
                features = model.backbone(features)
            if model.memory is not None:
                with bank_trace(model.memory) as events:
                    features = model.memory(features, seq["input_ids"][start:end].cuda(),
                                            window_context(seq, start, end))
                if collect:
                    trace.extend(events)
            seeded(noise + start)  # shared action RNG, after the memory operations
            loss = model.action_head(features, actions)["loss"]
        if not torch.isfinite(loss):
            raise RuntimeError("Non-finite action loss")
        weight = supervision_count(seq, start, end) / units
        total += float(loss.detach()) * weight
        if backward:
            (loss * weight).backward()
        if collect:
            outputs.append({k: v.detach().cpu() for k, v in features.items() if k != "visual_features"})
    combined = {k: torch.cat([v[k] for v in outputs]) for k in outputs[0]} if collect else None
    return total, combined, trace


@torch.no_grad()
def wrong_features(model, target, donors, trace):
    """Use another training Episode's content in the target's exact ToMe time slots."""
    adapter, key = model.memory, "wrong-history-control"
    adapter.reset([key])
    banks = (adapter.perceptual, adapter.cognitive)

    def stream(seq, i, t):
        return adapter(slice_features(seq["features"], i, i+1), seq["input_ids"][i:i+1].cuda(),
                       {"mode": "stream", "episode_keys": [key], "timesteps": [t]})

    times = {label: [e["read_timesteps"] for e in trace if e["bank"] == label]
             for label in ("perceptual", "cognitive")}
    def donor_frames():
        for seq in donors:
            for i in range(len(seq["timesteps"])):
                yield seq, i
    frames = iter(donor_frames())
    filled = len(times["perceptual"][0])
    result = []
    with torch.autocast("cuda", dtype=torch.bfloat16):
        for t in range(filled):
            seq, i = next(frames)
            stream(seq, i, t)
        for i, t in enumerate(target["timesteps"]):
            previous = [(list(b.bank.get(key, [])), b.last_written.get(key, -1), b.merge_count.get(key, 0)) for b in banks]
            for label, b in zip(("perceptual", "cognitive"), banks):
                assert len(times[label][i]) == len(b.bank.get(key, []))
                b.bank[key] = [(s, f) for s, (_, f) in zip(times[label][i], b.bank.get(key, []))]
                b.last_written[key] = t-1
            out = stream(target, i, t)
            result.append({k: v.detach().cpu() for k, v in out.items()})
            for b, (history, last, merges) in zip(banks, previous):
                b.bank[key], b.last_written[key], b.merge_count[key] = history, last, merges
            donor, j = next(frames)
            stream(donor, j, filled+i)
    adapter.reset([key])
    return {k: torch.cat([v[k] for v in result]) for k in result[0]}


@torch.no_grad()
def empty_features(model, seq):
    """Use the same conditioning interface, without reading or writing history."""
    if model.memory is None:
        return {k: v for k, v in seq["features"].items() if k != "visual_features"}
    outputs = []
    with torch.autocast("cuda", dtype=torch.bfloat16):
        for i, t in enumerate(seq["timesteps"]):
            out = model.memory(slice_features(seq["features"], i, i+1), seq["input_ids"][i:i+1].cuda(),
                {"mode": "empty", "episode_keys": [seq["episode_key"]], "timesteps": [t]})
            outputs.append({k: v.detach().cpu() for k, v in out.items()})
    return {k: torch.cat([v[k] for v in outputs]) for k in outputs[0]}


@torch.no_grad()
def action_metrics(model, seq, normal, wrong, root, step, arm, donor_episode, empty=None):
    empty = empty_features(model, seq) if empty is None else empty
    conditions = {"normal": normal, "empty": empty, "wrong": wrong} if model.memory else {"normal": empty}
    seed = 909 + seq["episode"] * 100
    predictions = {name: action_predictions(model, f, seq, seed) for name, f in conditions.items()}
    if model.memory is None:
        predictions.update(empty=predictions["normal"], wrong=predictions["normal"])
    target = seq["actions"]["action"][:, :32, :23].float().numpy()
    mask = seq["actions"].get("action_mask")
    mask = mask[:, :32, :23].float().numpy() if mask is not None else np.ones_like(target)
    rows = []
    for name, pred in predictions.items():
        error = np.abs(pred-target)
        row = {"step": step, "arm": arm, "episode": seq["episode"], "condition": name,
               "task": seq.get("task"),
               "unique_observations": len(target), "noise_seed": seed, "donor_episode": donor_episode,
               "scope": "fixed middle window with full preceding Episode memory; valid action labels only",
               "mae_first8": float((error[:, :8]*mask[:, :8]).sum()/mask[:, :8].sum()),
               "mae_horizon32": float((error*mask).sum()/mask.sum()),
               "normal_empty_action_shift": float(np.abs(predictions["normal"]-predictions["empty"]).mean()),
               "normal_wrong_action_shift": float(np.abs(predictions["normal"]-predictions["wrong"]).mean())}
        append_record(root / "metrics/action_by_episode.jsonl", row)
        rows.append(row)
    np.savez_compressed(root / f"diagnostics/actions/step{step:06d}_episode{seq['episode']}.npz",
        **predictions, target=target, action_mask=mask, frames=np.asarray(seq["frame_indices"]))
    return rows


def donor_episode(train, validation, eid):
    task = validation.task_by_episode.get(eid)
    candidates = sorted(e for e in train.by_episode
                        if train.task_by_episode.get(e) == task and e != eid)
    assert candidates, "No independent same-task training donor"
    return candidates[eid % len(candidates)]


@torch.no_grad()
def validate(model, processor, train, validation, args, root, step):
    model.eval()
    episode_results, action_rows = {}, []
    tick = time.monotonic()
    with isolated_validation(model), evaluation_processors(train.processor, validation.processor):
        for eid, indices in sorted(validation.by_episode.items()):
            losses, weights, decisions = [], [], 0
            middle = len(indices)//2
            for w, index in enumerate(indices):
                seq = cache_sequence(model, processor, validation, index, args.microbatch)
                loss, normal, trace = fused_window(model, seq, 100000+eid*10000+w*100,
                    microbatch=args.microbatch, collect=w == middle)
                losses.append(loss)
                weights.append(supervision_count(seq))
                decisions += len(seq["timesteps"])
                if w == 0 or (w+1) % 10 == 0 or w+1 == len(indices):
                    progress = {"status": "validating", "checkpoint_step": step, "arm": args.arm,
                        "episode": eid, "completed_episodes": len(episode_results),
                        "total_episodes": len(validation.by_episode), "episode_windows_done": w+1,
                        "episode_windows_total": len(indices), "loss_so_far": float(np.average(losses, weights=weights)),
                        "seconds": time.monotonic()-tick, "pid": os.getpid()}
                    atomic_json(root / "validation_status.json", progress)
                    print(json.dumps({"validation_progress": progress}), flush=True)
                if w == middle:
                    donor_eid = donor_episode(train, validation, eid)
                    donors = train.by_episode[donor_eid]
                    wrong = None
                    if model.memory is not None:
                        # Bound work by bank capacity + target window, and permit
                        # short donor episodes without borrowing target content.
                        def donor_sequences():
                            while True:
                                for donor_index in donors:
                                    yield cache_sequence(model, processor, train, donor_index, args.microbatch)
                        wrong = wrong_features(model, seq, donor_sequences(), trace)
                    action_rows.extend(action_metrics(model, seq, normal, wrong, root, step, args.arm, donor_eid))
            episode_results[str(eid)] = float(np.average(losses, weights=weights))
            row = {"step": step, "episode": eid, "windows": len(indices),
                   "task": validation.task_by_episode.get(eid),
                   "decisions": decisions, "loss": episode_results[str(eid)]}
            append_record(root / "metrics/validation_by_episode.jsonl", row)
            print(json.dumps({"validation_episode": row}), flush=True)
        if model.memory is not None:
            assert not model.memory.cognitive.bank and not model.memory.perceptual.bank
    result = {"kind": "eval", "arm": args.arm, "step": step,
              "validation_loss": float(np.mean(list(episode_results.values()))),
              "validation_by_episode": episode_results,
              "action_mae": {c: float(np.mean([r["mae_horizon32"] for r in action_rows if r["condition"] == c]))
                             for c in ("normal", "empty", "wrong")},
              "action_mae_scope": "one fixed middle window per independent validation Episode",
              "seconds": time.monotonic()-tick}
    if validation.task_by_episode:
        result["validation_by_task"] = {}
        for task in sorted(set(validation.task_by_episode.values())):
            ids = [e for e in episode_results if validation.task_by_episode[int(e)] == task]
            if not ids:
                continue
            result["validation_by_task"][str(task)] = {
                "episodes": len(ids), "loss": float(np.mean([episode_results[e] for e in ids])),
                "action_mae": {c: float(np.mean([r["mae_horizon32"] for r in action_rows
                    if r["task"] == task and r["condition"] == c]))
                    for c in ("normal", "empty", "wrong")}}
        result["task_macro_validation_loss"] = float(np.mean([
            r["loss"] for r in result["validation_by_task"].values()]))
    append_record(root / "metrics/metrics.jsonl", result)
    return result


def save_model(model, processor, directory):
    model.save_pretrained(directory, max_shard_size="5GB")
    processor.save_pretrained(directory)


def memory_runtime(model, device="cpu"):
    runtime = []
    if model.memory is not None:
        for b in (model.memory.perceptual, model.memory.cognitive):
            runtime.append({"bank": {k: [(t, f.detach().to(device)) for t, f in history]
                                     for k, history in b.bank.items()},
                            "last_written": dict(b.last_written), "merge_count": dict(b.merge_count)})
    return runtime


def restore_memory_runtime(model, runtime, device):
    assert len(runtime) == (2 if model.memory is not None else 0)
    if model.memory is not None:
        for b, state in zip((model.memory.perceptual, model.memory.cognitive), runtime):
            b.bank = {k: [(t, f.detach().to(device)) for t, f in history]
                      for k, history in state["bank"].items()}
            b.last_written, b.merge_count = dict(state["last_written"]), dict(state["merge_count"])


@contextmanager
def evaluation_processors(*processors):
    """Donor data also uses eval transforms; restore training augmentation after validation."""
    states = [(p, p.training) for p in processors]
    try:
        for p, _ in states:
            p.eval()
        yield
    finally:
        for p, training in states:
            p.train() if training else p.eval()


def save_checkpoint(model, processor, optimizer, root, step, epoch, position):
    directory = root / f"checkpoints/step-{step:06d}"
    save_model(model, processor, directory)
    temporary = root / "checkpoints/latest_training_state.partial.pt"
    torch.save({"step": step, "epoch": epoch, "position": position,
        "checkpoint": str(directory), "optimizer": optimizer.state_dict(),
        "memory_runtime": memory_runtime(model)}, temporary)
    temporary.replace(root / "checkpoints/latest_training_state.pt")
    return directory


def initialize(args, checkpoint=None):
    torch.set_num_threads(4)
    restoring = checkpoint is not None
    checkpoint = checkpoint or args.checkpoint
    saved = json.loads((Path(checkpoint) / "config.json").read_text())
    processor = Gr00tN1d6Processor.from_pretrained(checkpoint)
    processor.train() if getattr(args, "image_augmentation", True) else processor.eval()
    eval_processor = deepcopy(processor)
    eval_processor.eval()
    split = json.loads((args.output / "config/data_split.json").read_text())
    episode_tasks = validate_split(split)
    train = EpisodeWindows(args.dataset, processor, split["train"], episode_tasks=episode_tasks)
    validation = EpisodeWindows(args.dataset, eval_processor, split["validation"], episode_tasks=episode_tasks)
    # The async step-0 job also passes a checkpoint explicitly, but that is
    # still the original GR00T model: build the same new adapter as training.
    memory_options = {} if restoring and saved.get("memory_enabled") else {
        "memory_conditioning": getattr(args, "memory_conditioning", "residual"),
        "memory_cognitive_qk_norm": getattr(args, "memory_cognitive_qk_norm", "layernorm"),
        "memory_length": getattr(args, "memory_length", 64)}
    if not (restoring and saved.get("memory_enabled")) and memory_options["memory_conditioning"] == "residual":
        if not saved.get("memory_enabled"):
            memory_options["memory_residual_identity_init"] = True
        if getattr(args, "memory_cognitive_residual_projection", False):
            memory_options["memory_cognitive_residual_projection"] = True
        if getattr(args, "memory_cognitive_residual_scope", "eos") != "eos":
            memory_options["memory_cognitive_residual_scope"] = args.memory_cognitive_residual_scope
        if getattr(args, "memory_residual_readout", "delta") != "delta":
            if saved.get("memory_enabled") and saved.get("memory_residual_readout", "delta") != args.memory_residual_readout:
                raise ValueError("Memory residual readout differs from checkpoint; start from the GR00T baseline checkpoint")
            memory_options["memory_residual_readout"] = args.memory_residual_readout
    if not restoring and args.arm == "memory_v1":
        # A fresh architecture starts from GR00T, not silently from trained
        # residual-memory parameters under a different conditioning interface.
        if saved.get("memory_enabled") and saved.get("memory_conditioning", "residual") != memory_options["memory_conditioning"]:
            raise ValueError("Memory conditioning differs from checkpoint; start this run from the GR00T baseline checkpoint")
        if saved.get("memory_enabled") and saved.get("memory_cognitive_qk_norm", "none") != memory_options["memory_cognitive_qk_norm"]:
            raise ValueError("Memory QK normalization differs from checkpoint; start this run from the GR00T baseline checkpoint")
    model = load_model(checkpoint, args.arm == "memory_v1", args.seed, finetuning=args, **memory_options)
    model.backbone.cuda().eval()
    return processor, train, validation, model


def train_worker(args):
    root = args.output / args.arm
    for directory in ("config", "metrics", "diagnostics/actions", "checkpoints"):
        (root / directory).mkdir(parents=True, exist_ok=True)
    state = None
    if args.resume_step:
        state = torch.load(root / "checkpoints/latest_training_state.pt", map_location="cpu", weights_only=False)
        assert state["step"] == args.resume_step
    processor, train, validation, model = initialize(args, state["checkpoint"] if state else None)
    frozen = {n: p.requires_grad for n, p in model.action_head.named_parameters()}
    initial = {"action_head_sha256": head_hash(model), "action_head_freezing": frozen,
               "frozen_eagle": not any(p.requires_grad for p in model.backbone.parameters()),
               "trainable_parameters": sum(p.numel() for p in model.parameters() if p.requires_grad),
               "windows_per_epoch": len(train), "gpu": torch.cuda.get_device_name(0)}
    if not state:
        atomic_json(root / "config/initialization.json", initial)
    else:
        original = json.loads((root / "config/initialization.json").read_text())
        assert initial["action_head_freezing"] == original["action_head_freezing"]
        assert initial["frozen_eagle"] == original["frozen_eagle"]
    other = "baseline" if args.arm == "memory_v1" else "memory_v1"
    peer = args.output / other / "config/initialization.json"
    while not peer.exists():
        time.sleep(2)
    paired = json.loads(peer.read_text())
    if not state:
        assert initial["action_head_sha256"] == paired["action_head_sha256"]
    assert initial["action_head_freezing"] == paired["action_head_freezing"]
    assert initial["frozen_eagle"] and paired["frozen_eagle"]
    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(params, lr=args.learning_rate, weight_decay=.01, foreach=False)
    best, stale, epoch, position = float("inf"), 0, 0, 0
    if state:
        optimizer.load_state_dict(state["optimizer"])
        restore_memory_runtime(model, state["memory_runtime"], "cuda")
        epoch, position = state["epoch"], state["position"]
        best = json.loads((root / "checkpoints/best.json").read_text())["validation_loss"]
        stale = json.loads((root / f"metrics/round-{args.resume_step:06d}.json").read_text())["stale_rounds"]
        steps = {int(v["step"].item()) for v in optimizer.state.values() if "step" in v}
        assert steps and all(0 < step <= args.resume_step for step in steps)
        report = {"step": args.resume_step, "epoch": epoch, "position": position,
                  "optimizer_steps": sorted(steps), "checkpoint_interval": args.checkpoint_interval,
                  "memory_episodes": len(model.memory.cognitive.bank) if model.memory else 0,
                  "memory_last_written": dict(model.memory.cognitive.last_written) if model.memory else {},
                  "action_head_sha256": initial["action_head_sha256"], "pid": os.getpid()}
        atomic_json(root / f"config/resume-{args.resume_step:06d}.json", report)
        print(json.dumps({"resumed": report}), flush=True)
        del state
    order = train.order(args.seed+epoch)
    for step in range(args.resume_step+1, args.steps+1):
        if position == len(order):
            epoch, position = epoch+1, 0
            order = train.order(args.seed+epoch)
            if model.memory:
                assert not model.memory.cognitive.bank
        model.train()
        model.backbone.eval()
        optimizer.zero_grad(set_to_none=True)
        torch.cuda.reset_peak_memory_stats()
        torch.cuda.synchronize()
        tick = time.monotonic()
        seq = cache_sequence(model, processor, train, order[position], args.microbatch,
                             augmentation_seed=args.seed+100000*(step-1))
        loss, _, _ = fused_window(model, seq, args.seed+1000*(step-1), backward=True,
                                  microbatch=args.microbatch)
        gradients = gradient_norms(model) if model.memory else {}
        norm = torch.nn.utils.clip_grad_norm_(params, 1.)
        assert torch.isfinite(norm)
        optimizer.step()
        torch.cuda.synchronize()
        seconds = time.monotonic()-tick
        row = {"kind": "train", "arm": args.arm, "step": step, "loss": loss,
               "episode": seq["episode"], "window": seq["window"], "epoch": epoch,
               "frame_start": seq["frame_indices"][0], "episode_start": seq["episode_start"],
               "episode_end": seq["episode_end"], "seconds": seconds,
               "decisions_per_second_end_to_end": len(seq["timesteps"])/seconds,
               "peak_allocated_gib": torch.cuda.max_memory_allocated()/2**30,
               "peak_reserved_gib": torch.cuda.max_memory_reserved()/2**30,
               "gradient_norm": float(norm), "memory_gradients": gradients}
        append_record(root / "metrics/metrics.jsonl", row)
        print(json.dumps({k: v for k, v in row.items() if k != "memory_gradients"}), flush=True)
        position += 1
        atomic_json(root / "status.json", {"status": "training", "step": step,
                    "episode": seq["episode"], "pid": os.getpid(), "last_step_seconds": seconds})
        if step % args.eval_interval == 0 or step == args.steps:
            optimizer.zero_grad(set_to_none=True)
            atomic_json(root / "status.json", {"status": "validating", "step": step, "pid": os.getpid()})
            evaluation = validate(model, processor, train, validation, args, root, step)
            periodic = step % args.checkpoint_interval == 0 or step == args.steps
            checkpoint = save_checkpoint(model, processor, optimizer, root, step, epoch, position) if periodic else None
            # Keep the absolute minimum checkpoint; patience uses a 0.1% relative threshold.
            if evaluation["validation_loss"] < best:
                if checkpoint is None:
                    checkpoint = root / "checkpoints/best"
                    save_model(model, processor, checkpoint)
                atomic_json(root / "checkpoints/best.json", {"step": step, "path": str(checkpoint),
                            "validation_loss": evaluation["validation_loss"]})
            improved = evaluation["validation_loss"] < best * (1-args.min_relative_improvement)
            stale = 0 if improved else stale+1
            best = min(best, evaluation["validation_loss"])
            evaluation.update(stale_rounds=stale, checkpoint=str(checkpoint) if checkpoint else None,
                              periodic_checkpoint=periodic, best_loss=best)
            atomic_json(root / f"metrics/round-{step:06d}.json", evaluation)
            atomic_json(root / "status.json", {"status": "paired_validation_barrier", "step": step,
                                               "pid": os.getpid()})
            decision = args.output / f"metrics/decision-{step:06d}.json"
            while not decision.exists():
                time.sleep(2)
            if json.loads(decision.read_text())["stop"]:
                if not periodic:
                    save_checkpoint(model, processor, optimizer, root, step, epoch, position)
                break
    atomic_json(root / "status.json", {"status": "completed", "step": step,
                "best": json.loads((root / "checkpoints/best.json").read_text())})


def check_continuation(args):
    """One real 96-decision check, including boundaries 32/64 and another Episode."""
    args.arm = "memory_v1"
    processor, train, _, model = initialize(args)
    eid = next(e for e in sorted(train.by_episode)
               if len(train.by_episode[e]) >= 3 and len(train[train.by_episode[e][2]]["steps"]) == 32)
    sequences = [cache_sequence(model, processor, train, i, args.microbatch)
                 for i in train.by_episode[eid][:3]]
    features = {k: torch.cat([s["features"][k] for s in sequences]) for k in sequences[0]["features"]}
    ids = torch.cat([s["input_ids"] for s in sequences])
    adapter, key = model.memory, "long-sequence-check"
    all_outputs = []
    boundary_state = None
    with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
        for mode in ("stream", "train_sequence"):
            adapter.reset()
            outputs = []
            stride = 1 if mode == "stream" else args.microbatch
            for start in range(0, 96, stride):
                end = min(start+stride, 96)
                if start in (32, 64):
                    assert adapter.cognitive.last_written[key] == start-1
                    assert adapter.cognitive.merge_count[key] == max(0, start-adapter.cognitive.capacity)
                    if mode == "train_sequence" and start == 64:
                        boundary_state = [(list(b.bank[key]), b.last_written[key], b.merge_count[key])
                                          for b in (adapter.perceptual, adapter.cognitive)]
                context = {"mode": mode, "sequence_start": start == 0, "sequence_end": False,
                    "episode_keys": [key]*(end-start), "timesteps": list(range(start, end))}
                out = adapter(slice_features(features, start, end), ids[start:end].cuda(), context)
                outputs.append(out["backbone_features"].cpu())
                if mode == "train_sequence" and end in (32, 64):
                    before = list(adapter.cognitive.bank[key])
                    adapter(slice_features(features, 0, 1), ids[:1].cuda(),
                        {"mode": "train_sequence", "sequence_start": True, "sequence_end": True,
                         "episode_keys": ["other-episode"], "timesteps": [0]})
                    assert all(t == u and x is y for (t, x), (u, y) in zip(before, adapter.cognitive.bank[key]))
                for b in (adapter.perceptual, adapter.cognitive):
                    assert all(not f.requires_grad and f.grad_fn is None for _, f in b.bank[key])
            all_outputs.append(torch.cat(outputs))
            assert adapter.cognitive.merge_count[key] == max(0, 96-adapter.cognitive.capacity)
        torch.testing.assert_close(*all_outputs, rtol=0, atol=0)
    # Backpropagate the first genuine observation after the second window.
    for b, (history, last, merges) in zip((adapter.perceptual, adapter.cognitive), boundary_state):
        b.bank[key], b.last_written[key], b.merge_count[key] = history, last, merges
    model.train()
    model.backbone.eval()
    with torch.autocast("cuda", dtype=torch.bfloat16):
        fused = adapter(slice_features(features, 64, 65), ids[64:65].cuda(),
            {"mode": "train_sequence", "sequence_start": False, "sequence_end": False,
             "episode_keys": [key], "timesteps": [64]})
        adapter.eval()
        empty = adapter(slice_features(features, 64, 65), ids[64:65].cuda(),
            {"mode": "empty", "episode_keys": [key], "timesteps": [64]})
        adapter.train()
        delta = float((fused["backbone_features"]-empty["backbone_features"]).abs().mean())
        loss = model.action_head(fused, slice_features(sequences[-1]["actions"], 0, 1))["loss"]
    loss.backward()
    gradients = gradient_norms(model)
    assert delta > 0
    result = {"passed": True, "real_episode": eid, "decisions": 96, "window": 32,
              "training_stream_max_abs": float((all_outputs[0]-all_outputs[1]).abs().max()),
              "merges_after_96_writes": max(0, 96-adapter.cognitive.capacity), "detached_history": True,
              "interleaved_episode_reset_isolated": True, "current_loss": float(loss),
              "history_feature_delta": delta, "memory_gradients": gradients}
    atomic_json(args.output / "diagnostics/long_sequence_check.json", result)
    print(json.dumps(result), flush=True)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    add_memory_training_options(p)
    for name in ("dataset", "checkpoint", "output"):
        p.add_argument("--"+name, type=Path, required=True)
    p.add_argument("--arm", choices=("baseline", "memory_v1"), default="memory_v1")
    p.add_argument("--check-only", action="store_true")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--steps", type=int, default=10000)
    p.add_argument("--eval-interval", type=int, default=500)
    p.add_argument("--checkpoint-interval", type=int, default=1000)
    p.add_argument("--resume-step", type=int, default=0)
    p.add_argument("--microbatch", type=int, default=4)
    p.add_argument("--learning-rate", type=float, default=1e-5)
    p.add_argument("--min-relative-improvement", type=float, default=.001)
    args = p.parse_args()
    assert args.checkpoint_interval > 0 and args.checkpoint_interval % args.eval_interval == 0
    if args.check_only:
        check_continuation(args)
    else:
        train_worker(args)


if __name__ == "__main__":
    main()
