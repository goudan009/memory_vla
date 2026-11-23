"""Bounded single-Episode pilot; frozen Eagle cache, original flow-matching loss.

No simulation service is contacted. A 32-decision sequence is processed in
chronological microbatches, retaining detached fused history between backwards.
The optimizer steps only after the complete sequence, exactly as for an atomic
group. This diagnostic caches frozen Eagle features once; the formal trainer
recomputes each window after its seeded training image augmentation.
"""

import argparse
from contextlib import contextmanager
import csv
import gc
import hashlib
import json
from pathlib import Path
import random
import time

import numpy as np
import torch
from transformers.feature_extraction_utils import BatchFeature

from v1_groot_memoryvla.audit.train_official_batch import load_real_step
from v1_groot_memoryvla.data.dataset.history_dataset import HistoryDataset
from v1_groot_memoryvla.model.gr00t_n1d6.gr00t_n1d6 import Gr00tN1d6
from v1_groot_memoryvla.model.gr00t_n1d6.processing_gr00t_n1d6 import Gr00tN1d6Processor
from v1_groot_memoryvla.experiment.output import capture_console, create_run_directory


def write_json(path, value):
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n")


def seeded(seed):
    random.seed(seed)
    # Per-window seeds grow with the optimizer step; NumPy accepts only uint32.
    np.random.seed(seed % (2**32))
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def slice_features(features, start, end, device="cuda"):
    def move(value):
        if torch.is_tensor(value):
            return value.to(device)
        if isinstance(value, (list, tuple)):
            return [move(v) for v in value]
        return value
    batch_size = len(features['input_ids']) if 'input_ids' in features else None
    sliced = {}
    for k, v in features.items():
        # Eagle stores one image entry per camera, flattened over the sample batch.
        scale = len(v) // batch_size if k == 'pixel_values' and batch_size else 1
        if k == 'pixel_values' and (not scale or len(v) % batch_size):
            raise ValueError('Image list must contain the same camera count per sample')
        sliced[k] = move(v[start * scale:end * scale])
    return BatchFeature(data=sliced)


@contextmanager
def bank_trace(adapter):
    events, originals = [], []
    for label, bank in (("perceptual", adapter.perceptual), ("cognitive", adapter.cognitive)):
        original = bank.read_fuse_write
        originals.append((bank, original))

        def wrapped(current, key, timestep, original=original, bank=bank, label=label, **kwargs):
            event = {"bank": label, "timestep": timestep,
                     "read_length": len(bank.bank.get(key, [])),
                     "read_timesteps": [t for t, _ in bank.bank.get(key, [])],
                     "prior_merges": bank.merge_count.get(key, 0)}
            assert all(t < timestep for t in event["read_timesteps"])
            result = original(current, key, timestep, **kwargs)
            event["merges_after_write"] = bank.merge_count.get(key, 0)
            events.append(event)
            return result

        bank.read_fuse_write = wrapped
    try:
        yield events
    finally:
        for bank, original in originals:
            bank.read_fuse_write = original


def sequence_context(sequence, start, end):
    return {"mode": "train_sequence", "sequence_start": start == 0,
            "sequence_end": end == len(sequence["timesteps"]),
            "episode_keys": [sequence["episode_key"]] * (end - start),
            "timesteps": sequence["timesteps"][start:end]}


def run_sequence(model, sequence, microbatch, backward=False, noise_seed=None):
    total, count = 0.0, len(sequence["timesteps"])
    for start in range(0, count, microbatch):
        end = min(start + microbatch, count)
        if noise_seed is not None:
            seeded(noise_seed + start)
        features = slice_features(sequence["features"], start, end)
        inputs = slice_features(sequence["actions"], start, end)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            fused = model.memory(features, sequence["input_ids"][start:end].cuda(),
                                 sequence_context(sequence, start, end))
            loss = model.action_head(fused, inputs)["loss"]
        if not torch.isfinite(loss):
            raise RuntimeError("Non-finite flow-matching loss")
        weight = (end - start) / count
        total += float(loss.detach()) * weight
        if backward:
            (loss * weight).backward()
        del features, inputs, fused, loss
    assert not model.memory.perceptual.bank and not model.memory.cognitive.bank
    return total


def gradient_norms(model):
    result = {}
    for name, parameter in model.memory.named_parameters():
        # A one-observation episode has no retrieval, and an exhausted DDP slot
        # may contribute no local gradient. Neither is a disconnected module.
        if parameter.grad is None:
            result[name] = None
            continue
        if not torch.isfinite(parameter.grad).all():
            raise RuntimeError(f"Non-finite memory gradient: {name}")
        result[name] = float(parameter.grad.float().norm())
    return result


def get_training_group(dataset, index, augmentation_seed=None):
    processor = dataset.processor
    if augmentation_seed is not None:
        seeded(augmentation_seed)
        transform = processor.train_image_transform
        if hasattr(transform, "set_random_seed"):
            transform.set_random_seed(augmentation_seed)
    return dataset[index]


def cache_sequence(model, processor, dataset, index, microbatch, augmentation_seed=None, group=None):
    processor = dataset.processor
    if group is None:
        group = get_training_group(dataset, index, augmentation_seed)
    if getattr(model, 'training', False) and any(p.requires_grad for p in model.backbone.parameters()):
        # Cache processed inputs, never detached Eagle outputs, when Eagle learns.
        # The backbone forward/backward must run inside the DDP microbatch graph.
        batch = processor.collator(group['steps'])['inputs']
        backbone, action = model.prepare_input(batch)
        def cpu(value):
            if torch.is_tensor(value):
                return value.detach().cpu()
            if isinstance(value, (list, tuple)):
                return [cpu(v) for v in value]
            return value
        result = {k: v for k, v in group.items() if k != 'steps'}
        result['features'] = {k: cpu(backbone[k]) for k in ('input_ids', 'attention_mask', 'pixel_values')}
        result['actions'] = {k: cpu(action[k]) for k in ('state', 'action', 'action_mask', 'embodiment_id')}
        result['input_ids'] = cpu(backbone['input_ids'])
        return result
    features, action_inputs, ids = [], [], []
    for start in range(0, len(group["steps"]), microbatch):
        batch = processor.collator(group["steps"][start:start + microbatch])["inputs"]
        backbone, action = model.prepare_input(batch)
        with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
            output = model.backbone(backbone)
        features.append({k: v.detach().cpu() for k, v in output.items()})
        action_inputs.append({k: action[k].detach().cpu() for k in
                              ("state", "action", "action_mask", "embodiment_id")})
        ids.append(backbone["input_ids"].cpu())
    result = {k: v for k, v in group.items() if k != "steps"}
    result["features"] = {k: torch.cat([v[k] for v in features]) for k in features[0]}
    result["actions"] = {k: torch.cat([v[k] for v in action_inputs]) for k in action_inputs[0]}
    result["input_ids"] = torch.cat(ids)
    return result


def snapshot(adapter):
    return [(dict((k, [(t, f.detach().clone()) for t, f in v]) for k, v in bank.bank.items()),
             bank.last_written.copy(), bank.merge_count.copy())
            for bank in (adapter.perceptual, adapter.cognitive)]


def restore(adapter, state):
    adapter.reset()
    for bank, (history, last, merges) in zip((adapter.perceptual, adapter.cognitive), state):
        bank.bank.update({k: list(v) for k, v in history.items()})
        bank.last_written.update(last)
        bank.merge_count.update(merges)


def decision(model, sequence, index, key="probe"):
    features = slice_features(sequence["features"], index, index + 1)
    return model.memory(features, sequence["input_ids"][index:index + 1].cuda(), {
        "mode": "stream", "episode_keys": [key], "timesteps": [sequence["timesteps"][index]]})


def predict(model, fused, sequence, index, seed):
    seeded(seed)
    return model.action_head.get_action(fused, slice_features(sequence["actions"], index, index + 1)
                                        )["action_pred"].float().cpu()


def differences(a, b):
    delta = (a[:, :32, :23] - b[:, :32, :23]).float()
    return {"mae_first8": float(delta[:, :8].abs().mean()),
            "mae_horizon32": float(delta.abs().mean()),
            "rms_horizon32": float(delta.square().mean().sqrt()),
            "max_abs_horizon32": float(delta.abs().max())}


@torch.no_grad()
def interventions(model, normal, wrong, seed, output):
    model.eval()
    adapter = model.memory
    arrays, report = {}, {}
    # Swap content in the same temporal slots; otherwise this intervention
    # confounds the learned time encoding with the history's visual content.
    wrong = dict(wrong, timesteps=normal["timesteps"])
    report["history_age_matched"] = True
    with torch.autocast("cuda", dtype=torch.bfloat16):
        adapter.reset()
        cold = decision(model, normal, 0)
        raw = slice_features(normal["features"], 0, 1)
        assert torch.equal(cold.backbone_features, raw.backbone_features)
        empty = predict(model, cold, normal, 0, seed)
        plain = predict(model, raw, normal, 0, seed)
        assert torch.equal(empty, plain)
        report["empty_history_actions_exact"] = True
        adapter.reset()
        with bank_trace(adapter) as events:
            grouped = []
            for start in range(0, 32, 4):
                grouped.append(adapter(slice_features(normal["features"], start, start + 4),
                    normal["input_ids"][start:start + 4].cuda(),
                    sequence_context(normal, start, start + 4)).backbone_features.cpu())
        adapter.reset()
        online = []
        for i in range(32):
            online.append(decision(model, normal, i).backbone_features.cpu())
        assert torch.equal(torch.cat(grouped), torch.cat(online))
        report["chunked_training_online_32_exact"] = True
        report["causal_read_trace"] = events
        for bank in (adapter.perceptual, adapter.cognitive):
            assert len(bank.bank["probe"]) == 16 and bank.merge_count["probe"] == 16
        report["writes32_merges16_per_bank"] = True
        adapter.reset()
        # Compare normal history with wrong, strictly earlier observations.
        target = 31
        for i in range(target):
            decision(model, normal, i)
        normal_bank = snapshot(adapter)
        for variant in ("normal", "empty", "wrong"):
            if variant == "normal":
                restore(adapter, normal_bank)
            else:
                adapter.reset()
                if variant == "wrong":
                    for i in range(target):
                        decision(model, wrong, i)
            fused = decision(model, normal, target)
            arrays[variant] = predict(model, fused, normal, target, seed).numpy()
        report["normal_vs_wrong"] = differences(torch.from_numpy(arrays["normal"]),
                                                 torch.from_numpy(arrays["wrong"]))
        report["normal_vs_empty"] = differences(torch.from_numpy(arrays["normal"]),
                                                 torch.from_numpy(arrays["empty"]))
        label = normal["actions"]["action"][target:target + 1].float()
        report["normalized_action_mae_to_label"] = {
            name: differences(torch.from_numpy(pred), label) for name, pred in arrays.items()}
        report["wrong_history_source"] = {"group_id": wrong["group_id"],
            "frames": wrong["frame_indices"][:target], "same_task_other_earlier_segment": True}
        assert max(wrong["frame_indices"][:target]) < normal["frame_indices"][0]
        # First merge happens on decision 17. Remove only the averaged entry,
        # then repeat decision 18 with identical current input and action noise.
        adapter.reset()
        for i in range(16):
            decision(model, normal, i)
        before = snapshot(adapter)
        decision(model, normal, 16)
        after = snapshot(adapter)
        merged_times = []
        for bank, (old, _, _), (new, _, _) in zip(
                (adapter.perceptual, adapter.cognitive), before, after):
            old_by_t = dict(old["probe"])
            changed = [t for t, f in new["probe"] if t in old_by_t and
                       not torch.equal(old_by_t[t], f)]
            assert len(changed) == 1
            merged_times.append(changed[0])
            assert bank.merge_count["probe"] == 1
        for variant in ("merged", "merged_removed"):
            restore(adapter, after)
            if variant == "merged_removed":
                for bank, t in zip((adapter.perceptual, adapter.cognitive), merged_times):
                    bank.bank["probe"] = [(s, f) for s, f in bank.bank["probe"] if s != t]
            fused = decision(model, normal, 17)
            arrays[variant] = predict(model, fused, normal, 17, seed).numpy()
        report["merged_entry_intervention"] = differences(torch.from_numpy(arrays["merged"]),
                                                           torch.from_numpy(arrays["merged_removed"]))
        report["merged_entry_retained_timestamps"] = merged_times
        assert report["merged_entry_intervention"]["max_abs_horizon32"] > 0
        report["merged_memory_changes_subsequent_action"] = True
        # Strictly later current observations cannot affect earlier decisions.
        adapter.reset()
    np.savez_compressed(output / "history_actions.npz", **arrays,
                        target_actions=label.numpy())
    return report


def curves(output, records):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    train = [r for r in records if r["kind"] == "train"]
    evaluate = [r for r in records if r["kind"] == "eval"]
    axes[0, 0].plot([r["step"] for r in train], [r["loss"] for r in train], label="train stochastic")
    for key in ("train_fixed_loss", "validation_loss"):
        axes[0, 0].plot([r["step"] for r in evaluate], [r[key] for r in evaluate], "o-", label=key)
    axes[0, 0].set(ylabel="valid 32x23 flow-matching MSE", xlabel="optimizer step")
    axes[0, 0].legend()
    for key in ("perceptual_residual.weight", "cognitive.retrieval_blocks.0.q_proj.weight",
                "perceptual.retrieval_blocks.0.q_proj.weight"):
        axes[0, 1].plot([r["step"] for r in train],
                        [r["memory_gradients"][key] for r in train], label=key)
    axes[0, 1].set(yscale="log", ylabel="gradient L2 before clipping", xlabel="optimizer step")
    axes[0, 1].legend(fontsize=6)
    axes[1, 0].plot([r["step"] for r in train], [r["decisions_per_second"] for r in train])
    axes[1, 0].set(ylabel="cached training decisions/s", xlabel="optimizer step")
    axes[1, 1].plot([r["step"] for r in train], [r["peak_allocated_gib"] for r in train], label="allocated")
    axes[1, 1].plot([r["step"] for r in train], [r["peak_reserved_gib"] for r in train], label="reserved")
    axes[1, 1].set(ylabel="process CUDA GiB", xlabel="optimizer step")
    axes[1, 1].legend()
    fig.tight_layout()
    fig.savefig(output / "training_curves.png", dpi=160)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, default=Path("checkpoints/groot_n1d6_behavior1k"))
    parser.add_argument("--dataset", type=Path, default=Path("assets/datasets/b1k2025_groot_v1_smoke"))
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--steps", type=int, default=40)
    parser.add_argument("--sequence-length", type=int, default=32)
    parser.add_argument("--microbatch", type=int, default=2)
    parser.add_argument("--learning-rate", type=float, default=1e-5)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    assert args.sequence_length == 32, "Pilot data split and intervention probes use 32 decisions"
    assert args.steps > 0 and 32 % args.microbatch == 0
    create_run_directory(args.output)
    with capture_console(args.output / "logs/train.log"):
        run(args)


def run(args):
    records, started = [], time.monotonic()
    seeded(args.seed)
    torch.set_num_threads(8)
    config = {k: str(v.resolve()) if isinstance(v, Path) else v for k, v in vars(args).items()}
    config.update({"train_group_indices": [0, 20, 40, 60], "validation_group_indices": [100, 120],
                   "memory_capacity": 16, "group_size_legacy": 16, "history_stride": 8,
                   "retrieval_layers": 2, "fusion": "gate", "consolidation": "tome", "update_fused": True,
                   "optimizer": "AdamW", "weight_decay": 0.01, "clip_grad_norm": 1.0,
                   "warmup": 0, "lr_schedule": "constant", "precision": "FP32 trainable + BF16 autocast",
                   "frozen_eagle_cache": True, "image_augmentation": False,
                   "language_protocol": "B1K task_name.replace('_',' ').capitalize()+'.'",
                   "gpu": torch.cuda.get_device_name(), "torch_version": torch.__version__,
                   "gpu_uuid": str(torch.cuda.get_device_properties(0).uuid)})
    write_json(args.output / "config/run_config.json", config)
    processor = Gr00tN1d6Processor.from_pretrained(args.checkpoint)
    processor.eval()
    dataset = HistoryDataset([args.dataset], processor, group_size=32)
    step = load_real_step(dataset.loaders[0], 210020, 0)
    full_instruction = step.text
    canonical = processor.canonical_language(full_instruction)
    assert canonical == processor.canonical_language("Collecting childrens toys.")
    # Test actual train/inference tokenizer content without random image crops.
    step.text = full_instruction
    a = processor.collator([processor([{"type": "episode_step", "content": step}])])["inputs"]
    step.text = "Collecting childrens toys."
    b = processor.collator([processor([{"type": "episode_step", "content": step}])])["inputs"]
    for key in ("input_ids", "attention_mask"):
        assert torch.equal(a[key], b[key])
    write_json(args.output / "diagnostics/language_contract.json", {"source_instruction": full_instruction,
        "online_instruction": step.text, "canonical_text": canonical,
        "input_ids_equal": True, "attention_mask_equal": True,
        "input_ids": a["input_ids"].tolist()})
    model, loading = Gr00tN1d6.from_pretrained(args.checkpoint, memory_enabled=True,
        memory_train_sequence_length=32, tune_llm=False, tune_visual=False, tune_top_llm_layers=0,
        backbone_trainable_params_fp32=False, torch_dtype=torch.float32, output_loading_info=True)
    assert loading["missing_keys"] and all(k.startswith("memory.") for k in loading["missing_keys"])
    assert not loading["unexpected_keys"] and not loading.get("mismatched_keys")
    model.backbone.requires_grad_(False).bfloat16()
    model.cuda().eval()
    cache_start = time.monotonic()
    train, validation = [], []
    for indices, target in ((config["train_group_indices"], train), (config["validation_group_indices"], validation)):
        for index in indices:
            target.append(cache_sequence(model, processor, dataset, index, args.microbatch))
            print("CACHED", index, target[-1]["frame_indices"][0], flush=True)
    torch.cuda.synchronize()
    cache_seconds = time.monotonic() - cache_start
    assert all(s["episode_key"].endswith("episode210020") for s in train + validation)
    assert max(s["frame_indices"][-1] + 31 for s in train) < min(s["frame_indices"][0] for s in validation)
    split = {name: [{k: v for k, v in s.items() if k not in ("features", "actions", "input_ids")}
                    for s in seqs] for name, seqs in (("train", train), ("validation", validation))}
    write_json(args.output / "config/data_split.json", split)
    torch.save({"train": train, "validation": validation}, args.output / "cache/frozen_eagle_cache.pt")
    # Release frozen weights from the shared GPU once real features are cached.
    model.backbone.cpu()
    gc.collect()
    torch.cuda.empty_cache()
    parameters = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(parameters, lr=args.learning_rate, weight_decay=0.01, foreach=False)
    config["trainable_parameters"] = sum(p.numel() for p in parameters)
    config["frozen_feature_cache_seconds"] = cache_seconds
    config["feature_extraction_decisions_per_second"] = 192 / cache_seconds
    config["official_weight_index_sha256"] = hashlib.sha256(
        (args.checkpoint / "model.safetensors.index.json").read_bytes()).hexdigest()
    write_json(args.output / "config/run_config.json", config)

    def log(record):
        records.append(record)
        with (args.output / "metrics/metrics.jsonl").open("a") as file:
            file.write(json.dumps(record) + "\n")
        print(json.dumps({k: v for k, v in record.items() if k != "memory_gradients"}), flush=True)

    def evaluate(step):
        model.eval()
        with torch.no_grad():
            train_loss = np.mean([run_sequence(model, s, args.microbatch, noise_seed=1000 + i * 100)
                                  for i, s in enumerate(train)])
            val_loss = np.mean([run_sequence(model, s, args.microbatch, noise_seed=3000 + i * 100)
                                for i, s in enumerate(validation)])
        log({"kind": "eval", "step": step, "train_fixed_loss": float(train_loss),
             "validation_loss": float(val_loss)})
        curves(args.output / "plots", records)

    evaluate(0)
    before_dir = args.output / "diagnostics/history_before"
    before_dir.mkdir()
    before_interventions = interventions(model, validation[0], train[0], 909, before_dir)
    write_json(before_dir / "result.json", before_interventions)
    for index in range(args.steps):
        model.train()
        model.backbone.eval()
        optimizer.zero_grad(set_to_none=True)
        torch.cuda.reset_peak_memory_stats()
        torch.cuda.synchronize()
        tick = time.monotonic()
        seeded(args.seed + index * 100)
        with bank_trace(model.memory) as trace:
            loss = run_sequence(model, train[index % len(train)], args.microbatch, backward=True)
        grads = gradient_norms(model)
        norm = torch.nn.utils.clip_grad_norm_(parameters, 1.0)
        assert torch.isfinite(norm)
        optimizer.step()
        torch.cuda.synchronize()
        seconds = time.monotonic() - tick
        for label in ("perceptual", "cognitive"):
            events = [e for e in trace if e["bank"] == label]
            assert events[16]["merges_after_write"] == 1 and events[17]["prior_merges"] == 1
            assert events[-1]["merges_after_write"] == 16
        log({"kind": "train", "step": index + 1, "loss": loss, "seconds": seconds,
             "decisions_per_second": 32 / seconds, "gradient_norm_before_clip": float(norm),
             "memory_gradients": grads, "merges_per_bank": 16, "merged_history_reads_per_bank": 15,
             "peak_allocated_gib": torch.cuda.max_memory_allocated() / 2**30,
             "peak_reserved_gib": torch.cuda.max_memory_reserved() / 2**30})
        if index == 0:
            write_json(args.output / "diagnostics/first_sequence_trace.json", trace)
        if (index + 1) % 5 == 0 or index + 1 == args.steps:
            evaluate(index + 1)
    optimizer.zero_grad(set_to_none=True)
    model.eval()
    after_dir = args.output / "diagnostics/history_after"
    after_dir.mkdir()
    after_interventions = interventions(model, validation[0], train[0], 909, after_dir)
    write_json(after_dir / "result.json", after_interventions)
    del optimizer, parameters
    gc.collect()
    model.memory.reset()
    checkpoint = args.output / "checkpoints/final"
    model.save_pretrained(checkpoint, max_shard_size="5GB")
    processor.save_pretrained(checkpoint)
    # Compare true sequential actions after reloading every learned parameter.
    with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
        for i in range(18):
            fused = decision(model, validation[0], i)
        expected_action = predict(model, fused, validation[0], 17, 222)
    del model, fused
    gc.collect()
    torch.cuda.empty_cache()
    reloaded, info = Gr00tN1d6.from_pretrained(checkpoint, torch_dtype=torch.float32,
        backbone_trainable_params_fp32=False, output_loading_info=True)
    assert not info["missing_keys"] and not info["unexpected_keys"]
    assert not reloaded.memory.cognitive.bank
    reloaded.memory.cuda().eval()
    reloaded.action_head.cuda().eval()
    reloaded.eval()
    with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
        for i in range(18):
            fused = decision(reloaded, validation[0], i)
        actual_action = predict(reloaded, fused, validation[0], 17, 222)
    assert torch.equal(expected_action, actual_action)
    loaded_processor = Gr00tN1d6Processor.from_pretrained(checkpoint)
    assert loaded_processor.canonical_language(full_instruction) == canonical
    assert loaded_processor.canonical_language("Collecting childrens toys.") == canonical
    evals = [r for r in records if r["kind"] == "eval"]
    trains = [r for r in records if r["kind"] == "train"]
    with (args.output / "metrics/evaluation.csv").open("w") as file:
        writer = csv.DictWriter(file, fieldnames=list(evals[0]))
        writer.writeheader()
        writer.writerows(evals)
    report = {"status": "completed", "optimizer_steps": args.steps,
              "initial_eval": evals[0], "final_eval": evals[-1],
              "best_validation": min(evals, key=lambda r: r["validation_loss"]),
              "mean_cached_training_decisions_per_second": float(np.mean([r["decisions_per_second"] for r in trains])),
              "peak_training_allocated_gib": max(r["peak_allocated_gib"] for r in trains),
              "peak_training_reserved_gib": max(r["peak_reserved_gib"] for r in trains),
              "reload_merged_history_actions_exact": True, "reload_language_mapping_exact": True,
              "history_intervention": after_interventions["normal_vs_wrong"],
              "merged_entry_intervention": after_interventions["merged_entry_intervention"],
              "elapsed_seconds": time.monotonic() - started,
              "limitations": ["one Episode / same task temporal holdout; no closed-loop success rate",
                  "one training seed; action changes do not establish a repeatable memory benefit",
                  "throughput excludes precomputed frozen Eagle and uses no image augmentation",
                  "wrong history is from a different earlier segment of the same Episode",
                  "only final checkpoint saved; best validation step is diagnostic"]}
    write_json(args.output / "report.json", report)
    print(json.dumps(report, indent=2), flush=True)


if __name__ == "__main__":
    main()
