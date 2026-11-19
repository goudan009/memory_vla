"""Paired 200--400 step Task 21 diagnosis, with Episode-level interventions.

The frozen official Eagle is evaluated once. Both arms use exactly those
features, trainable action-head initialization, labels, order, random numbers,
and optimization schedule. No running policy or simulation is contacted.
"""

import argparse
import csv
import gc
import hashlib
import json
from pathlib import Path
import time

import numpy as np
import torch
from transformers.feature_extraction_utils import BatchFeature

from .first_training import (bank_trace, cache_sequence, gradient_norms, seeded,
                             sequence_context, slice_features, snapshot, restore, write_json)
from .output import capture_console, create_run_directory
from v3_bodymemory.data.dataset.history_dataset import HistoryDataset
from v3_bodymemory.model.gr00t_n1d6.gr00t_n1d6 import Gr00tN1d6
from v3_bodymemory.model.gr00t_n1d6.processing_gr00t_n1d6 import Gr00tN1d6Processor


def head_hash(model):
    h = hashlib.sha256()
    for name, p in model.action_head.named_parameters():
        h.update(name.encode())
        h.update(p.detach().cpu().contiguous().numpy().tobytes())
    return h.hexdigest()


def load_model(checkpoint, memory, seed, **memory_options):
    seeded(seed)
    model, info = Gr00tN1d6.from_pretrained(checkpoint, memory_enabled=memory,
        memory_train_sequence_length=32, tune_llm=False, tune_visual=False, tune_top_llm_layers=0,
        backbone_trainable_params_fp32=False, torch_dtype=torch.float32, output_loading_info=True,
        **memory_options)
    assert not info["unexpected_keys"] and not info.get("mismatched_keys")
    assert all(k.startswith("memory.") for k in info["missing_keys"])
    model.backbone.requires_grad_(False).bfloat16().eval()
    model.action_head.cuda()
    if memory:
        model.memory.cuda()
    model.eval()
    return model


def sequence_loss(model, seq, seed, backward=False, microbatch=2):
    total = 0.
    for start in range(0, 32, microbatch):
        end = start + microbatch
        features = slice_features(seq["features"], start, end)
        inputs = slice_features(seq["actions"], start, end)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            if model.memory is not None:
                features = model.memory(features, seq["input_ids"][start:end].cuda(),
                                        sequence_context(seq, start, end))
            # Reseed immediately before the shared action head. Additional memory
            # operations must not change action noise, state dropout or DiT RNG.
            seeded(seed + start)
            loss = model.action_head(features, inputs)["loss"]
        assert torch.isfinite(loss)
        total += float(loss.detach()) * microbatch / 32
        if backward:
            (loss * microbatch / 32).backward()
    if model.memory is not None:
        assert not model.memory.perceptual.bank and not model.memory.cognitive.bank
    return total


def append_record(path, record):
    with path.open("a") as f:
        f.write(json.dumps(record) + "\n")


def average_episodes(rows, key):
    ids = sorted({r["episode"] for r in rows})
    return float(np.mean([np.mean([r[key] for r in rows if r["episode"] == eid]) for eid in ids]))


@torch.no_grad()
def evaluate_loss(model, cache, step, root, arm):
    model.eval()
    rows = []
    # One fixed uniform train sequence per Episode is a training diagnostic;
    # every validation sequence is evaluated, preserving Episode/stratum keys.
    train_ids = {s["episode"] for s in cache["train"]}
    chosen = [next(s for s in cache["train"] if s["episode"] == e and s["stratum"] == "uniform")
              for e in sorted(train_ids)] + cache["validation"]
    for s in chosen:
        loss = sequence_loss(model, s, 100000 + s["group_id"] * 100)
        row = {"arm": arm, "step": step, "episode": s["episode"], "group_id": s["group_id"],
               "split": s["split"], "stratum": s["stratum"], "loss": loss}
        rows.append(row)
        append_record(root / "metrics/validation_groups.jsonl", row)
    train = [r for r in rows if r["split"] == "train"]
    val = [r for r in rows if r["split"] == "validation"]
    uniform = [r for r in val if r["stratum"] == "uniform"]
    by_episode = {str(e): float(np.mean([r["loss"] for r in uniform if r["episode"] == e]))
                  for e in sorted({r["episode"] for r in val})}
    result = {"kind": "eval", "arm": arm, "step": step,
              "train_fixed_loss": average_episodes(train, "loss"),
              "validation_loss": average_episodes(uniform, "loss"),
              "validation_panel_loss": average_episodes(val, "loss"),
              "validation_by_episode": by_episode}
    append_record(root / "metrics/metrics.jsonl", result)
    print(json.dumps(result), flush=True)
    return result


def train_arm(args, cache, processor, arm, start, end):
    root = args.output / arm
    memory = arm == "memory_v1"
    if start == 0:
        root.mkdir()
        for name in ("config", "metrics", "checkpoints", "diagnostics"):
            (root / name).mkdir()
    checkpoint = args.checkpoint if start == 0 else root / "checkpoints/final"
    model = load_model(checkpoint, memory, args.seed)
    if start == 0:
        initial = {"action_head_sha256": head_hash(model),
                   "head_trainable": {n: p.requires_grad for n, p in model.action_head.named_parameters()},
                   "trainable_parameters": sum(p.numel() for p in model.parameters() if p.requires_grad),
                   "frozen_backbone": not any(p.requires_grad for p in model.backbone.parameters())}
        write_json(root / "config/initialization.json", initial)
        expected = json.loads((args.output / "diagnostics/cache.json").read_text())
        assert initial["action_head_sha256"] == expected["initial_action_head_sha256"]
    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(params, lr=args.learning_rate, weight_decay=.01, foreach=False)
    if start:
        state = torch.load(root / "checkpoints/training_state.pt", map_location="cpu", weights_only=False)
        optimizer.load_state_dict(state["optimizer"])
        assert state["completed_steps"] == start
        del state
    else:
        evaluate_loss(model, cache, 0, root, arm)
    order = json.loads((args.output / "config/optimization_order.json").read_text())
    for index in range(start, end):
        model.train()
        model.backbone.eval()
        optimizer.zero_grad(set_to_none=True)
        torch.cuda.reset_peak_memory_stats()
        torch.cuda.synchronize()
        tick = time.monotonic()
        seq = cache["train"][order[index]]
        if memory:
            with bank_trace(model.memory) as trace:
                loss = sequence_loss(model, seq, args.seed + 1000 * index, backward=True)
            gradients = gradient_norms(model)
            for label in ("perceptual", "cognitive"):
                events = [e for e in trace if e["bank"] == label]
                assert events[16]["merges_after_write"] == 1 and events[17]["prior_merges"] == 1
                assert events[-1]["merges_after_write"] == 16
            if index == 0:
                write_json(root / "diagnostics/first_sequence_trace.json", trace)
        else:
            loss = sequence_loss(model, seq, args.seed + 1000 * index, backward=True)
            gradients = {}
        norm = torch.nn.utils.clip_grad_norm_(params, 1.)
        assert torch.isfinite(norm)
        optimizer.step()
        torch.cuda.synchronize()
        seconds = time.monotonic() - tick
        record = {"kind": "train", "arm": arm, "step": index + 1, "loss": loss,
                  "group_id": seq["group_id"], "episode": seq["episode"],
                  "seconds": seconds, "decisions_per_second": 32 / seconds,
                  "gradient_norm_before_clip": float(norm), "memory_gradients": gradients,
                  "merges_per_bank": 16 if memory else 0,
                  "peak_allocated_gib": torch.cuda.max_memory_allocated() / 2**30,
                  "peak_reserved_gib": torch.cuda.max_memory_reserved() / 2**30}
        append_record(root / "metrics/metrics.jsonl", record)
        print(json.dumps({k: v for k, v in record.items() if k != "memory_gradients"}), flush=True)
        if (index + 1) % args.eval_interval == 0 or index + 1 == end:
            evaluate_loss(model, cache, index + 1, root, arm)
    optimizer.zero_grad(set_to_none=True)
    torch.save({"optimizer": optimizer.state_dict(), "completed_steps": end},
               root / "checkpoints/training_state.pt")
    del optimizer, params
    gc.collect()
    model.eval()
    if memory:
        model.memory.reset()
    model.save_pretrained(root / "checkpoints/final", max_shard_size="5GB")
    processor.save_pretrained(root / "checkpoints/final")
    del model
    gc.collect()
    torch.cuda.empty_cache()


def raw_features(seq, i):
    return slice_features(seq["features"], i, i+1)


def stream(model, seq, i, timestep):
    return model.memory(raw_features(seq, i), seq["input_ids"][i:i+1].cuda(),
                        {"mode": "stream", "episode_keys": ["probe"], "timesteps": [timestep]})


def cpu_feature(f):
    return {k: v.detach().cpu() for k, v in f.items() if k != "visual_features"}


def join_features(features):
    return {k: torch.cat([f[k] for f in features]) for k in features[0]}


def match_history_ages(adapter, reference_times):
    for bank, times in zip((adapter.perceptual, adapter.cognitive), reference_times):
        entries = bank.bank.get("probe", [])
        assert len(entries) == len(times)
        bank.bank["probe"] = [(t, f) for t, (_, f) in zip(times, entries)]
        assert [t for t, _ in bank.bank["probe"]] == times


@torch.no_grad()
def intervention_features(model, seq, donor):
    assert donor["episode"] != seq["episode"] and donor["split"] == "train"
    adapter = model.memory
    adapter.reset()
    normal, ages = [], []
    with torch.autocast("cuda", dtype=torch.bfloat16):
        for i, t in enumerate(seq["timesteps"]):
            ages.append([[s for s, _ in b.bank.get("probe", [])]
                         for b in (adapter.perceptual, adapter.cognitive)])
            normal.append(cpu_feature(stream(model, seq, i, t)))
        adapter.reset()
        wrong = []
        for i, t in enumerate(seq["timesteps"]):
            previous = snapshot(adapter)
            # ToMe may merge different adjacent pairs for a different trajectory.
            # Align the *post-consolidation read slots* as well as input cadence.
            match_history_ages(adapter, ages[i])
            wrong.append(cpu_feature(stream(model, seq, i, t)))
            restore(adapter, previous)
            stream(model, donor, i, t)
        adapter.reset()
    normal, wrong = join_features(normal), join_features(wrong)
    empty = {k: v for k, v in seq["features"].items() if k != "visual_features"}
    assert torch.equal(normal["backbone_features"][:1], empty["backbone_features"][:1])
    assert torch.equal(wrong["backbone_features"][:1], empty["backbone_features"][:1])
    for v in (normal, wrong):
        assert v["backbone_features"].shape == empty["backbone_features"].shape
        for k in ("image_mask", "backbone_attention_mask"):
            assert torch.equal(v[k], empty[k])
    return {"normal": normal, "empty": empty, "wrong": wrong}


@torch.no_grad()
def action_predictions(model, features, seq, seed):
    preds = []
    for start in range(0, len(seq["actions"]["action"]), 2):
        seeded(seed + start)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            out = model.action_head.get_action(slice_features(features, start, start+2),
                 slice_features(seq["actions"], start, start+2))["action_pred"]
        preds.append(out[:, :32, :23].float().cpu().numpy())
    return np.concatenate(preds)


def write_csv(path, rows):
    with path.open("w") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


@torch.no_grad()
def action_evaluation(args, cache, arm):
    root = args.output / arm
    model = load_model(root / "checkpoints/final", arm == "memory_v1", args.seed)
    observations, shifts = [], []
    for seq in cache["validation"]:
        # Donor matches approximate relative task time, from another training
        # Episode; it never contains the target Episode or its future frames.
        fraction = seq["start"] / seq["episode_length"]
        donors = [d for d in cache["train"] if d["stratum"] == "uniform"]
        donor = min(donors, key=lambda d: abs(d["start"] / d["episode_length"] - fraction))
        features = (intervention_features(model, seq, donor) if model.memory is not None else
                    {"normal": {k: v for k, v in seq["features"].items() if k != "visual_features"}})
        arrays = {}
        for noise_seed in (909, 1234, 2345):
            preds = {name: action_predictions(model, feat, seq, noise_seed + seq["group_id"] * 100)
                     for name, feat in features.items()}
            if model.memory is None:
                preds.update({"empty": preds["normal"], "wrong": preds["normal"]})
            for name, pred in preds.items():
                arrays[f"{name}_seed{noise_seed}"] = pred
                target = seq["actions"]["action"][:, :32, :23].float().numpy()
                error = pred - target
                for i in range(32):
                    observations.append({"arm": arm, "condition": name, "episode": seq["episode"],
                        "group_id": seq["group_id"], "frame": seq["frame_indices"][i],
                        "decision_index": i, "stratum": seq["stratum"],
                        "candidate": seq["candidate_mask"][i], "noise_seed": noise_seed,
                        "mae_first8": float(np.abs(error[i, :8]).mean()),
                        "mae_horizon32": float(np.abs(error[i]).mean()),
                        "mse_horizon32": float(np.square(error[i]).mean())})
            if model.memory is not None:
                assert np.array_equal(preds["normal"][:1], preds["empty"][:1])
            shifts.append({"episode": seq["episode"], "group_id": seq["group_id"], "noise_seed": noise_seed,
                           "normal_empty_mae": float(np.abs(preds["normal"]-preds["empty"]).mean()),
                           "normal_wrong_mae": float(np.abs(preds["normal"]-preds["wrong"]).mean())})
        arrays["target"] = target
        arrays["frames"] = np.asarray(seq["frame_indices"])
        np.savez_compressed(root / f"diagnostics/actions_group{seq['group_id']}.npz", **arrays)
        append_record(root / "metrics/intervention_donors.jsonl", {"group_id": seq["group_id"],
            "episode": seq["episode"], "donor_group": donor["group_id"], "donor_episode": donor["episode"],
            "read_ages_matched_after_consolidation": True, "noise_seeds": [909, 1234, 2345]})
        print("ACTIONS", arm, seq["episode"], seq["group_id"], flush=True)
    write_csv(root / "metrics/action_observations.csv", observations)
    write_csv(root / "metrics/action_shifts.csv", shifts)
    aggregated = []
    for eid in sorted({r["episode"] for r in observations}):
        for condition in ("normal", "empty", "wrong"):
            for subset in ("uniform", "panel", "history_candidates", "post_merge"):
                selected = [r for r in observations if r["episode"] == eid and r["condition"] == condition
                            and (subset == "panel" or (subset == "uniform" and r["stratum"] == "uniform")
                                 or (subset == "history_candidates" and r["candidate"])
                                 or (subset == "post_merge" and r["decision_index"] >= 17))]
                if not selected:
                    continue
                # Equal noise repeats per frame; average within each Episode.
                aggregated.append({"arm": arm, "episode": eid, "condition": condition, "subset": subset,
                    "unique_observations": len({(r["group_id"], r["frame"]) for r in selected}),
                    "action_noise_seeds": 3,
                    "mae_first8": float(np.mean([r["mae_first8"] for r in selected])),
                    "mae_horizon32": float(np.mean([r["mae_horizon32"] for r in selected])),
                    "rmse_horizon32": float(np.sqrt(np.mean([r["mse_horizon32"] for r in selected])))})
    write_csv(root / "metrics/action_by_episode.csv", aggregated)
    del model
    gc.collect()
    torch.cuda.empty_cache()
    return aggregated


def records(root):
    return [json.loads(line) for line in (root / "metrics/metrics.jsonl").read_text().splitlines()]


def extend_decision(output):
    evidence = {}
    for arm in ("baseline", "memory_v1"):
        curve = [r for r in records(output / arm) if r["kind"] == "eval"]
        final = curve[-1]
        previous = next(r for r in curve if r["step"] == final["step"] - 50)
        gain = (previous["validation_loss"] - final["validation_loss"]) / previous["validation_loss"]
        evidence[arm] = {"last50_relative_validation_improvement": gain,
                         "final_validation_loss": final["validation_loss"]}
    return {"extend": any(e["last50_relative_validation_improvement"] > .01 for e in evidence.values()),
            "rule": "extend both equally once to 400 if either uniform held-out Episode mean improves >1% over steps150--200",
            "evidence": evidence}


def plot_results(output, aggregated):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(2, 2, figsize=(13, 8))
    for arm in ("baseline", "memory_v1"):
        rec = records(output / arm)
        ev = [r for r in rec if r["kind"] == "eval"]
        tr = [r for r in rec if r["kind"] == "train"]
        axes[0, 0].plot([r["step"] for r in ev], [r["validation_loss"] for r in ev], "o-", label=arm+" validation")
        axes[0, 0].plot([r["step"] for r in ev], [r["train_fixed_loss"] for r in ev], "--", label=arm+" train fixed")
        axes[1, 0].plot([r["step"] for r in tr], [r["decisions_per_second"] for r in tr], label=arm)
        for eid in ev[0]["validation_by_episode"]:
            axes[0, 1].plot([r["step"] for r in ev], [r["validation_by_episode"][eid] for r in ev], label=arm+" ep"+eid)
    labels = ["baseline", "normal", "empty", "wrong"]
    for j, subset in enumerate(("uniform", "history_candidates")):
        heights = []
        for label in labels:
            arm, cond = ("baseline", "normal") if label == "baseline" else ("memory_v1", label)
            selected = [r for r in aggregated if r["arm"] == arm and r["condition"] == cond and r["subset"] == subset]
            heights.append(float(np.mean([r["mae_horizon32"] for r in selected])))
        axes[1, 1].bar(np.arange(4) + j*.35, heights, width=.35, label=subset)
    axes[1, 1].set_xticks(np.arange(4)+.175, labels)
    axes[0, 0].set(ylabel="flow-matching validation/train loss", xlabel="optimizer steps")
    axes[0, 1].set(ylabel="per-Episode validation loss", xlabel="optimizer steps")
    axes[1, 0].set(ylabel="cached decisions / second", xlabel="optimizer steps")
    axes[1, 1].set(ylabel="normalized action MAE (Episode mean)")
    for ax in axes.flat:
        ax.legend(fontsize=6)
    fig.tight_layout()
    fig.savefig(output / "plots/paired_diagnostic.png", dpi=160)
    plt.close(fig)


def run(args):
    torch.set_num_threads(8)
    split = json.loads((args.output / "config/data_split.json").read_text())
    assert not set(split["train_episodes"]) & set(split["validation_episodes"])
    processor = Gr00tN1d6Processor.from_pretrained(args.checkpoint)
    processor.eval()
    dataset = HistoryDataset([args.dataset], processor, group_size=32)
    dataset.groups = [(0, g["episode"], g["start"]) for g in split["groups"]]
    # Persist the immutable training and random-noise schedule before loading models.
    rng = np.random.default_rng(args.seed)
    ntrain = sum(g["split"] == "train" for g in split["groups"])
    order = []
    while len(order) < args.max_steps:
        order.extend(rng.permutation(ntrain).tolist())
    write_json(args.output / "config/optimization_order.json", order[:args.max_steps])
    config = {k: str(v.resolve()) if isinstance(v, Path) else v for k, v in vars(args).items()}
    config.update({"optimizer": "AdamW", "weight_decay": .01, "clip_grad_norm": 1.,
        "lr_schedule": "constant", "warmup": 0, "microbatch": 2, "decisions_per_step": 32,
        "history_stride": 8, "capacity": 16, "retrieval_layers": 2, "update_fused": True,
        "fusion_type": "gate", "consolidate_type": "tome", "frozen_eagle": True,
        "image_augmentation": False, "trainable_precision": "FP32, BF16 autocast",
        "normalization": "official checkpoint, unchanged", "noise_seeds": [909, 1234, 2345],
        "sampling_unit": "Episode; action noise seeds are within-Episode repeats",
        "gpu": torch.cuda.get_device_name(), "gpu_uuid": str(torch.cuda.get_device_properties(0).uuid)})
    write_json(args.output / "config/run_config.json", config)
    cachepath = args.output / "cache/frozen_eagle_cache.pt"
    if not cachepath.exists():
        model = load_model(args.checkpoint, True, args.seed)
        initial_hash = head_hash(model)
        model.backbone.cuda()
        cache = {"train": [], "validation": []}
        tick = time.monotonic()
        for i, group in enumerate(split["groups"]):
            seq = cache_sequence(model, processor, dataset, i, 2)
            seq.update(group)
            seq["episode_length"] = next(e["length"] for e in dataset.loaders[0].episodes_metadata
                                          if e["episode_index"] == group["episode"])
            cache[group["split"]].append(seq)
            print("CACHED", i, group["episode"], group["start"], flush=True)
        torch.save(cache, cachepath)
        write_json(args.output / "diagnostics/cache.json", {"seconds": time.monotonic()-tick,
            "decisions": 32*len(split["groups"]), "real_official_eagle": True,
            "initial_action_head_sha256": initial_hash,
            "input_keys": list(cache["train"][0]["actions"]), "privileged_task_info_in_inputs": False})
        del model
        gc.collect()
        torch.cuda.empty_cache()
    else:
        cache = torch.load(cachepath, map_location="cpu", weights_only=False)
    for arm in ("baseline", "memory_v1"):
        with capture_console(args.output / f"logs/{arm}.log"):
            train_arm(args, cache, processor, arm, 0, args.steps)
    a = json.loads((args.output / "baseline/config/initialization.json").read_text())
    b = json.loads((args.output / "memory_v1/config/initialization.json").read_text())
    assert a["action_head_sha256"] == b["action_head_sha256"]
    assert a["head_trainable"] == b["head_trainable"] and a["frozen_backbone"] and b["frozen_backbone"]
    decision = extend_decision(args.output)
    write_json(args.output / "diagnostics/budget_decision.json", decision)
    completed = args.steps
    if decision["extend"] and args.max_steps > args.steps:
        for arm in ("baseline", "memory_v1"):
            with capture_console(args.output / f"logs/{arm}.log"):
                train_arm(args, cache, processor, arm, args.steps, args.max_steps)
        completed = args.max_steps
    aggregated = []
    for arm in ("baseline", "memory_v1"):
        with capture_console(args.output / f"logs/{arm}_actions.log"):
            aggregated.extend(action_evaluation(args, cache, arm))
    write_csv(args.output / "metrics/action_by_episode.csv", aggregated)
    summary = []
    for arm in ("baseline", "memory_v1"):
        for condition in ("normal", "empty", "wrong"):
            for subset in ("uniform", "panel", "history_candidates", "post_merge"):
                rows = [r for r in aggregated if r["arm"] == arm and r["condition"] == condition and r["subset"] == subset]
                if rows:
                    summary.append({"arm": arm, "condition": condition, "subset": subset,
                        "independent_episodes": len(rows),
                        **{k: float(np.mean([r[k] for r in rows])) for k in
                           ("mae_first8", "mae_horizon32", "rmse_horizon32")}})
    write_csv(args.output / "metrics/action_summary.csv", summary)
    plot_results(args.output, aggregated)
    report = {"status": "completed", "optimizer_steps_each": completed,
        "train_episodes": split["train_episodes"], "validation_episodes": split["validation_episodes"],
        "budget_decision": decision, "matched_initial_head_hash": a["action_head_sha256"],
        "action_summary": summary, "by_episode": aggregated,
        "arms": {arm: {"initial": [r for r in records(args.output/arm) if r["kind"] == "eval"][0],
                        "final": [r for r in records(args.output/arm) if r["kind"] == "eval"][-1],
                        "best": min([r for r in records(args.output/arm) if r["kind"] == "eval"], key=lambda r:r["validation_loss"]),
                        "peak_training_allocated_gib": max(r["peak_allocated_gib"] for r in records(args.output/arm) if r["kind"] == "train"),
                        "mean_cached_decisions_per_second": float(np.mean([r["decisions_per_second"] for r in records(args.output/arm) if r["kind"] == "train"]))}
                 for arm in ("baseline", "memory_v1")},
        "limitations": ["one training seed; four validation trajectories of the same scene instance",
            "open-loop sampled-panel normalized errors, not closed-loop B1K success",
            "grasp/release candidates are not certified history-essential segments",
            "best loss point is diagnostic; checkpoints contain final model and resumable optimizer",
            "cached throughput excludes Eagle and video decoding; no augmentation"]}
    write_json(args.output / "report.json", report)
    print("COMPLETED", completed, "steps each", flush=True)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--checkpoint", type=Path, default=Path("checkpoints/groot_n1d6_behavior1k"))
    p.add_argument("--dataset", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--steps", type=int, default=200)
    p.add_argument("--max-steps", type=int, default=400)
    p.add_argument("--eval-interval", type=int, default=25)
    p.add_argument("--learning-rate", type=float, default=1e-5)
    p.add_argument("--seed", type=int, default=42)
    args = p.parse_args()
    assert args.steps == 200 and args.max_steps in (200, 400)
    with capture_console(args.output / "logs/paired.log"):
        run(args)


if __name__ == "__main__":
    main()
