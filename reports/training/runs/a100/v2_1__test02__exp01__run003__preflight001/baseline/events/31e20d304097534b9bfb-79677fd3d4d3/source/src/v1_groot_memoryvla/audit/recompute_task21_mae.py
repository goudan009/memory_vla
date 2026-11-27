"""Recompute selected-checkpoint MAE without training or changing original results."""

import argparse
from contextlib import contextmanager
import hashlib
import json
from pathlib import Path
import time

import numpy as np
import torch

from v1_groot_memoryvla.experiment.first_training import cache_sequence, slice_features, bank_trace
from v1_groot_memoryvla.experiment.task21_diagnostic import action_predictions, head_hash
from v1_groot_memoryvla.experiment.task21_formal import (
    action_metrics, atomic_json, initialize, window_context, wrong_features)


def memory_window(model, seq, microbatch, collect=False):
    outputs, trace = [], []
    for start in range(0, 32, microbatch):
        end = min(start + microbatch, 32)
        with torch.autocast("cuda", dtype=torch.bfloat16), bank_trace(model.memory) as events:
            features = model.memory(slice_features(seq["features"], start, end),
                seq["input_ids"][start:end].cuda(), window_context(seq, start, end))
        if collect:
            trace.extend(events)
            outputs.append({k: v.detach().cpu() for k, v in features.items()})
    return ({k: torch.cat([f[k] for f in outputs]) for k in outputs[0]} if collect else None), trace


@contextmanager
def inference_audit(model, conditions):
    """Check actual get_action -> vlln -> four DiT calls for every condition."""
    head, records = model.action_head, []
    original = head.get_action

    def checked(features, actions):
        before = features["backbone_features"]
        copy = before.clone()
        record = {"shape": list(before.shape), "dtype": str(before.dtype),
            "noise_state": hashlib.sha256(torch.cuda.get_rng_state().cpu().numpy().tobytes()).hexdigest(),
            "vlln_calls": 0, "dit_calls": 0}
        shared = {**{k: features[k].cpu().clone() for k in ("image_mask", "backbone_attention_mask")},
                  **{k: actions[k].cpu().clone() for k in ("state", "embodiment_id")}}

        def count_ln(*_):
            record["vlln_calls"] += 1

        def count_dit(*_):
            record["dit_calls"] += 1

        hooks = [head.vlln.register_forward_hook(count_ln), head.model.register_forward_hook(count_dit)]
        try:
            result = original(features, actions)
        finally:
            for hook in hooks:
                hook.remove()
        assert features["backbone_features"] is before and torch.equal(before, copy)
        assert record["vlln_calls"] == 1
        assert record["dit_calls"] == head.num_inference_timesteps == 4
        position = len(records)
        if position >= 16:
            reference, reference_shared = records[position % 16]
            assert record == reference, "Conditions used different action inference settings/RNG"
            assert all(torch.equal(v, reference_shared[k]) for k, v in shared.items())
        records.append((record, shared))
        return result

    report = {}
    head.get_action = checked
    try:
        yield report
        assert len(records) == 16 * len(conditions)
        report.update(conditions=conditions, calls_per_condition=16, vlln_per_call=1,
            denoising_steps_per_call=4, same_rng_state_shape_dtype_state_masks=True,
            pre_vlln_inputs_preserved=True, action_microbatch=2,
            method="Gr00tN1d6ActionHead.get_action", autocast="cuda/bfloat16")
    finally:
        head.get_action = original


def checkpoint_files(checkpoint):
    return {p.name: {"size": p.stat().st_size, "mtime_ns": p.stat().st_mtime_ns}
            for p in checkpoint.glob("*.safetensors")}


@torch.no_grad()
def run(args):
    selected = json.loads((args.run / "selected_checkpoints.json").read_text())[args.arm]
    args.checkpoint = Path(selected["path"])
    step = selected["step"]
    args.output = args.run  # initialize reads the original fixed split here.
    processor, train, validation, model = initialize(args, args.checkpoint)
    model.eval().requires_grad_(False)
    root = args.artifact / args.arm / f"shard-{args.shard_index:03d}"
    for name in ("metrics", "diagnostics/actions"):
        (root / name).mkdir(parents=True, exist_ok=True)
    assert not (root / "metrics/action_by_episode.jsonl").exists(), "Use a fresh artifact directory"
    files_before = checkpoint_files(args.checkpoint)
    head_before = head_hash(model)
    episodes = sorted(validation.by_episode)[args.shard_index::args.shards]
    rows, audits, comparisons = [], [], []
    tick = time.monotonic()
    for eid in episodes:
        indices = validation.by_episode[eid]
        middle = len(indices) // 2
        normal, trace = None, None
        if model.memory is not None:
            model.memory.reset()
            for w, index in enumerate(indices[:middle + 1]):
                seq = cache_sequence(model, processor, validation, index, args.microbatch)
                normal, trace = memory_window(model, seq, args.microbatch, collect=w == middle)
                if w % 10 == 0 or w == middle:
                    progress = {"episode": eid, "window": w, "middle_window": middle,
                        "completed_episodes": len(audits), "total_episodes": len(episodes),
                        "seconds": time.monotonic() - tick}
                    atomic_json(root / "status.json", dict(status="running", **progress))
                    print(json.dumps(progress), flush=True)
        else:
            seq = cache_sequence(model, processor, validation, indices[middle], args.microbatch)
        donor_eid = sorted(train.by_episode)[eid % len(train.by_episode)]
        donors = train.by_episode[donor_eid]
        donor_w = max(1, min(len(donors)-1, round(middle/len(indices)*len(donors))))
        wrong = None
        if model.memory is not None:
            warm = cache_sequence(model, processor, train, donors[donor_w-1], args.microbatch)
            donor = cache_sequence(model, processor, train, donors[donor_w], args.microbatch)
            wrong = wrong_features(model, seq, warm, donor, trace)
            for features in (normal, wrong):
                assert features["backbone_features"].shape == seq["features"]["backbone_features"].shape
                for key in ("image_mask", "backbone_attention_mask"):
                    assert torch.equal(features[key], seq["features"][key])
        conditions = ["normal", "empty", "wrong"] if model.memory is not None else ["normal"]
        with inference_audit(model, conditions) as audit:
            rows.extend(action_metrics(model, seq, normal, wrong, root, step, args.arm, donor_eid))
        audits.append(dict(episode=eid, **audit))
        current = np.load(root / f"diagnostics/actions/step{step:06d}_episode{eid}.npz")
        old_path = args.run / args.arm / f"diagnostics/actions/step{step:06d}_episode{eid}.npz"
        old = np.load(old_path)
        for key in ("target", "frames"):
            np.testing.assert_array_equal(current[key], old[key])
        comparison = {"episode": eid, "previous_mae": {}, "prediction_difference_from_previous": {}}
        for condition in ("normal", "empty", "wrong"):
            comparison["previous_mae"][condition] = float(np.abs(old[condition]-old["target"]).mean())
            comparison["prediction_difference_from_previous"][condition] = {
                "mean_abs": float(np.abs(current[condition]-old[condition]).mean()),
                "max_abs": float(np.abs(current[condition]-old[condition]).max())}
        if model.memory is not None:
            # Reproduce the old normal branch by explicitly applying the extra
            # normalization that the old loss call wrote into the input dict.
            legacy = []
            for start in range(0, 32, args.microbatch):
                with torch.autocast("cuda", dtype=torch.bfloat16):
                    f = model.action_head.process_backbone_output(slice_features(normal, start, start+args.microbatch))
                legacy.append({k: v.cpu() for k, v in f.items()})
            legacy = {k: torch.cat([f[k] for f in legacy]) for k in legacy[0]}
            seed = 909 + eid * 100
            reproduced = action_predictions(model, legacy, seq, seed)
            comparison["old_normal_reproduction"] = {
                "mae": float(np.abs(reproduced-old["target"]).mean()),
                "max_abs_prediction_difference": float(np.abs(reproduced-old["normal"]).max())}
            # Same inputs must produce identical actions through all three
            # condition labels; run once per worker with the real action head.
            if len(audits) == 1:
                identical = [action_predictions(model, normal, seq, seed) for _ in conditions]
                assert all(np.array_equal(identical[0], x) for x in identical[1:])
                audit["identical_feature_control_bitwise_equal"] = True
                audits[-1].update(audit)
            model.memory.reset()
        comparisons.append(comparison)
        print(json.dumps({"completed_episode": eid, "arm": args.arm,
            "mae": {r["condition"]: r["mae_horizon32"] for r in rows if r["episode"] == eid},
            "comparison": comparison}), flush=True)
    assert head_hash(model) == head_before
    assert checkpoint_files(args.checkpoint) == files_before
    assert all(p.grad is None for p in model.parameters())
    summary = {"status": "completed", "arm": args.arm, "step": step,
        "checkpoint": str(args.checkpoint), "episodes": episodes,
        "shard_index": args.shard_index, "shards": args.shards,
        "checkpoint_files_unchanged": files_before, "action_head_sha256_unchanged": head_before,
        "training": False, "seconds": time.monotonic()-tick,
        "action_mae": {c: float(np.mean([r["mae_horizon32"] for r in rows if r["condition"] == c]))
                       for c in ("normal", "empty", "wrong")},
        "inference_audit": audits, "comparison": comparisons,
        "source_sha256": {str(p.relative_to(Path(__file__).parents[1])): hashlib.sha256(p.read_bytes()).hexdigest()
            for p in (Path(__file__), Path(__file__).parents[1] / "model/gr00t_n1d6/gr00t_n1d6.py")}}
    atomic_json(root / "summary.json", summary)
    atomic_json(root / "status.json", {"status": "completed", "episodes": episodes})
    print(json.dumps({"summary": summary}), flush=True)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for name in ("run", "dataset", "artifact"):
        p.add_argument("--" + name, type=Path, required=True)
    p.add_argument("--arm", choices=("memory_v1", "baseline"), required=True)
    p.add_argument("--shard-index", type=int, default=0)
    p.add_argument("--shards", type=int, default=1)
    p.add_argument("--microbatch", type=int, default=4)
    p.add_argument("--seed", type=int, default=42)
    args = p.parse_args()
    assert 0 <= args.shard_index < args.shards
    run(args)


if __name__ == "__main__":
    main()
