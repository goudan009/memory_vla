"""Age-matched history interventions and post-consolidation gradient check.

Reuses the pilot's actual frozen Eagle cache and final saved weights. Wrong
history comes from an earlier segment, with decision timestamps matched to the
normal history so that content and age are not changed simultaneously.
"""

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import torch
from safetensors import safe_open

from v1_groot_memoryvla.experiment.first_training import (
    decision, gradient_norms, interventions, seeded, slice_features, write_json,
)
from v1_groot_memoryvla.model.gr00t_n1d6.gr00t_n1d6 import Gr00tN1d6
from v1_groot_memoryvla.experiment.output import capture_console


def checkpoint_schema(path):
    schema = {}
    for shard in path.glob("model*.safetensors"):
        with safe_open(shard, framework="pt", device="cpu") as handle:
            for key in handle.keys():
                schema[key] = handle.get_slice(key).get_shape()
    return schema


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--reference-checkpoint", type=Path,
        default=Path("outputs/v1_groot_memoryvla/validation/acceptance/checkpoint"))
    args = parser.parse_args()
    with capture_console(args.run / "logs/history_checks.log"):
        analyze(args)


def analyze(args):
    torch.set_num_threads(8)
    output = args.run / "diagnostics/history_checks"
    output.mkdir(exist_ok=False)
    cached = torch.load(args.run / "cache/frozen_eagle_cache.pt", weights_only=True, map_location="cpu")
    model, info = Gr00tN1d6.from_pretrained(args.run / "checkpoints/final",
        torch_dtype=torch.float32, output_loading_info=True, backbone_trainable_params_fp32=False)
    assert not info["missing_keys"] and not info["unexpected_keys"]
    model.memory.cuda()
    model.action_head.cuda()
    model.eval()
    results = []
    for normal in cached["validation"]:
        # Counterfactual content is inserted into the exact same time slots.
        wrong = dict(cached["train"][0], timesteps=normal["timesteps"])
        for seed in (909, 1234, 2345):
            case = output / f"group{normal['group_id']}_seed{seed}"
            case.mkdir()
            result = interventions(model, normal, wrong, seed, case)
            result.update({"group_id": normal["group_id"], "seed": seed,
                           "history_age_matched": True})
            write_json(case / "result.json", result)
            results.append(result)
            print("AGE_MATCHED", normal["group_id"], seed, result["normal_vs_wrong"], flush=True)
    normal = cached["validation"][0]
    model.memory.reset()
    model.zero_grad(set_to_none=True)
    with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
        for index in range(17):
            decision(model, normal, index)
    for bank in (model.memory.perceptual, model.memory.cognitive):
        assert bank.merge_count["probe"] == 1
        assert all(not stored.requires_grad and stored.grad_fn is None
                   for _, stored in bank.bank["probe"])
    model.train()
    model.backbone.eval()
    seeded(987)
    with torch.autocast("cuda", dtype=torch.bfloat16):
        fused = model.memory(slice_features(normal["features"], 17, 18),
            normal["input_ids"][17:18].cuda(), {"mode": "train_sequence", "sequence_start": False,
                "sequence_end": True, "episode_keys": ["probe"], "timesteps": [normal["timesteps"][17]]})
        loss = model.action_head(fused, slice_features(normal["actions"], 17, 18))["loss"]
    loss.backward()
    gradients = gradient_norms(model)
    old = args.reference_checkpoint
    old_schema = checkpoint_schema(old)
    new_schema = checkpoint_schema(args.run / "checkpoints/final")
    assert old_schema and old_schema == new_schema, "Pilot must not change the existing v1 network"
    means = {name: {metric: float(np.mean([r[name][metric] for r in results]))
                    for metric in results[0][name]}
             for name in ("normal_vs_wrong", "normal_vs_empty", "merged_entry_intervention")}
    label_means = {name: float(np.mean([r["normalized_action_mae_to_label"][name]["mae_horizon32"]
                                       for r in results])) for name in ("normal", "empty", "wrong")}
    report = {"history_age_matched": True, "cases": 6, "training_seeds": 1,
        "probe_action_noise_seeds": [909, 1234, 2345], "validation_group_ids": [100, 120],
        "mean_action_differences": means, "mean_normalized_action_mae_to_label": label_means,
        "normal_history_better_than_empty_cases": sum(
            r["normalized_action_mae_to_label"]["normal"]["mae_horizon32"] <
            r["normalized_action_mae_to_label"]["empty"]["mae_horizon32"] for r in results),
        "normal_history_better_than_wrong_cases": sum(
            r["normalized_action_mae_to_label"]["normal"]["mae_horizon32"] <
            r["normalized_action_mae_to_label"]["wrong"]["mae_horizon32"] for r in results),
        "post_merge_decision18_loss": float(loss.detach()),
        "post_merge_decision18_memory_gradients": gradients,
        "post_merge_history_detached": True, "network_parameter_names_and_shapes_unchanged": True,
        "comparison_checkpoint": str(old),
        "limitations": ["six probes are two target observations x three action noise seeds, not independent episodes",
            "BF16 action changes alone do not establish an improvement",
            "original before/after pilot probes changed content and absolute time together; use these age-matched results"]}
    write_json(output / "summary.json", report)
    package = Path(__file__).resolve().parents[1]
    source_hashes = {str(p.relative_to(package)): hashlib.sha256(p.read_bytes()).hexdigest()
                     for p in package.rglob("*.py") if ".venv" not in p.relative_to(package).parts}
    write_json(args.run / "config/source_sha256.json", source_hashes)
    print(json.dumps({k: v for k, v in report.items() if k != "post_merge_decision18_memory_gradients"}, indent=2), flush=True)


if __name__ == "__main__":
    main()
