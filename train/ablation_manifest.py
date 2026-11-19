"""Validate a comparison manifest, not a B1K/OpenPI training entrypoint.

The SERF B1K checkpoint is PiBehavior, while this project currently contains a
standalone Pi0 snapshot. A validated manifest must not be mistaken for a
trained or loadable policy. Keep data/checkpoint adaptation explicit upstream.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any


def load_manifest(path: str | Path) -> dict[str, Any]:
    with open(path, encoding="utf-8") as source:
        manifest = json.load(source)
    validate_manifest(manifest)
    return manifest


def validate_manifest(manifest: dict[str, Any]) -> None:
    """Reject invalid action partitions and hidden differences in memory budget."""
    common = manifest["common"]
    arms = manifest["arms"]
    if set(arms) != {"none", "uniform", "action_conditioned"}:
        raise ValueError("the manifest needs exactly three named arms")
    if any(name != arm["memory_type"] for name, arm in arms.items()):
        raise ValueError("arm names and memory_type must agree")
    if "memory" in arms["none"]:
        raise ValueError("the none arm must not initialize a memory bank")
    if arms["uniform"]["memory"] != arms["action_conditioned"]["memory"]:
        raise ValueError("uniform and conditioned memory budgets must match exactly")

    model_dim = int(common["model_action_dim"])
    valid_dim = int(common["valid_action_dim"])
    groups = common["action_groups"]
    indices = [idx for group in groups.values() for idx in group]
    if not (0 < valid_dim <= model_dim) or len(indices) != valid_dim or sorted(indices) != list(range(valid_dim)):
        raise ValueError("action groups must partition exactly the valid action dimensions")
    passthrough = common["passthrough_action_indices"]
    if passthrough != list(range(valid_dim, model_dim)):
        raise ValueError("padded action dimensions must stay with the base readout")
    if common["step_batch_size"] != 1 or common["episode_order"] != "chronological":
        raise ValueError("stateful memory requires chronological single-step batches")

    memory = arms["uniform"]["memory"]
    if memory["num_action_groups"] != len(groups) or memory["num_mem_tokens"] != 1 + len(groups):
        raise ValueError("group and memory token counts must match")
    if memory["action_group_indices"] != list(groups.values()):
        raise ValueError("model group indices must match physical action groups")
    if memory["passthrough_action_indices"] != passthrough:
        raise ValueError("model padding dimensions must bypass grouped readouts")
    if memory["injection_mode"] != "modulation":
        raise ValueError("the paper's main three-arm comparison requires shared expert modulation")
    if memory["mem_length"] <= 0 or memory["retrieval_layers"] <= 0 or memory["num_attention_heads"] <= 0:
        raise ValueError("memory capacity and retrieval layers/heads must be positive")
