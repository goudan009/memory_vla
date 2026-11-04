#!/usr/bin/env python3
"""Load the public B1K π0.5 checkpoint and run one synthetic RGB/state inference."""

import os
import pathlib
import sys
import time
import dataclasses

os.environ.setdefault("XLA_PYTHON_CLIENT_MEM_FRACTION", "0.85")
os.environ.setdefault("XLA_PYTHON_CLIENT_ALLOCATOR", "platform")

import jax
import numpy as np

from b1k.policies import policy_config
from b1k.training import config as training_config

ROOT = pathlib.Path("/nvme3/projects/SERF-VLA/checkpoints/behavior-1k-solution/behavior_50t_checkpoint")
CONFIG_NAME = "pi_behavior_b1k_fast"


def main() -> None:
    private = [path for path in sys.path if "/home/2025201184WZR/" in path]
    if private:
        raise RuntimeError(f"Private WZR paths leaked into sys.path: {private}")

    print(f"Python: {sys.executable}", flush=True)
    print(f"JAX: {jax.__version__}; devices: {jax.devices()}", flush=True)
    if not any(device.platform == "gpu" for device in jax.devices()):
        raise RuntimeError("JAX GPU is unavailable; run this test with the policy GPU environment loaded")

    config = training_config.get_config(CONFIG_NAME)
    config = dataclasses.replace(
        config,
        assets_base_dir=str(ROOT / "assets"),
        checkpoint_base_dir=str(ROOT.parent),
    )
    policy = policy_config.create_trained_policy(config, ROOT, sample_kwargs={"num_steps": 10})

    obs = {
        "observation/egocentric_camera": np.zeros((224, 224, 3), dtype=np.uint8),
        "observation/wrist_image_left": np.zeros((224, 224, 3), dtype=np.uint8),
        "observation/wrist_image_right": np.zeros((224, 224, 3), dtype=np.uint8),
        "observation/state": np.zeros(256, dtype=np.float32),
        "task_index": np.asarray(21, dtype=np.int32),
    }
    start = time.monotonic()
    result = policy.infer(obs)
    actions = np.asarray(result["actions"])
    if actions.ndim != 3 or actions.shape[0] != 1 or actions.shape[-1] < 23:
        raise RuntimeError(f"Unexpected action shape: {actions.shape}")
    if not np.isfinite(actions).all():
        raise RuntimeError("Model produced NaN or Inf actions")
    print(f"Inference OK: action shape={actions.shape}; elapsed={time.monotonic()-start:.2f}s", flush=True)


if __name__ == "__main__":
    main()
