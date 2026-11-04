"""Run a B1K task with controller hold actions to validate the simulator."""

import argparse
import json
import os
import sys
import time
from importlib.metadata import version
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--task", default="turning_on_radio")
    parser.add_argument("--steps", type=int, default=200)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.steps < 1:
        parser.error("--steps must be positive")
    args.output.mkdir(parents=True, exist_ok=True)
    start = time.monotonic()

    import numpy as np
    import torch
    import omnigibson as og
    from omnigibson.macros import gm

    gm.ENABLE_FLATCACHE = True
    gm.USE_GPU_DYNAMICS = False
    gm.ENABLE_TRANSITION_RULES = True

    from gello.robots.sim_robot.og_teleop_utils import (
        augment_rooms,
        generate_robot_config,
        get_task_relevant_room_types,
        load_available_tasks,
    )
    from omnigibson.learning.utils.eval_utils import (
        flatten_obs_dict,
        generate_basic_environment_config,
    )

    torch.manual_seed(0)
    np.random.seed(0)
    task_cfg = load_available_tasks()[args.task][0]
    cfg = generate_basic_environment_config(args.task, task_cfg)
    cfg["scene"]["load_room_types"] = augment_rooms(
        get_task_relevant_room_types(args.task), task_cfg["scene_model"], args.task
    )
    cfg["task"]["termination_config"]["max_steps"] = args.steps + 100
    robot_cfg = generate_robot_config(args.task, task_cfg)
    robot_cfg["obs_modalities"] = ["proprio", "rgb"]
    cfg["robots"] = [robot_cfg]
    print("B1K_LOADING", args.task, task_cfg["scene_model"], flush=True)
    env = og.Environment(configs=cfg)
    obs, _ = env.reset()
    robot = env.robots[0]
    assert env.task.activity_goal_conditions, "Task goals were not loaded"
    print("B1K_READY", robot.action_dim, "action dimensions", flush=True)

    action = torch.zeros(robot.action_dim)
    control_dict = robot.get_control_dict()
    for name, controller in robot.controllers.items():
        action[robot.controller_action_idx[name]] = controller.compute_no_op_action(control_dict)
    rollout_start = time.monotonic()
    terminations = 0
    total_reward = 0.0
    for step in range(args.steps):
        obs, reward, terminated, truncated, info = env.step(action)
        assert torch.isfinite(robot.get_joint_positions()).all(), "Nonfinite robot state"
        total_reward += float(reward)
        if terminated or truncated:
            terminations += 1
            if step + 1 < args.steps:
                raise RuntimeError(f"Episode ended early at step {step + 1}")
        if (step + 1) % 25 == 0:
            print("B1K_STEP", step + 1, flush=True)

    from PIL import Image

    cameras = {}
    for key, value in flatten_obs_dict(obs).items():
        if key.endswith("::rgb"):
            array = value.detach().cpu().numpy() if torch.is_tensor(value) else np.asarray(value)
            cameras[key] = list(array.shape)
            filename = "head_rgb.png" if "zed_link" in key else f"camera_{len(cameras)}.png"
            Image.fromarray(array[..., :3].astype(np.uint8)).save(args.output / filename)
    assert cameras, "No RGB observations returned"
    result = {
        "status": "passed",
        "task": args.task,
        "scene": task_cfg["scene_model"],
        "activity_instance_id": cfg["task"]["activity_instance_id"],
        "steps": args.steps,
        "action": "controller_hold",
        "action_dim": robot.action_dim,
        "terminations": terminations,
        "total_reward": total_reward,
        "rollout_seconds": time.monotonic() - rollout_start,
        "total_seconds": time.monotonic() - start,
        "cameras": cameras,
        "python": sys.version,
        "executable": sys.executable,
        "packages": {name: version(name) for name in ("omnigibson", "isaacsim", "torch", "bddl", "joylo", "numpy")},
        "omnigibson_source": og.__file__,
        "isaac_path": os.environ.get("ISAAC_PATH"),
        "data_path": str(gm.DATA_PATH),
        "gpu_id": str(gm.GPU_ID),
    }
    (args.output / "result.json").write_text(json.dumps(result, indent=2) + "\n")
    print("B1K_PASSED", json.dumps(result), flush=True)
    og.shutdown()


if __name__ == "__main__":
    main()
