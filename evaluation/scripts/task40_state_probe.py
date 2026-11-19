"""Run a short Task40 simulator round trip and optional GR00T policy step."""
import argparse
import json
import os
import sys
from pathlib import Path

from groot_task40_trace import read_state, serializable
from task40_feasibility_audit import read_jsonl


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run", type=Path)
    parser.add_argument("--behavior", type=Path, required=True)
    parser.add_argument("--groot", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--policy-port", type=int)
    parser.add_argument("--instance", type=int, default=4)
    args = parser.parse_args()
    args.output.mkdir(parents=True)

    sys.path[:0] = [str(args.behavior / "OmniGibson"), str(args.behavior / "bddl"), str(args.groot)]
    import isaacsim
    import numpy as np
    import torch
    import omnigibson as og
    from omnigibson import object_states
    from gr00t.eval.sim.BEHAVIOR.behavior_env import BEHAVIORGr00tEnv, preprocess_obs

    states = {key: getattr(object_states, key) for key in ("Open", "Inside", "ToggledOn", "Cooked")}
    episode = next(row for row in read_jsonl(args.run / "episodes.jsonl") if row["instance_id"] == args.instance)
    os.environ["GROOT_EVAL_INSTANCE_IDS"] = str(args.instance)
    env = BEHAVIORGr00tEnv("make_microwave_popcorn")
    env.reset(seed=episode["seed"])
    print("TASK40_RESET_OK", flush=True)
    robot = env.robot
    microwave = env.env.task.object_scope["microwave.n.02_1"].wrapped_obj
    bag = env.env.task.object_scope["popcorn__bag.n.01_1"].wrapped_obj
    microwave_pos = microwave.get_position_orientation()[0].cpu().numpy()
    stages = args.run / "stages"
    trace_suffix = f'task40_gpu{episode["gpu_index"]}_env{episode["env_idx"]}.jsonl'
    rows = [row for row in read_jsonl(stages / trace_suffix)
            if row["instance_id"] == args.instance and not row["microwaves"][microwave.name]["Open"]
            and not any(row["grasped"].values()) and row["action_steps"] % 8 == 0]

    def distance(row):
        return min(np.linalg.norm(np.asarray(pose[0]) - microwave_pos)
                   for pose in row["robots"][robot.name]["end_effectors"].values())

    anchor = min(rows, key=distance)
    pose = anchor["robots"][robot.name]
    robot.set_position_orientation(torch.tensor(pose["position"]), torch.tensor(pose["orientation"]))
    robot.set_joint_positions(torch.tensor(pose["joint_positions"]))
    actions = next(row["action"] for row in read_jsonl(stages / f"actions_{trace_suffix}")
                   if row["instance_id"] == args.instance and row["action_start"] == anchor["action_steps"])

    def step(action):
        result = env.step(action)
        if env._physx_crashed or result[2] or result[3]:
            raise RuntimeError(f"Task40 step failed: {result[4]}")
        return result[0]

    def recorded(index):
        return {key: np.asarray(value[index], dtype=np.float32) for key, value in actions.items()}

    def measure():
        return {"poses": {obj.name: serializable(obj.get_position_orientation())
                          for obj in (robot, microwave, bag)},
                "joints": {obj.name: serializable(obj.get_joint_positions()) for obj in (robot, microwave)},
                "controllers": {name: serializable(controller.dump_state())
                                for name, controller in robot.controllers.items()},
                "truth": serializable(read_state(env.env, states)),
                "sim_time_s": float(og.sim.current_time)}

    def errors(before, after):
        return {key: float(np.max(np.abs(np.asarray(before[key]) - np.asarray(after[key]))))
                for key in before}

    step(recorded(0))
    before = measure()
    state = og.sim.dump_state(serialized=True).detach().cpu().clone()
    np.savez_compressed(args.output / "state.npz", state=state.numpy())
    perturbed = step(recorded(1))
    changed = measure()
    with np.load(args.output / "state.npz", allow_pickle=False) as saved:
        og.sim.load_state(torch.from_numpy(saved["state"]), serialized=True)
    after = measure()
    next_obs = step(recorded(1))
    resumed = measure()
    report = {"instance": args.instance, "anchor_action_step": anchor["action_steps"],
              "anchor_eef_distance_m": float(distance(anchor)), "anchor_kind": "historical robot pose only",
              "state_size": state.numel(), "before": before, "changed": changed, "after": after,
              "resumed": resumed,
              "position_max_abs_error": errors(*[{key: pose[0] for key, pose in row["poses"].items()}
                                                  for row in (before, after)]),
              "orientation_max_abs_error": errors(*[{key: pose[1] for key, pose in row["poses"].items()}
                                                     for row in (before, after)]),
              "joint_max_abs_error": errors(before["joints"], after["joints"]),
              "controller_dump_equal": before["controllers"] == after["controllers"],
              "next_camera_shapes": {key: list(value.shape) for key, value in next_obs.items() if key.startswith("video.")},
              "reference_obs_keys_match": sorted(perturbed) == sorted(next_obs)}
    (args.output / "roundtrip.json").write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    print("TASK40_ROUNDTRIP_OK", args.output / "roundtrip.json", flush=True)
    if args.policy_port:
        from gr00t.policy.server_client import PolicyClient

        og.sim.render()
        obs = preprocess_obs(env, env.env.get_obs()[0])
        batch = {key: [value] if isinstance(value, str) else np.asarray(value)[None, None]
                 for key, value in obs.items()}
        client = PolicyClient(host="127.0.0.1", port=args.policy_port, timeout_ms=45000)
        client.reset()
        proposed, _ = client.get_action(batch)
        action = {key: value[0, 0] for key, value in proposed.items()}
        assert all(np.isfinite(value).all() for value in action.values())
        step(action)
        (args.output / "policy.json").write_text(json.dumps({"action_shapes": {key: list(value.shape)
                                                                              for key, value in proposed.items()},
                                                             "after_policy_action": measure()}, indent=2) + "\n")
        print("TASK40_POLICY_STEP_OK", flush=True)
    og.shutdown()


if __name__ == "__main__":
    main()
