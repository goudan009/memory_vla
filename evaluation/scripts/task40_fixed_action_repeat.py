"""Replay identical recorded Task40 actions from one simulator checkpoint."""
import argparse
import copy
import json
import random
import sys
from pathlib import Path

import numpy as np


def main(args):
    sys.path[:0] = [str(args.behavior / "OmniGibson"), str(args.behavior / "bddl"), str(args.groot)]
    import isaacsim
    import omnigibson as og
    import torch
    from gr00t.eval.sim.BEHAVIOR.behavior_env import BEHAVIORGr00tEnv

    source = args.source
    saved = torch.load(source / "sim_checkpoint.pt", weights_only=False)
    start = json.loads((source / "start.json").read_text())
    random.seed(start["seed"])
    np.random.seed(start["seed"])
    torch.manual_seed(start["seed"])
    import os
    os.environ["GROOT_EVAL_INSTANCE_IDS"] = str(start["instance"])

    env = BEHAVIORGr00tEnv("make_microwave_popcorn")
    env.reset(seed=start["seed"])
    base = env.env
    while "env" in vars(base):
        base = vars(base)["env"]
    robot = env.robot
    microwave = base.task.object_scope["microwave.n.02_1"].wrapped_obj
    decision_dirs = sorted((source.parent / "metrics" / "repeat_01").glob("[0-9][0-9][0-9][0-9]"))
    if args.decisions:
        decision_dirs = decision_dirs[:args.decisions]
    actions = []
    for directory in decision_dirs:
        with np.load(directory / "executed.npz", allow_pickle=False) as archive:
            arrays = {key: archive[key].copy() for key in archive.files if key.startswith("action.")}
        length = len(next(iter(arrays.values())))
        actions.extend([{key: value[step] for key, value in arrays.items()} for step in range(length)])

    def restore():
        og.sim.load_state(saved["sim"], serialized=True)
        base._current_step = saved["step"]
        base.task._success = saved["task_success"]
        env._task_progress_dict = copy.deepcopy(saved["progress"])
        env.info = copy.deepcopy(saved["info"])
        for metric, metric_state in zip(env.metrics, saved["metrics"], strict=True):
            vars(metric).clear()
            vars(metric).update(copy.deepcopy(metric_state))
        random.setstate(saved["python_rng"])
        np.random.set_state(saved["numpy_rng"])
        torch.set_rng_state(saved["torch_rng"])
        torch.cuda.set_rng_state_all(saved["cuda_rng"])
        og.sim.render()
        base.get_obs()

    def measure():
        return np.concatenate([
            robot.get_position_orientation()[0].detach().cpu().numpy().reshape(-1),
            robot.get_position_orientation()[1].detach().cpu().numpy().reshape(-1),
            robot.get_joint_positions().detach().cpu().numpy().reshape(-1),
            *(robot.get_eef_position(arm).detach().cpu().numpy().reshape(-1) for arm in robot.arm_names),
            microwave.get_joint_positions().detach().cpu().numpy().reshape(-1),
        ]).astype(np.float64)

    trajectories = []
    try:
        for trial in range(args.repeats):
            restore()
            rows = [measure()]
            for action in actions:
                _, _, terminated, truncated, _ = env.step(action)
                if terminated or truncated or env._physx_crashed:
                    raise RuntimeError(f"Simulator stopped at repeat {trial + 1}, step {len(rows)}")
                rows.append(measure())
            trajectory = np.stack(rows)
            trajectories.append(trajectory)
            np.save(source.parent / "metrics" / f"fixed_actions_repeat_{trial + 1:02d}.npy", trajectory)
            print(f"TASK40_FIXED_ACTION_REPEAT {trial + 1} steps={len(actions)}", flush=True)
        reference = trajectories[0]
        comparison = []
        for trial, trajectory in enumerate(trajectories[1:], start=2):
            delta = np.abs(reference - trajectory)
            comparison.append({"repeat": trial, "max_abs_by_state_component": delta.max(axis=0).tolist(),
                               "max_abs_overall": float(delta.max()),
                               "mean_abs_overall": float(delta.mean()),
                               "max_base_position_m": float(delta[:, :3].max()),
                               "max_joint_position_rad": float(delta[:, 7:35].max())})
        result = {"repeats": args.repeats, "initial_sim_step": saved["step"], "action_steps": len(actions),
                  "action_decisions": [path.name for path in decision_dirs], "state_vector_size": int(reference.shape[1]),
                  "comparisons_to_repeat_01": comparison}
        target = source.parent / "metrics" / "fixed_action_comparison.json"
        target.write_text(json.dumps(result, indent=2) + "\n")
        print(f"TASK40_FIXED_ACTION_RESULT {target}", flush=True)
    finally:
        og.shutdown()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True, help="experiment_01/checkpoints")
    parser.add_argument("--behavior", type=Path, required=True)
    parser.add_argument("--groot", type=Path, required=True)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--decisions", type=int, default=19)
    main(parser.parse_args())
