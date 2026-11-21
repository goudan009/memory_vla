"""Record, rebuild and resume one live Task40 MemoryVLA trajectory."""
import argparse
import copy
import json
import os
import random
import sys
from pathlib import Path

from groot_task40_trace import read_state, save_policy_input, serializable


def write_json(path, value):
    path.write_text(json.dumps(serializable(value), indent=2, allow_nan=False) + "\n")


def approach_pose(path, instance, step):
    with path.open() as stream:
        for line in stream:
            row = json.loads(line)
            if row["instance_id"] == instance and row["action_steps"] == step:
                return row["robots"]["robot_r1"]
    raise ValueError(f"No approach pose at step {step} for instance {instance}")


def main(args):
    sys.path[:0] = [str(args.behavior / "OmniGibson"), str(args.behavior / "bddl"), str(args.groot)]
    import isaacsim
    import av
    import numpy as np
    import omnigibson as og
    import torch
    import zmq
    from omnigibson import object_states
    from omnigibson.utils.usd_utils import GripperRigidContactAPI
    from gr00t.eval.sim.BEHAVIOR.behavior_env import BEHAVIORGr00tEnv, preprocess_action, preprocess_obs
    from gr00t.policy.server_client import PolicyClient

    source_start = json.loads((args.source_run / "start.json").read_text()) if args.source_run else None
    if source_start:
        args.seed, args.instance = source_start["seed"], source_start["instance"]
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    os.environ["GROOT_EVAL_INSTANCE_IDS"] = str(args.instance)
    client = PolicyClient(host="127.0.0.1", port=args.port)
    client.socket.setsockopt(zmq.RCVTIMEO, 180000)
    client.reset()
    env = BEHAVIORGr00tEnv("make_microwave_popcorn")
    obs, _ = env.reset(seed=args.seed)
    base = env.env
    while "env" in vars(base):
        base = vars(base)["env"]
    robot = env.robot
    microwave = base.task.object_scope["microwave.n.02_1"].wrapped_obj
    bag = base.task.object_scope["popcorn__bag.n.01_1"].wrapped_obj
    states = {name: getattr(object_states, name) for name in ("Open", "Inside", "ToggledOn", "Cooked")}
    repeat = args.repeat_number
    intervention = args.memory_intervention
    record_mode = bool(repeat or intervention)
    if record_mode:
        door_joint, = microwave.states[states["Open"]].relevant_joints_info[1]
        door = next(link for link in microwave.links.values() if link.prim_path == door_joint.body1)
        camera_sensors = {name: sensor for name, sensor in robot.sensors.items()
                          if any(link in name for link in ("zed_link", "realsense_link"))}
        assert len(camera_sensors) == 3, list(robot.sensors)
    if args.approach_trace:
        pose = approach_pose(args.approach_trace, args.instance, args.approach_step)
        robot.set_position_orientation(torch.tensor(pose["position"]), torch.tensor(pose["orientation"]))
        robot.set_joint_positions(torch.tensor(pose["joint_positions"]))
        og.sim.render()
        obs = preprocess_obs(env, base.get_obs()[0])

    episode = f"task40-instance{args.instance}-seed{args.seed}"
    session = source_start["clock"] if source_start else {"session_id": args.output.name, "env_ids": [0], "episode_ids": [episode]}
    decision = 0
    fixed_first_state = {}
    cameras = ("left_wrist", "right_wrist", "head")

    def rgb(observation):
        return {camera: observation[f"video.observation.images.rgb.{camera}_256_256"] for camera in cameras}

    def distances():
        origin = microwave.get_position_orientation()[0]
        return {"base_xy_m": float(torch.linalg.vector_norm((robot.get_position_orientation()[0] - origin)[:2])),
                "eef_m": {arm: float(torch.linalg.vector_norm(robot.get_eef_position(arm) - origin))
                          for arm in robot.arm_names}}

    def truth():
        value = read_state(base, states)
        value.update(step=base._current_step, sim_time_s=float(og.sim.current_time), distance=distances(),
                     goal_status=(env.info or {}).get("done", {}).get("goal_status"),
                     task_progress=(env.info or {}).get("task_progress"),
                     predicate_options=[[bool(predicate.evaluate()) for predicate in option]
                                        for option in base.task.ground_goal_state_options])
        return serializable(value)

    def measure():
        return serializable({"poses": {item.name: item.get_position_orientation() for item in (robot, microwave, bag)},
               "joints": {item.name: item.get_joint_positions() for item in (robot, microwave)},
               "controllers": {name: controller.dump_state() for name, controller in robot.controllers.items()},
               "truth": truth()})

    def interaction():
        low, high = door.aabb
        contacts = {}
        closest = {}
        for arm, fingers in robot.finger_links.items():
            paths = {finger.prim_path for finger in fingers}
            points = [finger.get_position_orientation()[0] for finger in fingers]
            closest[arm] = min(float(torch.linalg.vector_norm(torch.maximum(low - point, point - high).clamp_min(0)))
                               for point in points)
            contacts[arm] = [{"body": body, "finger": finger, "point_world": point, "impulse": force}
                             for body, finger, force, point, _, _ in GripperRigidContactAPI.get_contact_data(
                                 robot.scene.idx, column_prim_paths=paths)
                             if body.startswith(microwave.prim_path + "/") and float(force) != 0.0]
        return {"door_body": door.prim_path, "door_aabb": [low, high],
                "finger_origin_to_door_aabb_m": closest, "microwave_contacts": contacts}

    def stop_reason(row):
        micro = row["microwaves"][microwave.name]
        if micro["Open"]:
            return "door_opened"
        if args.memory_intervention:
            return "later_stage_before_opening" if micro["ToggledOn"] else None
        if ("TRUE" in row["grasp_states"].values()
                or row["popcorn"][bag.name]["inside"][microwave.name] or micro["ToggledOn"]):
            return "later_stage_before_opening"

    def clock():
        return client.call_endpoint("task40_clock", requires_input=False)

    def check_clock(expected):
        current, = clock()
        assert current["key"] == [session["session_id"], "0", episode]
        assert current["next_decision"] == expected
        assert all(current[name]["last_written"] == expected - 1 for name in ("perceptual", "cognitive"))
        return current

    def run_decision(branch, stream, container, video):
        nonlocal decision, obs
        directory = (args.output / "metrics" if record_mode else args.output) / branch / f"{decision:04d}"
        batch = {key: [value] if isinstance(value, str) else np.asarray(value)[None, None]
                 for key, value in obs.items()}
        if args.memory_intervention and decision == anchor:
            first_state_delta = {}
            with np.load(args.source_run / "anchor_live_observation.npz", allow_pickle=False) as archive:
                for key in archive.files:
                    if key.startswith("video."):
                        batch[key] = archive[key]
                    elif key.startswith("state."):
                        reference = fixed_first_state.setdefault(key, batch[key].copy())
                        first_state_delta[key] = float(np.max(np.abs(batch[key] - reference)))
                        batch[key] = reference
            write_json(directory.parent / "first_input_delta.json", first_state_delta)
        language = save_policy_input(directory, batch)
        first_step, first_time = base._current_step, float(og.sim.current_time)
        options = dict(session, memory_diagnostics=True) if args.memory_intervention else session
        proposed, diagnostics = client.get_action(batch, options=options)
        if args.memory_intervention:
            write_json(directory / "retrieval.json", diagnostics["memory_diagnostics"])
            np.savez_compressed(directory / "proposed.npz", **proposed)
        if any(value.shape[1] < 8 or not np.isfinite(value).all() for value in proposed.values()):
            raise RuntimeError("Invalid policy action chunk")
        executed = {key: value[0, :8] for key, value in proposed.items()}
        env_actions = []
        reason = None
        for index in range(8):
            action = {key: value[index] for key, value in executed.items()}
            env_actions.append(preprocess_action(action)["robot_r1"].cpu().numpy())
            obs, _, terminated, truncated, info = env.step(action)
            if terminated or truncated or env._physx_crashed:
                raise RuntimeError(f"Task40 terminated inside decision {decision}, control step {index}: {info}")
            row = dict(truth(), decision=decision, branch=branch)
            if record_mode:
                row["interaction"] = serializable(interaction())
                reason = stop_reason(row)
            stream.write(json.dumps(row) + "\n")
            inputs = list(rgb(obs).values())
            frame = av.VideoFrame.from_ndarray(np.concatenate(inputs, axis=1), format="rgb24")
            for packet in video.encode(frame):
                container.mux(packet)
            if reason:
                break
        n_steps = len(env_actions)
        np.savez_compressed(directory / "executed.npz", **{key: value[:n_steps] for key, value in executed.items()},
                            robot_r1=np.stack(env_actions))
        next_clock = check_clock(decision + 1)
        assert base._current_step - first_step == n_steps and (n_steps == 8 or reason)
        write_json(directory / "decision.json", {"decision": decision, "episode": episode, "session": session,
                   "language": language, "steps": [first_step, base._current_step],
                   "sim_time_s": [first_time, float(og.sim.current_time)], "memory_after": next_clock,
                   "stop_reason": reason})
        print(f"TASK40_DECISION {branch} {decision} {base._current_step} {distances()}", flush=True)
        decision += 1
        return reason

    def run_branch(branch, count, stop=None):
        directory = (args.output / "metrics" if record_mode else args.output) / branch
        directory.mkdir()
        video_path = args.output / "videos" / f"{branch}.mp4" if record_mode else directory / "short.mp4"
        with (directory / "truth.jsonl").open("x", buffering=1) as stream, av.open(str(video_path), "w") as video_file:
            encoder = video_file.add_stream("libx264", rate=30)
            encoder.width, encoder.height, encoder.pix_fmt = 768, 256, "yuv420p"
            try:
                for _ in range(count):
                    reason = run_decision(branch, stream, video_file, encoder)
                    if reason:
                        return reason
                    if stop and stop():
                        return True
            finally:
                for packet in encoder.encode():
                    video_file.mux(packet)
        return False

    def at_door():
        proximity = distances()
        return (decision >= 3 and proximity["base_xy_m"] <= 1.6 and min(proximity["eef_m"].values()) <= 0.60
                and not microwave.states[states["Open"]].get_value()
                and not any(robot.is_grasping(arm=arm, candidate_obj=bag).name == "TRUE" for arm in robot.arm_names))

    def restore(saved, before, path, repeat_index=None):
        nonlocal decision, obs
        assert saved["session"] == session
        og.sim.load_state(saved["sim"], serialized=True)
        base._current_step = saved["step"]
        base.task._success = saved["task_success"]
        env._task_progress_dict = saved["progress"]
        env.info = saved["info"]
        for metric, metric_state in zip(env.metrics, saved["metrics"], strict=True):
            vars(metric).clear()
            vars(metric).update(metric_state)
        random.setstate(saved["python_rng"])
        np.random.set_state(saved["numpy_rng"])
        torch.set_rng_state(saved["torch_rng"])
        torch.cuda.set_rng_state_all(saved["cuda_rng"])
        decision = saved["decision"]
        check_clock(decision)
        after = measure()
        position = {name: float(np.abs(np.asarray(pose[0]) - after["poses"][name][0]).max()) for name, pose in before["poses"].items()}
        rotation = {name: float(min(np.abs(np.asarray(pose[1]) - after["poses"][name][1]).max(),
                                    np.abs(np.asarray(pose[1]) + after["poses"][name][1]).max())) for name, pose in before["poses"].items()}
        joints = {name: float(np.abs(np.asarray(value) - after["joints"][name]).max()) for name, value in before["joints"].items()}
        controller_equal = before["controllers"] == after["controllers"]
        write_json(path, {"position_m": position, "quaternion": rotation, "joint": joints,
                          "controllers_equal": controller_equal, "before": before, "after": after})
        assert max((*position.values(), *rotation.values(), *joints.values())) < 1e-4 and controller_equal
        sim_time = float(og.sim.current_time)
        refresh = {}
        camera_poses = []
        for frame_index in range(1, 7 if record_mode else 2):
            og.sim.render()
            obs = preprocess_obs(env, base.get_obs()[0])
            if record_mode:
                refresh.update({f"{frame_index:02d}_{name}": frame.copy() for name, frame in rgb(obs).items()})
                camera_poses.append({name: sensor.get_position_orientation() for name, sensor in camera_sensors.items()})
        if record_mode:
            assert float(og.sim.current_time) == sim_time
            np.savez_compressed(args.output / "metrics" / f"refresh_{repeat_index:02d}.npz", **refresh)
            write_json(args.output / "metrics" / f"camera_poses_{repeat_index:02d}.json", camera_poses)

    if args.source_run:
        before = json.loads((args.source_run / "anchor.json").read_text())
        checkpoint_path = args.source_run / "sim_checkpoint.pt"
        trials = []
        numbers = range(1, 4) if intervention or not repeat else (repeat,)
        for number in numbers:
            saved = torch.load(checkpoint_path, weights_only=False)
            anchor = saved["decision"]
            condition = ("A_normal", "B_empty", "C_wrong")[number - 1] if args.memory_intervention else None
            rebuilt = client.call_endpoint("task40_rebuild", {"decisions": anchor, "repeat_number": number,
                                                             "intervention": condition})
            assert rebuilt == clock()
            metrics = args.output / "metrics" if record_mode else args.output
            restore(saved, before, metrics / f"restore_compare_{number}.json", number if record_mode else None)
            branch = condition or (f"repeat_{number:02d}" if repeat else f"restore_{number}")
            print("TASK40_REPEAT_START", number, anchor, flush=True)
            reason = run_branch(branch, args.cycles)
            trials.append({"branch": branch, "decisions": decision - anchor,
                           "steps": base._current_step - saved["step"], "final_clock": clock(),
                           "stop_reason": reason or "decision_limit"})
        if intervention:
            for number, trial in enumerate(trials, 1):
                write_json(metrics / f"trial_{number:02d}.json", {"anchor_decision": anchor, "anchor_step": saved["step"], **trial})
            og.shutdown()
            return
        if repeat:
            write_json(metrics / f"trial_{repeat:02d}.json", {"anchor_decision": anchor, "anchor_step": saved["step"], **trials[0]})
            print("TASK40_REPEAT_DONE", repeat, trials[0]["stop_reason"], flush=True)
            og.shutdown()
            return
        write_json(args.output / "repeat_summary.json", {"source_run": str(args.source_run), "anchor_decision": anchor,
                   "anchor_step": saved["step"], "trials": trials})
        print("TASK40_REPEAT_DONE", len(trials), flush=True)
        client.kill_server()
        og.shutdown()
        return

    write_json(args.output / "start.json", {"instance": args.instance, "seed": args.seed,
               "approach_pose_step": args.approach_step if args.approach_trace else None,
               "clock": session, "truth": truth()})
    print("TASK40_START", distances(), flush=True)
    if not run_branch("prefix", args.max_prefix, stop=at_door):
        raise RuntimeError("Closed-door near-field checkpoint not reached within prefix budget")

    anchor = decision
    before = measure()
    checkpoint = {"sim": og.sim.dump_state(serialized=True).detach().cpu().clone(), "step": base._current_step,
                  "progress": copy.deepcopy(env._task_progress_dict), "info": copy.deepcopy(env.info),
                  "task_success": base.task._success, "metrics": [copy.deepcopy(vars(metric)) for metric in env.metrics],
                  "python_rng": random.getstate(), "numpy_rng": np.random.get_state(), "torch_rng": torch.get_rng_state(),
                  "cuda_rng": torch.cuda.get_rng_state_all(), "decision": anchor, "session": session}
    torch.save(checkpoint, args.output / "sim_checkpoint.pt")
    write_json(args.output / "anchor.json", before)
    assert client.call_endpoint("task40_checkpoint", requires_input=False) == clock()
    print("TASK40_CHECKPOINT", anchor, flush=True)
    run_branch("original", args.cycles)

    rebuilt = client.call_endpoint("task40_rebuild", {"decisions": anchor})
    assert rebuilt == clock()
    saved = torch.load(args.output / "sim_checkpoint.pt", weights_only=False)
    restore(saved, before, args.output / "restore_compare.json")
    run_branch("restored", args.cycles)
    write_json(args.output / "summary.json", {"anchor_decision": anchor, "anchor_step": saved["step"],
               "original_cycles": args.cycles, "restored_cycles": decision - anchor,
               "restored_steps": base._current_step - saved["step"], "final_clock": clock()})
    print(f"TASK40_REPLAY_DONE {decision - anchor} decisions", flush=True)
    client.kill_server()
    og.shutdown()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--source-run", type=Path)
    parser.add_argument("--repeat-number", type=int, choices=range(1, 4))
    parser.add_argument("--memory-intervention", action="store_true")
    parser.add_argument("--behavior", type=Path, required=True)
    parser.add_argument("--groot", type=Path, required=True)
    parser.add_argument("--instance", type=int, default=4)
    parser.add_argument("--seed", type=int, default=22004)
    parser.add_argument("--port", type=int, default=5570)
    parser.add_argument("--approach-trace", type=Path)
    parser.add_argument("--approach-step", type=int, default=320)
    parser.add_argument("--max-prefix", type=int, default=45)
    parser.add_argument("--cycles", type=int, default=10)
    parsed = parser.parse_args()
    if parsed.repeat_number and not parsed.source_run:
        parser.error("--repeat-number requires --source-run")
    if parsed.memory_intervention and (not parsed.source_run or parsed.repeat_number):
        parser.error("--memory-intervention requires --source-run and all three branches")
    main(parsed)
