"""One bounded diagnostic using a private policy server and unmodified eval sources.

All hooks live in this process only. No training, runtime source edits or contacts
to existing baseline/Memory servers. Simulation is hard capped at 400 actions.
"""
import argparse
import json
import os
from pathlib import Path
import sys
import time


def save_json(path, value):
    Path(path).write_text(json.dumps(value, indent=2, ensure_ascii=False)+"\n")


def serve(args):
    import numpy as np
    import torch
    sys.path.insert(0, args.source)
    from gr00t.data.embodiment_tags import EmbodimentTag
    from gr00t.policy.gr00t_policy import Gr00tPolicy, Gr00tSimPolicyWrapper
    from gr00t.policy.server_client import PolicyServer
    torch.manual_seed(42)
    policy = Gr00tPolicy(EmbodimentTag.BEHAVIOR_R1_PRO, args.checkpoint, device="cuda")
    policy.model.requires_grad_(False)
    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=True)
    capture = {}
    orig_model = policy.model.get_action
    orig_decode = policy.processor.decode_action
    def model_hook(*a, **kw):
        value = orig_model(*a, **kw)
        capture["raw_action"] = value["action_pred"].detach().float().cpu().numpy().copy()
        return value
    def decode_hook(*a, **kw):
        value = orig_decode(*a, **kw)
        capture.update({"decoded/"+k: np.asarray(v).copy() for k,v in value.items()})
        return value
    policy.model.get_action = model_hook
    policy.processor.decode_action = decode_hook
    wrapped = Gr00tSimPolicyWrapper(policy)
    orig_wrapped = wrapped.get_action
    count = 0
    def request_hook(observation, options=None):
        nonlocal count
        if count >= 50:
            raise RuntimeError("Diagnostic server accepts at most 50 action requests")
        seed = 10000+count
        torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)
        capture.clear()
        result = orig_wrapped(observation, options)
        values = {"input/"+k: np.asarray(v) for k,v in observation.items()}
        values.update(capture)
        values.update({"wrapped/"+k: np.asarray(v) for k,v in result[0].items()})
        np.savez_compressed(out/f"request{count:03d}.npz", **values)
        save_json(out/f"request{count:03d}.json", {"request": count, "seed": seed, "time_unix": time.time()})
        count += 1
        print("REQUEST", count, flush=True)
        return result
    wrapped.get_action = request_hook
    server = PolicyServer(wrapped, host=args.host, port=args.port)
    save_json(out/"identity.json", {"pid": os.getpid(), "source": args.source, "checkpoint": args.checkpoint,
                                   "port": args.port, "embodiment": "behavior_r1_pro", "memory": False})
    print("PROBE_SERVER_READY", flush=True)
    server.run()
    server.socket.close(); server.context.term()


def simulate(args):
    # Import Isaac before OmniGibson as required by the actual simulation runtime.
    import isaacsim  # noqa: F401
    sys.path[:0] = [args.source, args.behavior+"/OmniGibson", args.behavior+"/bddl"]
    import av
    import numpy as np
    import gymnasium as gym
    import omnigibson as og
    from gr00t.eval.sim.BEHAVIOR import behavior_env as be
    from gr00t.eval.rollout_policy import WrapperConfigs, create_eval_env
    from gr00t.policy.server_client import PolicyClient
    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=True)
    client = PolicyClient(args.host, args.port)
    assert client.ping(), "Private diagnostic server is unavailable"
    context = {"step": 0, "raw_proprio": None, "robot": None, "executed": None}
    orig_obs, orig_action, orig_step = be.preprocess_obs, be.preprocess_action, be.BEHAVIORGr00tEnv.step
    container = av.open(str(out/"head_left_right.mp4"), "w")
    stream = container.add_stream("libx264", rate=30)
    stream.width, stream.height, stream.pix_fmt = 768, 256, "yuv420p"
    stats_handle = (out/"steps.jsonl").open("w", buffering=1)
    def array(x):
        if hasattr(x, "detach"):
            return x.detach().cpu().numpy().copy()
        return np.asarray(x).copy()
    def obs_hook(env, observation):
        context["raw_proprio"] = array(observation["robot_r1"]["proprio"])
        return orig_obs(env, observation)
    def action_hook(action):
        value = orig_action(action)
        context["executed"] = array(value["robot_r1"])
        return value
    def step_hook(env, action):
        i = context["step"]
        if i >= 400:
            raise RuntimeError("Hard diagnostic cap: no more than 400 simulation actions")
        robot = env.robot
        before = array(robot.get_joint_positions())
        result = orig_step(env, action)
        obs = result[0]
        values = {"env_action": context["executed"], "joint_before": before,
                  "joint_after": array(robot.get_joint_positions()), "raw_proprio_after": context["raw_proprio"]}
        for k,v in action.items():
            values["policy_action/"+k] = array(v)
        for name,c in robot.controllers.items():
            for k,v in (c._goal or {}).items():
                values["controller/"+name+"/"+k] = array(v)
        np.savez_compressed(out/f"step{i:04d}.npz", **values)
        images = [obs["video.observation.images.rgb."+k+"_256_256"] for k in ("head", "left_wrist", "right_wrist")]
        video_frame = av.VideoFrame.from_ndarray(np.concatenate(images, axis=1), format="rgb24")
        for packet in stream.encode(video_frame):
            container.mux(packet)
        stats_handle.write(json.dumps({"step": i, "request": i//8, "chunk_index": i%8,
            "time_unix": time.time(), "joint_max_abs_change": float(np.max(np.abs(values["joint_after"]-before))),
            "terminated": bool(result[2]), "truncated": bool(result[3]), "physx_crashed": bool(env._physx_crashed)})+"\n")
        context["step"] += 1
        return result
    be.preprocess_obs, be.preprocess_action, be.BEHAVIORGr00tEnv.step = obs_hook, action_hook, step_hook
    configs = WrapperConfigs()
    configs.multistep.n_action_steps = 8
    configs.multistep.max_episode_steps = 400
    configs.video.video_dir = None
    env = None
    exit_code = 0
    try:
        env = gym.vector.SyncVectorEnv([lambda: create_eval_env(
            env_idx=0, env_name="sim_behavior_r1_pro/collecting_childrens_toys", total_n_envs=1, wrapper_configs=configs)])
        observations, _ = env.reset()
        current = env.envs[0]
        while type(current).__name__ != "BEHAVIORGr00tEnv":
            current = vars(current)["env"]
        robot = current.robot
        from groot_ab_audit import render_inventory
        save_json(out/"runtime.json", {"source": args.source, "seed": os.environ.get("GROOT_EVAL_SEED"),
            "instance_ids": os.environ.get("GROOT_EVAL_INSTANCE_IDS"), "controller_order": robot.controller_order,
            "joint_names": list(robot.joints), "controller_config": str(robot._controller_config),
            "action_normalize": str(robot._action_normalize), "physics_dt": og.sim.get_physics_dt(),
            "sim_step_dt": og.sim.get_sim_step_dt(), "rendering_dt": og.sim.get_rendering_dt(),
            "render_inventory": render_inventory()})
        np.savez_compressed(out/"initial_raw_proprio.npz", state=context["raw_proprio"])
        client.reset()
        for request in range(50):
            np.savez_compressed(out/f"request{request:03d}_input.npz", **observations)
            actions, _ = client.get_action(observations)
            np.savez_compressed(out/f"request{request:03d}_output.npz", **actions)
            observations, _, terminated, truncated, _ = env.step(actions)
            print("PROBE_CYCLE", request+1, "steps", context["step"], flush=True)
            if np.any(terminated) or np.any(truncated):
                break
        save_json(out/"completion.json", {"steps": context["step"], "requests": request+1,
            "official_episode_result": False, "completed": context["step"]==400})
    except BaseException:
        import traceback
        failure = traceback.format_exc()
        print(failure, file=sys.stderr, flush=True)
        save_json(out/"failure.json", {"steps": context["step"], "traceback": failure})
        exit_code = 1
    finally:
        for packet in stream.encode():
            container.mux(packet)
        container.close(); stats_handle.close()
        if env is not None:
            env.close()
        # Only the dedicated diagnostic port is accepted by argument validation.
        client.kill_server()
        sys.stdout.flush(); sys.stderr.flush()
        os._exit(exit_code)


def analyze(args):
    import ast
    from collections import OrderedDict
    import hashlib
    import numpy as np
    out, server = Path(args.output), Path(args.server)
    runtime = json.loads((out/"runtime.json").read_text())
    names = runtime["joint_names"]
    indices = {"torso": [names.index(f"torso_joint{i}") for i in range(1,5)],
               **{arm+"_arm": [names.index(f"{arm}_arm_joint{i}") for i in range(1,8)] for arm in ("left","right")}}
    controllers = {"torso":"trunk", "left_arm":"arm_left", "right_arm":"arm_right"}
    groups = ["base","torso","left_arm","left_gripper","right_arm","right_gripper"]
    tree = ast.parse((Path(args.source)/"gr00t/eval/sim/BEHAVIOR/behavior_env.py").read_text())
    node = next(n for n in tree.body if isinstance(n,ast.Assign) and any(isinstance(t,ast.Name) and t.id=="PROPRIOCEPTION_INDICES" for t in n.targets))
    ns = {"np":np,"OrderedDict":OrderedDict}
    exec(compile(ast.Module(body=[node],type_ignores=[]),"actual_state_schema","exec"),ns)
    state_slices = ns["PROPRIOCEPTION_INDICES"]["R1Pro"]
    state_before = np.load(out/"initial_raw_proprio.npz")["state"]
    comparisons = {k:0.0 for k in ("network_input", "network_output", "decoded_to_wrapper_float32", "wrapper_to_step", "adapter_to_env", "latest_state_to_request", "goal_vs_arm_torso_command", "base_scaled_speed_norm", "base_yaw_command", "gripper_goal_mapping", "between_step_qpos_continuity")}
    ranges, camera_hashes, fields, tracking, step_rows = {}, {k:[] for k in ("head","left_wrist","right_wrist")}, {}, {k:[] for k in indices}, []
    actions_all, target_errors, gripper_values = [], [], {k:[] for k in ("left", "right")}
    previous_after = None
    def delta(key,a,b):
        comparisons[key] = max(comparisons[key],float(np.max(np.abs(np.asarray(a)-np.asarray(b)))))
    for request in range(50):
        net = np.load(server/f"request{request:03d}.npz")
        inp = np.load(out/f"request{request:03d}_input.npz")
        result = np.load(out/f"request{request:03d}_output.npz")
        for k in inp.files:
            arr = inp[k]
            if arr.dtype.kind in "USO":
                assert np.array_equal(arr,net["input/"+k])
                continue
            delta("network_input",arr,net["input/"+k])
            if request == 0:
                fields[k] = {"shape":list(arr.shape),"dtype":str(arr.dtype),"min":float(arr.min()),"max":float(arr.max()),"finite":bool(np.isfinite(arr).all())}
            if k.startswith("state."):
                delta("latest_state_to_request",arr[0,0],state_before[state_slices[k[6:]]])
        for k in result.files:
            delta("network_output",result[k],net["wrapped/"+k])
            delta("decoded_to_wrapper_float32",net["decoded/"+k[7:]].astype(np.float32),net["wrapped/"+k])
        for camera in camera_hashes:
            image = inp[f"video.observation.images.rgb.{camera}_256_256"][0,0]
            camera_hashes[camera].append(hashlib.sha256(image.tobytes()).hexdigest())
        for j in range(8):
            step = request*8+j
            z = np.load(out/f"step{step:04d}.npz")
            for k in groups:
                delta("wrapper_to_step",net["wrapped/action."+k][0,j],z["policy_action/action."+k])
            expected = np.concatenate([z["policy_action/action."+k] for k in groups])
            delta("adapter_to_env",expected,z["env_action"])
            if previous_after is not None:
                delta("between_step_qpos_continuity",previous_after,z["joint_before"])
            previous_after = z["joint_after"].copy()
            for k,idx in indices.items():
                target = z["controller/"+controllers[k]+"/target"]
                delta("goal_vs_arm_torso_command",target,z["policy_action/action."+k])
                e = np.abs(target-z["joint_after"][idx])
                tracking[k].extend(e.tolist())
            base = z["policy_action/action.base"]
            goal = z["controller/base/target"]
            delta("base_scaled_speed_norm",np.linalg.norm(np.clip(base[:2],-1,1)*.75),np.linalg.norm(goal[:2]))
            delta("base_yaw_command",np.clip(base[2],-1,1),goal[2])
            for arm in ("left","right"):
                command = z["policy_action/action."+arm+"_gripper"]
                goal = z["controller/gripper_"+arm+"/target"]
                # Actual runtime controller_config reports finger limits [0, .05].
                delta("gripper_goal_mapping",(.025*(np.clip(command,-1,1)+1)),goal)
                gripper_values[arm].extend(command.tolist())
            state_before = z["raw_proprio_after"].copy()
            actions_all.append(z["env_action"])
            z.close()
        inp.close();result.close();net.close()
    actions_all = np.stack(actions_all)
    for k,sl in {"base":slice(0,3),"torso":slice(3,7),"left_arm":slice(7,14),"left_gripper":slice(14,15),"right_arm":slice(15,22),"right_gripper":slice(22,23)}.items():
        x=actions_all[:,sl]
        ranges[k]={"min":float(x.min()),"max":float(x.max()),"max_adjacent_step_jump":float(np.abs(np.diff(x,axis=0)).max())}
    rows=[json.loads(line) for line in (out/"steps.jsonl").read_text().splitlines()]
    assert len(rows)==400 and len(actions_all)==400
    report={"steps":400,"requests":50,"all_executed_actions_finite":bool(np.isfinite(actions_all).all()),
        "boundary_max_abs_delta":comparisons,"first_real_input_fields":fields,
        "camera_unique_frames_per_50_requests":{k:len(set(v)) for k,v in camera_hashes.items()},
        "controller_tracking_absolute_position_error_rad":{k:{"max":float(np.max(v)),"p95":float(np.percentile(v,95))} for k,v in tracking.items()},
        "executed_action_ranges":ranges,"closed_gripper_command_steps":{k:int(np.sum(np.array(v)<0)) for k,v in gripper_values.items()},
        "physx_crashed_steps":[r["step"] for r in rows if r["physx_crashed"]],
        "maximum_single_step_joint_change":max(rows,key=lambda r:r["joint_max_abs_change"]),
        "frame_timestamp_capture":False,"demonstration_action_replay_performed":False}
    first_server = np.load(server/"request000.npz")
    first_step = np.load(out/"step0000.npz")
    report["first_action_boundaries"] = {
        "raw_first23":first_server["raw_action"][0,0,:23].tolist(),
        "processor_decoded":{k:first_server["decoded/"+k][0,0].tolist() for k in groups},
        "wrapper":{k:first_server["wrapped/action."+k][0,0].tolist() for k in groups},
        "environment_received":first_step["env_action"].tolist(),
        "controller_targets":{k:first_step[k].tolist() for k in first_step.files if k.startswith("controller/")}}
    first_server.close();first_step.close()
    save_json(out/"analysis.json",report)
    print(json.dumps(report,indent=2))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=["serve", "simulate", "analyze"])
    parser.add_argument("--source", required=True)
    parser.add_argument("--checkpoint")
    parser.add_argument("--behavior", default="/nvme3/projects/GR00T/BEHAVIOR-1K-task-progress")
    parser.add_argument("--host", default="172.28.62.17")
    parser.add_argument("--port", type=int, default=5568, choices=[5568])
    parser.add_argument("--output", required=True)
    parser.add_argument("--server", help="saved private-server request directory for analyze")
    args = parser.parse_args()
    if args.mode == "serve":
        serve(args)
    elif args.mode == "simulate":
        simulate(args)
    else:
        analyze(args)


if __name__ == "__main__":
    main()
