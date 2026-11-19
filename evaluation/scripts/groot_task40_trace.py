"""Read-only task40 event traces; never used as policy input or for scoring."""
import json
import uuid


def serializable(value):
    if hasattr(value, 'detach'):
        value = value.detach().cpu()
    if hasattr(value, 'tolist'):
        return value.tolist()
    if isinstance(value, dict):
        return {str(k):serializable(v) for k,v in value.items()}
    if isinstance(value, (list,tuple)):
        return [serializable(v) for v in value]
    return value


def read_state(env, states):
    objects = env.scene.objects
    microwaves = [x for x in objects if x.category == "microwave"]
    bags = [x for x in objects if x.category == "popcorn_bag"]
    row = {"microwaves": {}, "popcorn": {}, "grasped": {}, "grasp_modes": {}, "grasp_states": {}, "robots": {}}
    for obj in microwaves:
        row["microwaves"][obj.name] = {
            name: bool(obj.states[states[name]].get_value())
            for name in ("Open", "ToggledOn") if states[name] in obj.states}
        if hasattr(obj, "get_joint_positions"):
            row["microwaves"][obj.name]["joint_positions"] = obj.get_joint_positions().tolist()
    for obj in bags:
        item = {"inside": {target.name: bool(obj.states[states["Inside"]].get_value(target))
                           for target in microwaves} if states["Inside"] in obj.states else {}}
        if states["Cooked"] in obj.states:
            item["cooked"] = bool(obj.states[states["Cooked"]].get_value())
        if hasattr(obj, "get_position_orientation"):
            item["position"] = obj.get_position_orientation()[0].tolist()
        row["popcorn"][obj.name] = item
    for robot in env.robots:
        item = {}
        if hasattr(robot, 'get_position_orientation'):
            position, orientation = robot.get_position_orientation()
            item.update(position=serializable(position),orientation=serializable(orientation))
        if hasattr(robot, 'get_joint_positions'):
            item['joint_positions'] = serializable(robot.get_joint_positions())
        if hasattr(robot, 'get_eef_pose'):
            item['end_effectors'] = {arm:serializable(robot.get_eef_pose(arm=arm))
                                     for arm in getattr(robot, 'arm_names', [])}
        row['robots'][robot.name] = item
        row["grasp_modes"][robot.name] = getattr(robot, "grasping_mode", "unreported")
        for arm, held in getattr(robot, "_ag_obj_in_hand", {}).items():
            row["grasped"][robot.name + "/" + arm] = held.name if held is not None else None
            if hasattr(robot, "is_grasping"):
                for bag in bags:
                    value = robot.is_grasping(arm=arm, candidate_obj=bag)
                    row["grasp_states"][robot.name + "/" + arm + "/" + bag.name] = getattr(value, "name", str(value))
    return row


def append_trace(env, path, identity, steps):
    import omnigibson as og
    import omnigibson.object_states as state_module
    states = {name: getattr(state_module, name) for name in ("Open", "ToggledOn", "Inside", "Cooked")}
    record = dict(identity, schema_version=2, action_steps=steps,
                  sim_time_s=float(og.sim.current_time), **read_state(env, states))
    with path.open("a") as stream:
        stream.write(json.dumps(record) + "\n")


def append_action(path, identity, start, end, action):
    import omnigibson as og
    record = dict(identity, schema_version=1, action_start=start, action_end=end,
                  sim_time_s=float(og.sim.current_time), action=serializable(action),
                  coverage='每次外层策略调用的完整action；物理状态另按stride采样')
    with path.open('a') as stream:
        stream.write(json.dumps(record)+'\n')


def save_policy_input(directory, observation):
    import numpy as np

    directory.mkdir(parents=True)
    language = {key: value for key, value in observation.items() if key.startswith("annotation.")}
    np.savez_compressed(directory / "observation.npz",
                        **{key: value for key, value in observation.items() if key not in language})
    return language


def install_video_trace(env, root, identity, gpu_index):
    """Observe actual encoded frames without changing camera images or stepping."""
    import omnigibson as og
    current = env
    while current is not None:
        state = vars(current)
        recorder = state.get('video_recorder')
        if recorder is not None:
            # reset() reuses the recorder; do not accumulate nested callbacks.
            original = getattr(recorder, '_task40_original_write_frame', recorder.write_frame)
            recorder._task40_original_write_frame = original
            root.mkdir(parents=True, exist_ok=True)
            path = root / f"video_gpu{gpu_index}_env{identity['env_idx']}.jsonl"
            frame_index, recording_id = 0, uuid.uuid4().hex

            def write_frame(image, *args, **kwargs):
                nonlocal frame_index
                # The fixed wrapper increments step_count before encoding and
                # initializes it to 1. Thus the first stride-24 frame is step23.
                record = dict(identity, schema_version=1, recording_id=recording_id,
                    frame_index=frame_index, action_steps=current.step_count-1,
                    sim_time_s=float(og.sim.current_time), video_fps=recorder.fps)
                result = original(image, *args, **kwargs)
                with path.open('a') as stream:
                    stream.write(json.dumps(record)+'\n')
                frame_index += 1
                return result

            recorder.write_frame = write_frame
            return
        current = state.get('env')
