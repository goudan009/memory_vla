import importlib.util
from pathlib import Path
from types import SimpleNamespace

spec = importlib.util.spec_from_file_location("task40_trace", Path(__file__).resolve().parents[1] /
    "evaluation/scripts/groot_task40_trace.py")
trace = importlib.util.module_from_spec(spec)
spec.loader.exec_module(trace)


def test_stage_reader_reports_actual_open_inside_and_grasp_without_changing_state():
    class Value:
        def __init__(self, value): self.value = value
        def get_value(self, *args): return self.value
    states = {name: object() for name in ("Open", "ToggledOn", "Inside", "Cooked")}
    microwave = SimpleNamespace(category="microwave", name="micro", states={
        states["Open"]: Value(False), states["ToggledOn"]: Value(False)})
    bag = SimpleNamespace(category="popcorn_bag", name="bag", states={
        states["Inside"]: Value(False), states["Cooked"]: Value(False)})
    env = SimpleNamespace(scene=SimpleNamespace(objects=[microwave, bag]), robots=[
        SimpleNamespace(name="r1", _ag_obj_in_hand={"left": bag, "right": None})])
    row = trace.read_state(env, states)
    assert row["microwaves"]["micro"]["Open"] is False
    assert row["popcorn"]["bag"]["inside"]["micro"] is False
    assert row["grasped"]["r1/left"] == "bag"
    assert microwave.states[states["Open"]].value is False


def test_extra_robot_pose_and_joint_facts_are_read_only():
    robot=SimpleNamespace(name='r1',arm_names=['left'],
        get_position_orientation=lambda:([1,2,3],[0,0,0,1]),
        get_joint_positions=lambda:[.1,.2],get_eef_pose=lambda arm:([4,5,6],[0,0,0,1]))
    states={name:object() for name in ('Open','ToggledOn','Inside','Cooked')}
    row=trace.read_state(SimpleNamespace(scene=SimpleNamespace(objects=[]),robots=[robot]),states)
    assert row['robots']['r1']['position']==[1,2,3]
    assert row['robots']['r1']['end_effectors']['left'][0]==[4,5,6]
    assert trace.serializable({'action':([1,2],[3,4])})=={'action':[[1,2],[3,4]]}


def test_recorded_frame_index_keeps_start_offset_and_resets_per_episode(tmp_path, monkeypatch):
    import json
    import sys
    sim = SimpleNamespace(current_time=10.)
    monkeypatch.setitem(sys.modules, 'omnigibson', SimpleNamespace(sim=sim))
    writes = []
    recorder = SimpleNamespace(fps=20, write_frame=lambda image: writes.append(image))
    wrapper = SimpleNamespace(video_recorder=recorder, step_count=24)
    identity = dict(instance_id=4, seed=22004, env_idx=0, episode_number=1)
    trace.install_video_trace(wrapper, tmp_path, identity, 0)
    image = object()
    recorder.write_frame(image)
    wrapper.step_count = 48
    sim.current_time = 10.8
    recorder.write_frame(image)
    next_identity = dict(identity, seed=22005, episode_number=2)
    trace.install_video_trace(wrapper, tmp_path, next_identity, 0)
    wrapper.step_count = 24
    recorder.write_frame(image)
    rows = [json.loads(line) for line in (tmp_path/'video_gpu0_env0.jsonl').read_text().splitlines()]
    assert [row['frame_index'] for row in rows] == [0, 1, 0]
    assert [row['action_steps'] for row in rows] == [23, 47, 23]
    assert [row['sim_time_s'] for row in rows] == [10., 10.8, 10.8]
    assert [row['episode_number'] for row in rows] == [1, 1, 2]
    assert all(row['video_fps'] == 20 for row in rows)
    assert writes == [image, image, image]


def test_policy_input_roundtrips_without_rgb_or_proprio_conversion(tmp_path):
    import json

    import numpy as np

    observation = {f"video.{camera}": np.arange(24, dtype=np.uint8).reshape(1, 1, 2, 4, 3)
                   for camera in ("left_wrist", "right_wrist", "head")}
    observation["state.arm"] = np.array([[[0.125, -0.375]]], dtype=np.float32)
    observation["annotation.human.action.task_description"] = ["Make microwave popcorn."]
    decision_dir = tmp_path / "0000"
    language = json.loads(json.dumps(trace.save_policy_input(decision_dir, observation)))
    with np.load(decision_dir / "observation.npz", allow_pickle=False) as archive:
        restored = {key: archive[key] for key in archive.files} | language
    assert restored.keys() == observation.keys()
    for key, value in observation.items():
        if key.startswith("annotation."):
            assert restored[key] == value
        else:
            assert restored[key].dtype == value.dtype
            np.testing.assert_array_equal(restored[key], value)
