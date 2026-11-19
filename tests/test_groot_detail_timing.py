"""Verify nested accounting without launching Isaac Sim."""

import importlib.util
from pathlib import Path
import sys
from types import SimpleNamespace

def test_nested_wall_times_and_chunk_positions(monkeypatch):
    path = Path(__file__).resolve().parents[1] / "scripts/groot_eval_timing.py"
    spec = importlib.util.spec_from_file_location("detail_timing_test", path)
    timing = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(timing)
    monkeypatch.setattr(timing, "_DETAIL_ENABLED", True)
    clock = iter([0.0, 1.0, 3.0, 5.0])
    monkeypatch.setattr(timing.time, "perf_counter", lambda: next(clock))
    timing.start_detail_cycle(True)
    with timing.profile_scope("parent"):
        timing.set_chunk_position(7)
        with timing.profile_scope("camera"):
            pass
    timing.start_detail_cycle(False)
    with timing.profile_scope("ignored_reset"):
        pass
    snapshot = timing.detail_snapshot()
    parent = snapshot["stages"]["parent"]
    camera = snapshot["stages"]["camera"]
    assert parent["inclusive_total_s"] == 5.0
    assert parent["self_total_s"] == 3.0
    assert camera["self_total_s"] == 2.0
    assert parent["self_total_s"] + camera["self_total_s"] == parent["inclusive_total_s"]
    assert snapshot["by_chunk_position"]["7"]["camera"]["count"] == 1
    assert "ignored_reset" not in snapshot["stages"]
    assert timing._STACK == []


def test_physics_callback_interval_is_nested_and_keeps_behavior(monkeypatch):
    scripts = Path(__file__).resolve().parents[1] / "scripts"
    spec = importlib.util.spec_from_file_location("groot_eval_timing", scripts / "groot_eval_timing.py")
    timing = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(timing)
    monkeypatch.setitem(sys.modules, "groot_eval_timing", timing)
    monkeypatch.setattr(timing, "_DETAIL_ENABLED", True)
    monkeypatch.setenv("GROOT_EVAL_SIM_PROFILE", "1")
    spec = importlib.util.spec_from_file_location("sim_timing_test", scripts / "groot_sim_timing.py")
    sim = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(sim)
    clock = iter(range(10))
    monkeypatch.setattr(timing.time, "perf_counter", lambda: next(clock))
    calls = []

    @sim.physics_pre
    def before():
        calls.append("pre")
        return 42

    @sim.physics_post
    def after():
        calls.append("post")

    timing.start_detail_cycle(True)
    with sim.sim_scope("kit"):
        assert before() == 42
        with sim.sim_scope("contact"):
            calls.append("contact")
        after()
    timing.start_detail_cycle(False)
    snapshot = timing.detail_snapshot()
    stages = snapshot["stages"]
    assert calls == ["pre", "contact", "post"]
    assert stages["physics.native_interval"]["inclusive_total_s"] == 3
    assert stages["physics.native_interval"]["self_total_s"] == 2
    assert sum(v["self_total_s"] for v in stages.values()) == stages["kit"]["inclusive_total_s"]
    assert sim._PHYSICS_SPANS == {}
    assert timing._STACK == []
    assert "physics.unpaired_pre" not in snapshot["counters"]


def test_existing_camera_graph_counters_are_read_at_window_boundaries(monkeypatch):
    path = Path(__file__).resolve().parents[1] / "scripts/groot_eval_timing.py"
    spec = importlib.util.spec_from_file_location("camera_timing_test", path)
    timing = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(timing)
    monkeypatch.setattr(timing, "_DETAIL_ENABLED", True)
    monkeypatch.setenv("GROOT_EVAL_CAMERA_COUNTERS", "1")
    counts = iter([17, 1217])
    node = SimpleNamespace(get_compute_count=lambda: next(counts), get_prim_path=lambda: "/rgb")
    sensor = SimpleNamespace(name="head", image_width=256, image_height=256,
                             _annotators={"rgb": SimpleNamespace(get_node=lambda: node)})
    module = SimpleNamespace(VisionSensor=SimpleNamespace(SENSORS={"head": sensor}))
    monkeypatch.setitem(sys.modules, "omnigibson.sensors.vision_sensor", module)
    stream = SimpleNamespace(create_subscription_to_pop=lambda *args, **kwargs: object())
    lazy = SimpleNamespace(omni=SimpleNamespace(usd=SimpleNamespace(
        get_context=lambda: SimpleNamespace(get_rendering_event_stream=lambda: stream))))
    monkeypatch.setitem(sys.modules, "omnigibson", SimpleNamespace(lazy=lazy))
    monkeypatch.setitem(sys.modules, "omnigibson.lazy", lazy)
    timing.start_detail_cycle(True)
    timing.start_detail_cycle(False)
    timing.start_detail_cycle(True)
    timing.start_detail_cycle(False)
    result = timing.detail_snapshot()["camera_graph_counters"]
    assert result[0]["compute_count_delta"] == 1200
    assert result[0]["compute_count_before"] == 17
    timing.start_detail_cycle(True)
    # Kit payloads are carb.dictionary.Item, whose mapping is exposed by get_dict().
    payload = SimpleNamespace(get_dict=lambda: {"renderProductPath": "/Render/Product"})
    timing._audit_render_event(SimpleNamespace(type=0, payload=payload))
    assert timing._COUNTERS["camera_audit.render_event.0"] == 1
    assert timing._METADATA["render_event_examples.0"][0]["renderProductPath"] == "/Render/Product"
