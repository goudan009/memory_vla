"""Preview leases, real JPEGs, observation identity and the LAN HTTP API."""

import base64
import importlib.util
import json
from pathlib import Path
import random
import sys
import threading
from urllib.error import HTTPError
from urllib.parse import urlencode
from urllib.request import Request, urlopen

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "evaluation/scripts"))
import groot_live_preview as preview
import groot_eval_results as results

spec = importlib.util.spec_from_file_location("preview_viewer", ROOT / "evaluation/scripts/view_results.py")
viewer = importlib.util.module_from_spec(spec)
spec.loader.exec_module(viewer)


def bundle(root, name="run"):
    run = root / name
    outdir = run / ".runtime/gpu0/task21"
    outdir.mkdir(parents=True)
    metadata = {"public_output_dir": str(run), "gpu_index": 0, "task_id": 21,
                "task_name": "collecting_childrens_toys", "episodes": 10}
    (run / "results.json").write_text(json.dumps({"status": "running", "evaluation_mode": "official-eval"}))
    (outdir / "run_meta.json").write_text(json.dumps(metadata))
    return run, outdir, metadata


def observations():
    np = pytest.importorskip("numpy")
    return {key: np.stack([np.full((32, 32, 3), [0, 0, 255], dtype=np.uint8),
                          np.full((32, 32, 3), [255, 0, 0], dtype=np.uint8)])
            for _, _, key in preview.CAMERAS}


def test_only_interested_viewers_enable_encoding_and_one_client_cannot_disable_another(tmp_path):
    run, _, _ = bundle(tmp_path)
    preview.subscribe(run, "client_one", True, now=100)
    preview.subscribe(run, "client_two", True, now=101)
    assert preview.requested(run, now=110)
    preview.subscribe(run, "client_one", False)
    assert preview.requested(run, now=110)
    assert not preview.requested(run, now=117)
    preview.subscribe(run, "client_two", False)
    assert not preview.requested(run, now=100)
    with pytest.raises(ValueError):
        preview.subscribe(run, "../../outside", True)


def test_snapshots_are_throttled_take_latest_rgb_and_leave_observations_and_rng_unchanged(tmp_path, monkeypatch):
    np = pytest.importorskip("numpy")
    cv2 = pytest.importorskip("cv2")
    run, outdir, metadata = bundle(tmp_path)
    publisher = preview.PreviewPublisher(outdir, metadata, 0)
    identity = {"instance_id": 4, "seed": 7, "episode_number": 1}
    publisher.reset(identity)
    obs = observations()
    originals = {key: value.copy() for key, value in obs.items()}
    monotonic = [10.0]
    monkeypatch.setattr(preview.time, "monotonic", lambda: monotonic[0])
    publisher.publish(obs, action_steps=8, outer_steps=1)
    assert preview.read_preview(run)["environments"][0]["cameras"] == []
    preview.subscribe(run, "client_one", True)
    monotonic[0] = 11.1
    python_rng, numpy_rng = random.getstate(), np.random.get_state()
    publisher.publish(obs, action_steps=16, outer_steps=2)
    first = preview.read_preview(run)["environments"][0]
    assert first["action_steps"] == 16
    assert {key: first[key] for key in identity} == identity
    assert [camera["name"] for camera in first["cameras"]] == ["head", "left_wrist", "right_wrist"]
    for camera in first["cameras"]:
        jpeg = base64.b64decode(camera["src"].split(",")[1])
        image = cv2.imdecode(np.frombuffer(jpeg, dtype=np.uint8), cv2.IMREAD_COLOR)
        assert image.shape == (32, 32, 3)
        # RGB red must stay red, rather than becoming blue or using the old frame.
        assert image[0, 0, 2] > 240 and image[0, 0, 0] < 10
    for key in obs:
        np.testing.assert_array_equal(obs[key], originals[key])
    assert random.getstate() == python_rng
    after = np.random.get_state()
    assert numpy_rng[0] == after[0] and numpy_rng[2:] == after[2:]
    np.testing.assert_array_equal(numpy_rng[1], after[1])
    monotonic[0] = 11.2
    publisher.publish(obs, action_steps=24, outer_steps=3)
    assert preview.read_preview(run)["environments"][0] == first
    preview.subscribe(run, "client_one", False)
    monotonic[0] = 12.3
    publisher.publish(obs, action_steps=32, outer_steps=4)
    assert preview.read_preview(run)["environments"][0] == first
    publisher.reset({"instance_id": 8, "seed": 9, "episode_number": 2})
    reset = preview.read_preview(run)["environments"][0]
    assert reset["instance_id"] == 8 and reset["captured_at"] is None and reset["cameras"] == []
    publisher.close()
    assert preview.read_preview(run)["environments"][0]["status"] == "closed"


def test_gpus_environments_and_runs_have_separate_snapshots(tmp_path):
    run, outdir, metadata = bundle(tmp_path)
    other, otherdir, othermeta = bundle(tmp_path, "other")
    for gpu, env_idx in [(0, 0), (0, 1), (1, 0), (1, 1)]:
        publisher = preview.PreviewPublisher(outdir, {**metadata, "gpu_index": gpu}, env_idx)
        publisher.reset({"instance_id": gpu * 2 + env_idx, "seed": env_idx, "episode_number": 1})
    preview.PreviewPublisher(otherdir, othermeta, 0)
    values = preview.read_preview(run)["environments"]
    assert {(value["gpu_index"], value["env_idx"]) for value in values} == {(0, 0), (0, 1), (1, 0), (1, 1)}
    assert len(preview.read_preview(other)["environments"]) == 1
    assert not preview.read_preview(tmp_path / "old-run")["supported"]


def test_preview_encoding_error_does_not_interrupt_episode(tmp_path, monkeypatch):
    run, outdir, metadata = bundle(tmp_path)
    publisher = preview.PreviewPublisher(outdir, metadata, 0)
    preview.subscribe(run, "client_one", True)
    def failing_encoder(*args):
        raise RuntimeError("JPEG encoder unavailable")
    monkeypatch.setattr(publisher, "_publish_frame", failing_encoder)
    publisher.publish({}, action_steps=8, outer_steps=1)
    assert preview.read_preview(run)["environments"][0]["status"] == "error"


def test_http_toggle_and_read_preview_are_scoped_to_a_real_run(tmp_path):
    run, outdir, metadata = bundle(tmp_path)
    preview.PreviewPublisher(outdir, metadata, 0)
    class Handler(viewer.Handler):
        root = tmp_path

    server = viewer.ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    base = f"http://127.0.0.1:{server.server_port}/api/preview"
    def toggle(name="run", enabled=True):
        data = json.dumps({"run": name, "client": "client_one", "enabled": enabled}).encode()
        return urlopen(Request(base, data=data, headers={"Content-Type": "application/json"}), timeout=2)
    try:
        with toggle() as response:
            assert response.status == 204
        assert preview.requested(run)
        with urlopen(base + "?" + urlencode({"run": "run"}), timeout=2) as response:
            data = json.load(response)
            assert data["supported"] and len(data["environments"]) == 1
            assert response.headers["Cache-Control"] == "no-store"
        with toggle(enabled=False) as response:
            assert response.status == 204
        assert not preview.requested(run)
        for name in ("../outside", "missing", ""):
            with pytest.raises(HTTPError) as error:
                toggle(name=name)
            assert error.value.code == 404
        (run / "results.json").write_text(json.dumps({"status": "complete"}))
        with pytest.raises(HTTPError) as error:
            toggle()
        assert error.value.code == 400
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=2)


def test_recording_wrapper_publishes_without_extra_env_calls_or_altering_returned_observations(tmp_path):
    gym = pytest.importorskip("gymnasium")
    np = pytest.importorskip("numpy")
    pytest.importorskip("cv2")
    run, outdir, metadata = bundle(tmp_path)
    preview.subscribe(run, "client_one", True)

    class TinyEnv(gym.Env):
        observation_space = gym.spaces.Dict({key: gym.spaces.Box(0, 255, (2, 32, 32, 3), np.uint8)
                                             for _, _, key in preview.CAMERAS})
        action_space = gym.spaces.Box(0, 1, (1,), np.float32)
        def __init__(self):
            self._instance_indices_this_env = [0]
            self._all_test_instances = [7]
            self._instance_idx_pointer = 1
            self.obs = observations()
            self.step_calls = self.reset_calls = 0
        def reset(self, **kwargs):
            self.reset_calls += 1
            return self.obs, {}
        def step(self, action):
            self.step_calls += 1
            return self.obs, 0, False, False, {"rewards": [0] * 8, "success": False}
        def render(self):
            raise AssertionError("Preview must not render")
        def get_obs(self):
            raise AssertionError("Preview must reuse observations")

    inner = TinyEnv()
    env = results.record_episode_env(lambda: inner, outdir, env_idx=0)
    try:
        obs, _ = env.reset()
        assert obs is inner.obs
        assert len(preview.read_preview(run)["environments"][0]["cameras"]) == 3
        obs, reward, terminated, truncated, info = env.step(np.zeros(1))
        assert obs is inner.obs and reward == 0 and not terminated and not truncated
        assert inner.reset_calls == inner.step_calls == 1
        assert env.sim_steps == 8
    finally:
        env.close()


def test_spawned_vector_workers_publish_independent_camera_snapshots(tmp_path):
    gym = pytest.importorskip("gymnasium")
    np = pytest.importorskip("numpy")
    pytest.importorskip("cv2")
    run, outdir, metadata = bundle(tmp_path)
    preview.subscribe(run, "client_one", True)

    class WorkerEnv(gym.Env):
        observation_space = gym.spaces.Dict({key: gym.spaces.Box(0, 255, (1, 16, 16, 3), np.uint8)
                                             for _, _, key in preview.CAMERAS})
        action_space = gym.spaces.Box(0, 1, (1,), np.float32)
        def __init__(self, instance):
            self._instance_indices_this_env = [0]
            self._all_test_instances = [instance]
            self._instance_idx_pointer = 1
            self.obs = {key: np.full((1, 16, 16, 3), instance, dtype=np.uint8)
                        for _, _, key in preview.CAMERAS}
        def reset(self, **kwargs):
            return self.obs, {}
        def step(self, action):
            return self.obs, 0, False, False, {"rewards": np.zeros(8), "success": False}

    writer = results.EpisodeWriter(outdir)
    env = gym.vector.AsyncVectorEnv(writer.wrap_env_fns([
        lambda: WorkerEnv(0), lambda: WorkerEnv(5),
    ]), context="spawn")
    processes = list(env.processes)
    try:
        env.reset()
        env.step(np.zeros((2, 1), dtype=np.float32))
        snapshots = preview.read_preview(run)["environments"]
        assert [(value["env_idx"], value["instance_id"], value["seed"])
                for value in snapshots] == [(0, 0, 0), (1, 5, 1)]
        assert all(len(value["cameras"]) == 3 for value in snapshots)
        assert snapshots[0]["cameras"][0]["src"] != snapshots[1]["cameras"][0]["src"]
    finally:
        env.close(timeout=10)
    assert all(not process.is_alive() for process in processes)
