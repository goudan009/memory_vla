"""Finished episode videos should be visible while the next episode runs."""

import importlib.util
import json
from pathlib import Path
import threading
from urllib.error import HTTPError
from urllib.parse import urlencode
from urllib.request import Request, urlopen

import pytest


ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("live_video_viewer", ROOT / "evaluation/scripts/view_results.py")
viewer = importlib.util.module_from_spec(spec)
spec.loader.exec_module(viewer)


def run_bundle(root, attempt="gpu0"):
    run = root / "V_1/baseline/groot_task21_1"
    run.mkdir(parents=True)
    (run / "results.json").write_text(json.dumps({"status": "running", "evaluation_mode": "official-eval"}))
    runtime = run / ".runtime" / attempt / "groot_n1d6_task021_run" / "videos"
    runtime.mkdir(parents=True)
    return run, runtime


@pytest.mark.parametrize("attempt", ["gpu0", "gpu1_resume2"])
def test_running_bundle_lists_sealed_videos_only_and_keeps_paths_after_archive(tmp_path, attempt):
    run, runtime = run_bundle(tmp_path, attempt)
    closed = runtime / "finished_s0.mp4"
    closed.write_bytes(b"sealed-video")
    (runtime / "recording.mp4").write_bytes(b"still-recording")
    (runtime / "recording_something.mp4").write_bytes(b"still-recording")
    logical = f"videos/{attempt}/{closed.name}"
    assert viewer.load_runs(tmp_path)[0]["_videos"] == [logical]
    destination = run / logical
    destination.parent.mkdir(parents=True)
    closed.rename(destination)
    assert viewer.load_runs(tmp_path)[0]["_videos"] == [logical]


def test_video_identity_remains_available_for_episodes_older_than_recent_table(tmp_path):
    run, runtime = run_bundle(tmp_path)
    closed = runtime / "episode_001_instance0_seed0_gpu0_env0_s0.mp4"
    closed.write_bytes(b"sealed-video")
    logical = f"videos/gpu0/{closed.name}"
    records = [{"episode_index": index, "instance_id": index - 1, "seed": index - 1,
                "env_idx": 0, "gpu_index": 0, "valid": True, "success": False,
                "video_file": logical if index == 1 else ""}
               for index in range(1, 26)]
    (run / "episodes.jsonl").write_text("".join(json.dumps(record) + "\n" for record in records))
    bundle = viewer.load_runs(tmp_path)[0]
    assert len(bundle["_episodes"]) == 25  # Every episode remains selectable.
    assert bundle["_episodes"][0]["episode_index"] == 1
    assert bundle["_video_episodes"][logical] == records[0]
    assert bundle["_videos"] == [logical]


def test_runtime_video_supports_range_requests_and_archive_without_exposing_recording(tmp_path):
    run, runtime = run_bundle(tmp_path)
    closed = runtime / "finished_s1.mp4"
    closed.write_bytes(b"0123456789-sealed-video")
    (runtime / "recording.mp4").write_bytes(b"still-recording")
    class Handler(viewer.Handler):
        root = tmp_path

    server = viewer.ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    def url(path):
        return f"http://127.0.0.1:{server.server_port}/video?" + urlencode({
            "run": run.relative_to(tmp_path).as_posix(), "path": path})
    logical = "videos/gpu0/finished_s1.mp4"
    try:
        with urlopen(Request(url(logical), headers={"Range": "bytes=2-5"}), timeout=2) as response:
            assert response.status == 206
            assert response.headers["Content-Range"] == "bytes 2-5/23"
            assert response.read() == b"2345"
        for hidden in ("videos/gpu0/recording.mp4", "../outside.mp4"):
            with pytest.raises(HTTPError) as error:
                urlopen(url(hidden), timeout=2)
            assert error.value.code == 404
        destination = run / logical
        destination.parent.mkdir(parents=True)
        closed.rename(destination)
        with urlopen(url(logical), timeout=2) as response:
            assert response.status == 200
            assert response.read() == b"0123456789-sealed-video"
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=2)
