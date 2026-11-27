"""Exercise the read-only migration audit against real Parquet and MP4 files."""
import importlib.util
import json
from pathlib import Path

import pytest

np = pytest.importorskip("numpy")
pa = pytest.importorskip("pyarrow")
pq = pytest.importorskip("pyarrow.parquet")
av = pytest.importorskip("av")

spec = importlib.util.spec_from_file_location(
    "audit_v12_task40", Path(__file__).resolve().parents[1] / "scripts/audit_v12_task40.py")
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)
from v1_2_groot_baseline.data_contract import CAMERAS, STATE_WIDTHS
from v1_2_groot_baseline.contracts import ACTION_GROUPS


@pytest.fixture
def dataset(tmp_path):
    def write(relative, value):
        path = tmp_path / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(value))

    state_layout = {}
    offset = 0
    for key, width in STATE_WIDTHS.items():
        state_layout[key] = dict(start=offset, end=offset + width)
        offset += width
    write("meta/modality.json", dict(
        state=state_layout,
        action={k: dict(start=a, end=b) for k, (a, b) in ACTION_GROUPS.items()},
        video={k: dict(original_key=k) for k in CAMERAS},
        annotation={"human.coarse_action": dict(original_key="annotation.human.coarse_action")}))
    write("meta/info.json", dict(
        fps=30, features={"observation.state": dict(shape=[82]), "action": dict(shape=[23])},
        splits=dict(train_episode_ids=[400001], validation_episode_ids=[400002]),
        chunks_size=1000, total_episodes=2, total_frames=6, total_videos=6,
        video_path="videos/chunk-{episode_chunk:03d}/{video_key}/episode_{episode_index:08d}.mp4"))
    (tmp_path / "meta/episodes.jsonl").write_text("\n".join(
        json.dumps(dict(episode_index=eid, length=3)) for eid in (400001, 400002)))
    write("meta/tasks.jsonl", dict(task_index=40, task_name="make_microwave_popcorn",
                                  task="Make microwave popcorn."))
    for eid in (400001, 400002):
        state = np.zeros((3, 82), dtype=np.float32)
        state[:, 32:39] = state[:, 62:69] = 1  # cos(qpos)
        state[:, 45] = state[:, 75] = 1  # unit XYZW quaternions
        table = pa.table({
            "frame_index": [0, 1, 2], "episode_index": [eid] * 3,
            "task_index": [40] * 3, "timestamp": [0, 1 / 30, 2 / 30],
            "annotation.human.coarse_action": [40] * 3,
            "observation.state": state.tolist(), "action": np.zeros((3, 23)).tolist()})
        path = tmp_path / f"data/chunk-400/episode_{eid:08d}.parquet"
        path.parent.mkdir(parents=True, exist_ok=True)
        pq.write_table(table, path)
        for camera in CAMERAS:
            video = tmp_path / f"videos/chunk-400/{camera}/episode_{eid:08d}.mp4"
            video.parent.mkdir(parents=True, exist_ok=True)
            with av.open(str(video), "w") as container:
                stream = container.add_stream("mpeg4", rate=30)
                stream.width = stream.height = 256
                stream.pix_fmt = "yuv420p"
                for index in range(3):
                    frame = av.VideoFrame.from_ndarray(
                        np.full((256, 256, 3), index * 30, dtype=np.uint8), format="rgb24")
                    for packet in stream.encode(frame):
                        container.mux(packet)
                for packet in stream.encode():
                    container.mux(packet)
    return tmp_path


def test_real_files_pass_without_mutation(dataset):
    before = {p: (p.stat().st_size, p.stat().st_mtime_ns) for p in dataset.rglob("*") if p.is_file()}
    result = module.audit(dataset)
    assert result["status"] == "PASS"
    assert (result["rows"], result["video_headers"], result["decoded_frames"]) == (6, 6, 18)
    assert result["native_train_ready"] is False
    assert before == {p: (p.stat().st_size, p.stat().st_mtime_ns) for p in before}


def test_misaligned_physical_timestamps_rejected(dataset):
    path = dataset / "data/chunk-400/episode_00400001.parquet"
    table = pq.read_table(path)
    table = table.set_column(table.schema.get_field_index("timestamp"), "timestamp", pa.array([0, 0.1, 0.2]))
    pq.write_table(table, path)
    with pytest.raises(AssertionError):
        module.audit(dataset)


def test_overlapping_original_split_rejected(dataset):
    path = dataset / "meta/info.json"
    info = json.loads(path.read_text())
    info["splits"]["validation_episode_ids"] = [400001, 400002]
    path.write_text(json.dumps(info))
    with pytest.raises(ValueError, match="partition"):
        module.audit(dataset)


def test_missing_physical_episode_rejected(dataset):
    (dataset / "data/chunk-400/episode_00400002.parquet").unlink()
    with pytest.raises(ValueError, match="physical parquet"):
        module.audit(dataset)
