"""Read-only B1K2025 Task40 migration audit; prints JSON, never trains.

Run with a CPU Python containing numpy, pyarrow and av. This checks the
existing files, not the N1.7 loader, processor, model or simulator runtime.
"""
import argparse
import hashlib
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from v1_2_groot_baseline.data_contract import CAMERAS, audit_2025


def audit(dataset, *, video_sample_episodes=3):
    import av
    import numpy as np
    import pyarrow.parquet as pq

    dataset = Path(dataset).resolve(strict=True)
    contract = audit_2025(dataset)
    info = json.loads((dataset / "meta/info.json").read_text())
    episodes = [json.loads(s) for s in (dataset / "meta/episodes.jsonl").read_text().splitlines() if s.strip()]
    tasks = {t["task_index"]: t for t in map(json.loads, (dataset / "meta/tasks.jsonl").read_text().splitlines())}
    expected = {e["episode_index"]: e for e in episodes}
    if len(expected) != len(episodes):
        raise ValueError("duplicate episode metadata")
    train = info["splits"]["train_episode_ids"]
    validation = info["splits"]["validation_episode_ids"]
    if (len(set(train)) != len(train) or len(set(validation)) != len(validation)
            or set(train) & set(validation) or set(train) | set(validation) != set(expected)):
        raise ValueError("invalid original train/validation partition")
    files = sorted((dataset / "data").glob("*/*.parquet"))
    physical = [int(p.stem.rsplit("_", 1)[1]) for p in files]
    if sorted(physical) != sorted(expected):
        raise ValueError("physical parquet set differs from episode metadata")
    if info["total_episodes"] != len(episodes):
        raise ValueError("episode count mismatch")
    sample_count = min(video_sample_episodes, len(episodes))
    sampled = {sorted(expected)[int(i)] for i in np.linspace(0, len(episodes) - 1, sample_count)}
    checked_frames = 0
    frame_total = 0
    video_count = 0
    decoded = []
    expected_videos = set()
    for path, eid in zip(files, physical):
        table = pq.read_table(path)
        n = len(table)
        if n != expected[eid]["length"] or n < 1:
            raise ValueError(f"episode length mismatch: {eid}")
        values = {k: table[k].to_numpy(zero_copy_only=False) for k in
                  ("frame_index", "episode_index", "task_index", "timestamp", "annotation.human.coarse_action")}
        if not np.array_equal(values["frame_index"], np.arange(n)):
            raise ValueError(f"noncontiguous frame_index: {eid}")
        if not np.all(values["episode_index"] == eid):
            raise ValueError(f"mixed episode_index: {eid}")
        np.testing.assert_allclose(values["timestamp"], np.arange(n) / 30, atol=2e-5, rtol=0)
        if not np.all(values["task_index"] == 40) or not np.all(values["annotation.human.coarse_action"] == 40):
            raise ValueError(f"unexpected Task40 language/task label: {eid}")
        if tasks[40]["task_name"] != "make_microwave_popcorn" or not tasks[40]["task"].strip():
            raise ValueError("Task40 language lookup mismatch")
        state = np.stack(table["observation.state"].to_numpy(zero_copy_only=False))
        action = np.stack(table["action"].to_numpy(zero_copy_only=False))
        if state.shape != (n, 82) or action.shape != (n, 23):
            raise ValueError(f"incorrect physical tensor dimensions: {eid}")
        if not np.isfinite(state).all() or not np.isfinite(action).all():
            raise ValueError(f"nonfinite state/action: {eid}")
        # Redundant coordinates validate the named packed layout, including
        # the arm positions used as relative-action reference states.
        for start, sin_start, cos_start in ((18, 25, 32), (48, 55, 62)):
            np.testing.assert_allclose(np.sin(state[:, start:start + 7]), state[:, sin_start:sin_start + 7], atol=2e-6)
            np.testing.assert_allclose(np.cos(state[:, start:start + 7]), state[:, cos_start:cos_start + 7], atol=2e-6)
        for start in (42, 72):
            np.testing.assert_allclose(np.linalg.norm(state[:, start:start + 4], axis=1), 1, atol=2e-5)
        for camera in CAMERAS:
            video = dataset / info["video_path"].format(
                episode_chunk=eid // info["chunks_size"], episode_index=eid, video_key=camera)
            expected_videos.add(video)
            with av.open(str(video)) as container:
                stream = container.streams.video[0]
                if stream.frames != n or float(stream.average_rate) != 30 or (stream.width, stream.height) != (256, 256):
                    raise ValueError(f"video header/row alignment mismatch: {video}")
                if eid in sampled:
                    for index in sorted({0, n // 2, n - 1}):
                        target = index / 30
                        container.seek(int(target / stream.time_base), stream=stream, backward=True)
                        frame = next((f for f in container.decode(video=0)
                                      if float(f.pts * stream.time_base) >= target - 1e-6), None)
                        if frame is None or abs(float(frame.pts * stream.time_base) - target) > 1e-5:
                            raise ValueError(f"video timestamp decode mismatch: {video}, frame {index}")
                        if frame.to_ndarray(format="rgb24").shape != (256, 256, 3):
                            raise ValueError(f"decoded image shape mismatch: {video}")
                        checked_frames += 1
                    decoded.append({"episode_index": eid, "camera": camera, "indices": sorted({0, n // 2, n - 1})})
            video_count += 1
        frame_total += n
    if set((dataset / "videos").glob("*/*/*.mp4")) != expected_videos:
        raise ValueError("physical video set differs from episode metadata")
    if frame_total != info["total_frames"] or video_count != info["total_videos"]:
        raise ValueError("frame/video total mismatch")
    metadata_hashes = {name: hashlib.sha256((dataset / "meta" / name).read_bytes()).hexdigest()
                       for name in ("info.json", "modality.json", "episodes.jsonl", "tasks.jsonl")}
    return dict(status="PASS", scope="existing files only; native N1.7 runtime not exercised",
                dataset=str(dataset), contract=contract, episodes=len(episodes), rows=frame_total,
                original_split=dict(train=len(train), validation=len(validation), overlap=0),
                parquet_files=len(files), video_headers=video_count, decoded_frames=checked_frames,
                decoded_samples=decoded, metadata_sha256=metadata_hashes,
                native_train_ready=False,
                remaining=["separate physical train-only view and newly computed statistics",
                           "N1.7 loader/processor/model and transport runtime checks",
                           "approved experiment identity and scientific training configuration"])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("dataset", type=Path)
    parser.add_argument("--video-sample-episodes", type=int, default=3)
    args = parser.parse_args()
    if args.video_sample_episodes < 1:
        parser.error("video-sample-episodes must be positive")
    try:
        result = audit(args.dataset, video_sample_episodes=args.video_sample_episodes)
    except Exception as error:
        print(json.dumps(dict(status="FAIL", error_type=type(error).__name__, error=str(error)), ensure_ascii=False, indent=2))
        return 1
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
