"""Convert raw 2025 B1K demonstrations to the N1.6 LeRobot input contract.

R1Pro's 256-D recording layout follows BEHAVIOR-1K
omnigibson/learning/utils/eval_utils.py (no grasp scalar fields). Validate its
redundant sin/cos and quaternion values before slicing. The older GR00T example
uses a different 258-D layout and must not be used on these recordings.
"""

import argparse
from concurrent.futures import ThreadPoolExecutor
import json
from pathlib import Path
import subprocess

import av
import numpy as np
import pandas as pd


# Ordered exactly as behavior_r1_pro's checkpoint state modality.
STATE_SLICES = {
    "robot_pos": (140, 143), "robot_ori_cos": (143, 146),
    "robot_ori_sin": (146, 149), "robot_2d_ori": (149, 150),
    "robot_2d_ori_cos": (150, 151), "robot_2d_ori_sin": (151, 152),
    "robot_lin_vel": (152, 155), "robot_ang_vel": (155, 158),
    "arm_left_qpos": (158, 165), "arm_left_qpos_sin": (165, 172),
    "arm_left_qpos_cos": (172, 179), "eef_left_pos": (186, 189),
    "eef_left_quat": (189, 193), "gripper_left_qpos": (193, 195),
    "arm_right_qpos": (197, 204), "arm_right_qpos_sin": (204, 211),
    "arm_right_qpos_cos": (211, 218), "eef_right_pos": (225, 228),
    "eef_right_quat": (228, 232), "gripper_right_qpos": (232, 234),
    "trunk_qpos": (236, 240),
}
ACTION_SLICES = {"base": (0, 3), "torso": (3, 7), "left_arm": (7, 14),
                 "left_gripper": (14, 15), "right_arm": (15, 22), "right_gripper": (22, 23)}
RELATIVE_STATE = {"torso": "trunk_qpos", "left_arm": "arm_left_qpos", "right_arm": "arm_right_qpos"}
CAMERAS = ("head", "left_wrist", "right_wrist")


def write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2) + "\n")


def validate_raw_state(state):
    if state.ndim != 2 or state.shape[1] != 256 or not np.isfinite(state).all():
        raise ValueError(f"Expected finite raw R1Pro state [T,256], got {state.shape}")
    for start, sin_start, cos_start, size in ((0, 28, 56, 28), (158, 165, 172, 7),
                                             (197, 204, 211, 7), (244, 247, 250, 3)):
        q = state[:, start:start + size]
        np.testing.assert_allclose(np.sin(q), state[:, sin_start:sin_start + size], atol=2e-6)
        np.testing.assert_allclose(np.cos(q), state[:, cos_start:cos_start + size], atol=2e-6)
    for start in (189, 228):
        np.testing.assert_allclose(np.linalg.norm(state[:, start:start + 4], axis=1), 1, atol=2e-5)


def statistics(array):
    array = np.asarray(array, dtype=np.float64)
    return {name: value.tolist() for name, value in {
        "mean": array.mean(0), "std": array.std(0), "min": array.min(0),
        "max": array.max(0), "q01": np.quantile(array, .01, axis=0),
        "q99": np.quantile(array, .99, axis=0),
    }.items()}


def video_info(path):
    with av.open(str(path)) as container:
        stream = container.streams.video[0]
        first = next(container.decode(video=0))
        return {"frames": stream.frames, "fps": float(stream.average_rate),
                "width": stream.width, "height": stream.height,
                "first_timestamp": float(first.pts * stream.time_base)}


def convert_video(source, target, length):
    before = video_info(source)
    if before["frames"] != length or before["fps"] != 30 or before["first_timestamp"] != 0:
        raise ValueError(f"Video/parquet alignment mismatch: {source}: {before}, rows={length}")
    target.parent.mkdir(parents=True, exist_ok=True)
    if not target.exists():
        partial = target.with_suffix(".partial.mp4")
        subprocess.run(["ffmpeg", "-nostdin", "-v", "error", "-n", "-i", str(source),
                        "-vf", "scale=256:256", "-vsync", "0", "-c:v", "libx264",
                        "-preset", "veryfast", "-crf", "18", "-threads", "2", "-an", str(partial)], check=True)
        partial.rename(target)
    after = video_info(target)
    if after != dict(before, width=256, height=256):
        raise ValueError(f"Converted video contract failed: {target}: {after}")
    return {"source": str(source.resolve()), "target": str(target), "input": before, "output": after}


def convert(source, destination, episode_ids):
    source, destination = Path(source), Path(destination)
    if source.resolve() == destination.resolve():
        raise ValueError("Conversion output must be separate from the raw dataset")
    original_info = json.loads((source / "meta/info.json").read_text())
    if original_info["fps"] != 30:
        raise ValueError("This conversion requires 30 Hz recordings")
    metadata = {e["episode_index"]: e for e in map(json.loads, (source / "meta/episodes.jsonl").read_text().splitlines())}
    tasks = [json.loads(line) for line in (source / "meta/tasks.jsonl").read_text().splitlines()]
    modality = {"state": {}, "action": {}, "video": {}, "annotation": {
        "human.coarse_action": {"original_key": "annotation.human.coarse_action"}}}
    offset = 0
    for key, (start, stop) in STATE_SLICES.items():
        modality["state"][key] = {"start": offset, "end": offset + stop - start}
        offset += stop - start
    assert offset == 82
    modality["action"] = {key: {"start": start, "end": stop} for key, (start, stop) in ACTION_SLICES.items()}
    features = {"observation.state": {"dtype": "float32", "shape": [82], "names": None},
                "action": {"dtype": "float32", "shape": [23], "names": None}}
    for camera in CAMERAS:
        key = f"observation.images.rgb.{camera}_256_256"
        modality["video"][key] = {"original_key": key}
        features[key] = {"dtype": "video", "shape": [256, 256, 3], "info": {"video.fps": 30}}
    state_arrays, action_arrays, relative_arrays = [], [], {key: [] for key in RELATIVE_STATE}
    episodes, manifest = [], []
    for eid in episode_ids:
        chunk = eid // original_info["chunks_size"]
        raw_path = source / original_info["data_path"].format(episode_chunk=chunk, episode_index=eid)
        df = pd.read_parquet(raw_path)
        raw_state = np.stack(df["observation.state"])
        validate_raw_state(raw_state)
        if set(df["episode_index"]) != {eid}:
            raise ValueError("Parquet contains mixed episodes")
        np.testing.assert_allclose(df.timestamp, np.arange(len(df)) / 30, atol=2e-5)
        packed = np.concatenate([raw_state[:, start:stop] for start, stop in STATE_SLICES.values()], axis=1).astype(np.float32)
        action = np.stack(df["action"]).astype(np.float32)
        if action.shape != (len(df), 23) or not np.isfinite(action).all():
            raise ValueError("Invalid action shape/values")
        out = pd.DataFrame({"index": df["index"], "frame_index": np.arange(len(df)),
                            "episode_index": eid, "task_index": df["task_index"], "timestamp": df.timestamp,
                            "observation.state": list(packed), "action": list(action),
                            "annotation.human.coarse_action": df["task_index"]})
        output_parquet = destination / f"data/task-{chunk:04d}/episode_{eid:08d}.parquet"
        output_parquet.parent.mkdir(parents=True, exist_ok=True)
        out.to_parquet(output_parquet, index=False)
        state_arrays.append(packed)
        action_arrays.append(action)
        usable = len(df) - 31
        for key, state_key in RELATIVE_STATE.items():
            a, b = ACTION_SLICES[key]
            s, e = STATE_SLICES[state_key]
            # Same 32-frame action chunks and current-state subtraction as
            # StateActionProcessor/JointActionChunk; no future state is used.
            relative_arrays[key].append(np.concatenate([
                action[d:d + usable, a:b] - raw_state[:usable, s:e] for d in range(32)], axis=0))
        with ThreadPoolExecutor(max_workers=3) as pool:
            jobs = []
            for camera in CAMERAS:
                raw_key = f"observation.images.rgb.{camera}"
                raw_video = source / original_info["video_path"].format(episode_chunk=chunk, episode_index=eid, video_key=raw_key)
                target_video = destination / f"videos/task-{chunk:04d}/{raw_key}_256_256/episode_{eid:08d}.mp4"
                jobs.append(pool.submit(convert_video, raw_video, target_video, len(df)))
            video_records = [job.result() for job in jobs]
        episodes.append(dict(metadata[eid], length=len(df)))
        manifest.append({"episode_index": eid, "source_parquet": str(raw_path.resolve()),
                         "rows": len(df), "raw_state_dim": 256, "packed_state_dim": 82,
                         "state_layout_checks": "joint sin/cos + unit eef quaternions",
                         "videos": video_records})
        print(f"Converted episode {eid}: {len(df)} synchronized rows, 3 videos", flush=True)
    stats = {"observation.state": statistics(np.concatenate(state_arrays)), "action": statistics(np.concatenate(action_arrays))}
    relative_stats = {key: statistics(np.concatenate(arrays)) for key, arrays in relative_arrays.items()}
    info = dict(original_info, total_episodes=len(episodes), total_frames=sum(e["length"] for e in episodes),
                total_videos=3 * len(episodes), features=features, splits={"train": f"0:{len(episodes)}"})
    write_json(destination / "meta/info.json", info)
    write_json(destination / "meta/modality.json", modality)
    write_json(destination / "meta/stats.json", stats)
    write_json(destination / "meta/relative_stats.json", relative_stats)
    (destination / "meta/episodes.jsonl").write_text("".join(json.dumps(e) + "\n" for e in episodes))
    (destination / "meta/tasks.jsonl").write_text("".join(json.dumps(t) + "\n" for t in tasks))
    write_json(destination / "meta/conversion.json", {"format": "b1k256_to_gr00t82_v1", "fps": 30,
               "state_slices_raw": STATE_SLICES, "action_slices": ACTION_SLICES,
               "label_horizon": 32, "source_dataset": str(source.resolve()), "episodes": manifest})
    return destination


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--destination", type=Path, required=True)
    parser.add_argument("--episodes", type=int, nargs="+", required=True)
    args = parser.parse_args()
    print(convert(args.source, args.destination, args.episodes))


if __name__ == "__main__":
    main()
