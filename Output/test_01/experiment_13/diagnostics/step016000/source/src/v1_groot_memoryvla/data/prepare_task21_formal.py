"""Convert 200 trajectories of one B1K task; retain official normalization."""

import argparse
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
from pathlib import Path
import shutil

import numpy as np
import pandas as pd

from .convert_b1k import (STATE_SLICES, ACTION_SLICES, CAMERAS, validate_raw_state,
                         convert_video, write_json)


def prepare(source, destination, checkpoint, output, workers=8, seed=42, task=21):
    source, destination, checkpoint, output = map(Path, (source, destination, checkpoint, output))
    output.mkdir(parents=True, exist_ok=True)
    task_directory = f"task-{task:04d}"
    paths = sorted((source / "data" / task_directory).glob("*.parquet"))
    assert len(paths) == 200, f"Expected 200 demonstrations, found {len(paths)}"
    episodes = {e["episode_index"]: e for e in map(json.loads,
                (source / "meta/episodes.jsonl").read_text().splitlines())}
    ids = np.array([int(p.stem.split("_")[-1]) for p in paths])
    np.random.default_rng(seed).shuffle(ids)
    split = {"seed": seed, "task": task, "train": list(map(int, ids[:180])),
             "validation": list(map(int, ids[180:])), "sampling_stride": 8,
             "gradient_window": 32, "normalization": "unchanged official checkpoint"}
    assert not set(split["train"]) & set(split["validation"])
    write_json(output / "data_split.json", split)

    def one(path):
        eid = int(path.stem.split("_")[-1])
        df = pd.read_parquet(path)
        raw = np.stack(df["observation.state"])
        validate_raw_state(raw)
        actions = np.stack(df["action"]).astype(np.float32)
        assert actions.shape == (len(df), 23) and np.isfinite(actions).all()
        assert set(df.episode_index) == {eid} and set(df.task_index) == {task}
        np.testing.assert_allclose(df.timestamp, np.arange(len(df)) / 30, atol=2e-5)
        packed = np.concatenate([raw[:, a:b] for a, b in STATE_SLICES.values()], 1).astype(np.float32)
        target = destination / "data" / task_directory / path.name
        target.parent.mkdir(parents=True, exist_ok=True)
        pd.DataFrame({"index": df["index"], "frame_index": np.arange(len(df)),
            "episode_index": eid, "task_index": df.task_index, "timestamp": df.timestamp,
            "observation.state": list(packed), "action": list(actions),
            "annotation.human.coarse_action": df.task_index}).to_parquet(target, index=False)
        videos = []
        for cam in CAMERAS:
            key = f"observation.images.rgb.{cam}"
            videos.append(convert_video(source / f"videos/{task_directory}/{key}/episode_{eid:08d}.mp4",
                destination / f"videos/{task_directory}/{key}_256_256/episode_{eid:08d}.mp4", len(df)))
        print(json.dumps({"converted_episode": eid, "frames": len(df)}), flush=True)
        return {"episode_index": eid, "length": len(df), "videos": videos,
                "action_sha256": hashlib.sha256(actions.tobytes()).hexdigest()}

    with ThreadPoolExecutor(max_workers=workers) as pool:
        records = list(pool.map(one, paths))
    assert len({r["action_sha256"] for r in records}) == 200
    modality = {"state": {}, "action": {}, "video": {}, "annotation": {
        "human.coarse_action": {"original_key": "annotation.human.coarse_action"}}}
    offset = 0
    for name, (a, b) in STATE_SLICES.items():
        modality["state"][name] = {"start": offset, "end": offset + b-a}
        offset += b-a
    modality["action"] = {k: {"start": a, "end": b} for k, (a, b) in ACTION_SLICES.items()}
    features = {"observation.state": {"dtype": "float32", "shape": [82], "names": None},
                "action": {"dtype": "float32", "shape": [23], "names": None}}
    for cam in CAMERAS:
        key = f"observation.images.rgb.{cam}_256_256"
        modality["video"][key] = {"original_key": key}
        features[key] = {"dtype": "video", "shape": [256, 256, 3], "info": {"video.fps": 30}}
    official = json.loads((checkpoint / "statistics.json").read_text())["behavior_r1_pro"]
    stats = {}
    for mode, column in (("state", "observation.state"), ("action", "action")):
        stats[column] = {k: sum((official[mode][name][k] for name in modality[mode]), [])
                         for k in ("min", "max", "mean", "std", "q01", "q99")}
    info = json.loads((source / "meta/info.json").read_text())
    info.update(total_episodes=200, total_frames=sum(r["length"] for r in records),
                total_videos=600, features=features,
                splits={"train_episode_ids": split["train"], "validation_episode_ids": split["validation"]})
    meta = destination / "meta"
    write_json(meta / "info.json", info)
    write_json(meta / "modality.json", modality)
    write_json(meta / "stats.json", stats)
    write_json(meta / "relative_stats.json", official["relative_action"])
    (meta / "episodes.jsonl").write_text("".join(json.dumps(dict(episodes[r["episode_index"]],
         length=r["length"])) + "\n" for r in records))
    shutil.copyfile(source / "meta/tasks.jsonl", meta / "tasks.jsonl")
    write_json(output / "conversion.json", {"source": str(source), "dataset": str(destination),
        "normalization": "official statistics; no validation fitting", "episodes": records})


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    for name in ("source", "destination", "checkpoint", "output"):
        p.add_argument("--" + name, type=Path, required=True)
    p.add_argument("--workers", type=int, default=8)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--task", type=int, default=21)
    a = p.parse_args()
    prepare(**vars(a))
