"""Pre-register an Episode-isolated, bounded Task 21 diagnostic panel.

Privileged object/grasp annotations select candidate evaluation slices only;
they are never converted into model inputs or additional supervision.
"""

import argparse
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

from .convert_b1k import CAMERAS, convert, video_info, write_json
from v3_bodymemory.experiment.output import create_run_directory


def inspect_state(path):
    state = np.stack(pd.read_parquet(path, columns=["observation.state"])["observation.state"])
    assert state.ndim == 2 and state.shape[1] == 256 and np.isfinite(state).all(), path
    return {"rows": len(state), "dimension": 256,
            "state_sha256": hashlib.sha256(state.tobytes()).hexdigest()}


def inspect_episode(source, path):
    eid = int(path.stem.split("_")[-1])
    table = pq.read_table(path, columns=["episode_index", "task_index", "timestamp", "action"])
    n = len(table)
    assert set(table["episode_index"].to_pylist()) == {eid}
    assert set(table["task_index"].to_pylist()) == {21}
    np.testing.assert_allclose(table["timestamp"].to_numpy(), np.arange(n) / 30, atol=2e-5)
    actions = np.stack(table["action"].to_pylist()).astype(np.float32)
    assert actions.shape == (n, 23) and np.isfinite(actions).all()
    state = inspect_state(path)
    assert state["rows"] == n
    videos = {}
    for camera in CAMERAS:
        p = source / f"videos/task-0021/observation.images.rgb.{camera}/episode_{eid:08d}.mp4"
        v = video_info(p)
        assert v["frames"] == n and v["fps"] == 30 and v["first_timestamp"] == 0, (p, v, n)
        videos[camera] = v
    meta = json.loads((source / f"meta/episodes/task-0021/episode_{eid:08d}.json").read_text())
    config = json.loads(meta["config"])
    return {"episode": eid, "frames": n, "state": state,
            "action_sha256": hashlib.sha256(actions.tobytes()).hexdigest(),
            "scene_instance": config["scene"].get("scene_instance"), "videos": videos}


def movement_events(position_at_decisions, object_name):
    """Motion bouts separated by >=8 stationary decisions (not grasp labels)."""
    p = np.asarray(position_at_decisions)
    moving = np.linalg.norm(np.diff(p, axis=0), axis=1) > .005
    active = np.flatnonzero(moving)
    if not len(active):
        return []
    chunks = np.split(active, np.flatnonzero(np.diff(active) > 8) + 1)
    events = []
    for chunk in chunks:
        first, last = int(chunk[0]), int(chunk[-1]) + 1
        if last - first < 15 or np.linalg.norm(p[last] - p[first]) < .10:
            continue
        for index, label in ((first, "start"), (last, "stop")):
            events.append({"frame": index * 8, "object": object_name,
                           "event": "object_transport_" + label})
    return events


def candidate_events(source, eid):
    meta = json.loads((source / f"meta/episodes/task-0021/episode_{eid:08d}.json").read_text())
    table = pq.read_table(source / f"data/task-0021/episode_{eid:08d}.parquet",
                          columns=["observation.task_info"])
    info = np.stack(table.column(0).to_pylist())
    offset, events = 0, []
    for key in meta["task_obs_keys"]:
        dim = 3 if key.endswith(("_pos", "_ori_cos", "_ori_sin")) else 1
        if key.endswith("_pos") and key.startswith(("die", "teddy", "train_set", "board_game")):
            events.extend(movement_events(info[::8, offset:offset+3], key[:-4]))
        offset += dim
    assert offset == info.shape[1]
    return sorted(events, key=lambda e: e["frame"])


def panel(eid, length, events, split):
    # Four uniform windows estimate broad sampled performance. Event-enriched
    # windows are reported separately, not passed off as a whole-Episode score.
    limit = ((length - 280) // 8) * 8
    starts = [int(x // 8 * 8) for x in np.linspace(0, limit, 4)]
    out = [{"episode": eid, "start": s, "split": split, "stratum": "uniform",
            "candidate_mask": [False] * 32} for s in starts]
    valid = [e for e in events if 192 <= e["frame"] <= limit + 192]
    count = 4 if split == "train" else 2
    # Spread candidates through the trajectory, prefer release and grasp alike.
    for j in np.linspace(0, max(0, len(valid)-1), min(count, len(valid)), dtype=int):
        event = valid[int(j)]
        start = max(0, min(limit, ((event["frame"] - 192) // 8) * 8))
        if any(abs(start - p["start"]) < 280 for p in out):
            continue
        mask = [i >= 17 and event["frame"] <= start + 8*i <= event["frame"] + 56 for i in range(32)]
        assert any(mask)
        out.append({"episode": eid, "start": start, "split": split, "stratum": "event_candidate",
                    "candidate_mask": mask, "event": event,
                    "interpretation": "object transport transition candidate; necessity of history is not established"})
    return out


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--source", type=Path, default=Path("assets/datasets/b1k2025"))
    p.add_argument("--dataset", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--panel-only", action="store_true", help="Rebuild annotations from the existing inventory; no conversion")
    a = p.parse_args()
    if a.panel_only:
        records = json.loads((a.output / "config/inventory.json").read_text())["episodes"]
    else:
        create_run_directory(a.output)
        (a.output / "checkpoints").rmdir()  # paired arm checkpoints live below each arm
        paths = sorted((a.source / "data/task-0021").glob("*.parquet"))
        with ThreadPoolExecutor(max_workers=6) as pool:
            records = list(pool.map(lambda path: inspect_episode(a.source, path), paths))
    assert len({r["action_sha256"] for r in records}) == len(records), "Duplicate action trajectories"
    ids = np.array([r["episode"] for r in records])
    np.random.default_rng(a.seed).shuffle(ids)
    train, val = list(map(int, ids[:8])), list(map(int, ids[8:12]))
    assert not set(train) & set(val)
    write_json(a.output / "config/inventory.json", {"usable_episodes": len(records),
        "unique_action_trajectories": len(records), "scene_instances": sorted({r["scene_instance"] for r in records}),
        "video_checks": "600 headers + first decoded frame, 30Hz aligned to parquet; not full-video corruption scan",
        "episodes": records})
    groups, all_events = [], {}
    by_id = {r["episode"]: r for r in records}
    for split, selected in (("train", train), ("validation", val)):
        for eid in selected:
            events = candidate_events(a.source, eid)
            all_events[str(eid)] = events
            groups.extend(panel(eid, by_id[eid]["frames"], events, split))
            print("PANEL", eid, split, "events", len(events), flush=True)
    for i, g in enumerate(groups):
        g["group_id"] = i
    write_json(a.output / "config/data_split.json", {"seed": a.seed, "train_episodes": train,
        "validation_episodes": val, "unused_episodes": list(map(int, ids[12:])), "groups": groups,
        "observation_stride": 8, "sequence_length": 32, "action_label_horizon": 32,
        "normalization": "unchanged official checkpoint statistics; converted all-split statistics never applied",
        "candidate_definition": "object motion >5mm/decision, bouts >=15 decisions and >10cm net displacement, gap <=8 decisions; first8 decisions after start/stop with >=17 earlier decisions",
        "unusable_annotation": "all selected in_gripper flags are -1; never used as actual grasp/release labels",
        "candidate_limit": "offline heuristic candidates, not certified history-essential observations",
        "scope": "bounded diagnostic panel, not exhaustive whole-Episode or unseen-scene evaluation"})
    write_json(a.output / "diagnostics/task_events.json", all_events)
    print("INVENTORY", len(records), "TRAIN", train, "VALIDATION", val, "GROUPS", len(groups), flush=True)
    if not a.panel_only:
        convert(a.source, a.dataset, train + val)


if __name__ == "__main__":
    main()
