"""Compare three same-memory Task40 restores at matching decision and control indices."""
import argparse
import itertools
import json
from pathlib import Path

import numpy as np

from task40_replay_audit import load


def phase(row):
    microwave = "microwave_hjjxmi_0"
    bag = "popcorn_bag_73"
    return {"open": row["microwaves"][microwave]["Open"],
            "grasp": {arm: row["grasp_states"][f"robot_r1/{arm}/{bag}"] for arm in ("left", "right")},
            "inside": row["popcorn"][bag]["inside"][microwave],
            "toggled": row["microwaves"][microwave]["ToggledOn"],
            "goal": row["predicate_options"], "progress": row["task_progress"]}


def compare_arrays(left, right):
    diff = np.abs(left.astype(np.float64) - right.astype(np.float64))
    return {"mae": float(diff.mean()), "max": float(diff.max()), "changed_fraction": float((diff != 0).mean())}


def compare_decision(left, right):
    obs_file, action_file = "observation.npz", "executed.npz"
    with np.load(left / obs_file) as first, np.load(right / obs_file) as second:
        assert first.files == second.files
        obs = {key: compare_arrays(first[key], second[key]) for key in first.files}
    with np.load(left / action_file) as first, np.load(right / action_file) as second:
        assert first.files == second.files
        actions = {key: compare_arrays(first[key], second[key]) for key in first.files}
    return {"observation": obs, "executed_actions": actions}


def audit(run):
    summary = load(run / "repeat_summary.json")
    anchor, step = summary["anchor_decision"], summary["anchor_step"]
    assert len(summary["trials"]) == 3
    logs, trial_results = {}, {}
    for number, trial in enumerate(summary["trials"], 1):
        branch = trial["branch"]
        assert trial["decisions"] == 10 and trial["steps"] == 80
        assert load(run / f"memory_compare_{number}.json")["equal"]
        rows = [json.loads(line) for line in (run / branch / "truth.jsonl").read_text().splitlines()]
        assert [row["step"] for row in rows] == list(range(step + 1, step + 81))
        logs[branch] = rows
        edges = [{"step": row["step"], "phase": phase(row)}
                 for previous, row in zip(rows, rows[1:]) if phase(previous) != phase(row)]
        right = [row["distance"]["eef_m"]["right"] for row in rows]
        door = [row["microwaves"]["microwave_hjjxmi_0"]["joint_positions"][0] for row in rows]
        trial_results[branch] = {"initial_phase": phase(rows[0]), "phase_edges": edges,
                                 "min_right_eef_m": min(right), "final_right_eef_m": right[-1],
                                 "door_joint_range": [min(door), max(door)]}
    pairs = {}
    for left, right in itertools.combinations(logs, 2):
        pair_rows = list(zip(logs[left], logs[right], strict=True))
        decisions = {}
        for index in range(anchor, anchor + 10):
            directories = [run / branch / f"{index:04d}" for branch in (left, right)]
            first, second = (load(directory / "decision.json") for directory in directories)
            for key in ("decision", "episode", "steps", "session", "language", "memory_after"):
                assert first[key] == second[key]
            decisions[index] = compare_decision(*directories)
        pairs[f"{left}_vs_{right}"] = {
            "phase_disagreement_steps": [first["step"] for first, second in pair_rows if phase(first) != phase(second)],
            "max_right_eef_position_difference_m": max(float(np.linalg.norm(
                np.asarray(first["robots"]["robot_r1"]["end_effectors"]["right"][0])
                - second["robots"]["robot_r1"]["end_effectors"]["right"][0])) for first, second in pair_rows),
            "max_base_xy_position_difference_m": max(float(np.linalg.norm(
                np.asarray(first["robots"]["robot_r1"]["position"][:2])
                - second["robots"]["robot_r1"]["position"][:2])) for first, second in pair_rows),
            "decisions": decisions}
    return {"trials": trial_results, "pairs": pairs}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run", type=Path)
    print(json.dumps(audit(parser.parse_args().run), indent=2, allow_nan=False))
