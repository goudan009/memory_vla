"""Check decision continuity and summarize physical events in a Task40 replay run."""
import argparse
import json
from pathlib import Path

import numpy as np


def load(path):
    return json.loads(path.read_text())


def audit(run):
    summary = load(run / "summary.json")
    anchor = summary["anchor_decision"]
    session = load(run / "start.json")["clock"]
    result = {}
    for branch, first, count in (("prefix", 0, anchor), ("original", anchor, summary["original_cycles"]),
                                  ("restored", anchor, summary["restored_cycles"])):
        folder = run / branch
        rows = [json.loads(line) for line in (folder / "truth.jsonl").read_text().splitlines()]
        assert [row["step"] for row in rows] == list(range(8 * first + 1, 8 * (first + count) + 1))
        assert all(left["sim_time_s"] < right["sim_time_s"] for left, right in zip(rows, rows[1:]))
        for index in range(first, first + count):
            entry = folder / f"{index:04d}"
            metadata = load(entry / "decision.json")
            assert metadata["decision"] == index and metadata["steps"] == [8 * index, 8 * (index + 1)]
            assert metadata["session"] == session and metadata["episode"] == session["episode_ids"][0]
            assert metadata["sim_time_s"][0] < metadata["sim_time_s"][1] == rows[8 * (index - first) + 7]["sim_time_s"]
            assert metadata["memory_after"]["next_decision"] == index + 1
            assert all(metadata["memory_after"][name]["last_written"] == index for name in ("perceptual", "cognitive"))
            with np.load(entry / "observation.npz", allow_pickle=False) as obs:
                views = [obs[key] for key in obs.files if key.startswith("video.")]
                assert len(views) == 3 and all(view.shape == (1, 1, 256, 256, 3) and view.dtype == np.uint8 for view in views)
                assert any(key.startswith("state.") for key in obs.files) and metadata["language"]
            with np.load(entry / "executed.npz", allow_pickle=False) as actions:
                assert actions["robot_r1"].shape == (8, 23) and np.isfinite(actions["robot_r1"]).all()
        micro = [row["microwaves"]["microwave_hjjxmi_0"] for row in rows]
        opened = [row["step"] for row, state in zip(rows, micro) if state["Open"]]
        toggled = [row["step"] for row, state in zip(rows, micro) if state["ToggledOn"]]
        inside = [row["step"] for row in rows if row["popcorn"]["popcorn_bag_73"]["inside"]["microwave_hjjxmi_0"]]
        grasped = [row["step"] for row in rows if "TRUE" in row["grasp_states"].values()]
        result[branch] = {"decisions": count, "steps": [rows[0]["step"], rows[-1]["step"]],
                          "opened": opened, "grasped": grasped, "inside": inside, "toggled": toggled,
                          "closed_after_opening": [row["step"] for row, previous, current in zip(rows[1:], micro, micro[1:])
                                                    if previous["Open"] and not current["Open"]],
                          "later_stage_before_opening": sorted({step for step in grasped + inside + toggled
                                                               if not opened or step < opened[0]}),
                          "goal_satisfied_steps": [row["step"] for row in rows if any(map(all, row["predicate_options"]))],
                          "grasp_values": sorted({value for row in rows for value in row["grasp_states"].values()}),
                          "door_joint_min_max": [min(state["joint_positions"][0] for state in micro),
                                                 max(state["joint_positions"][0] for state in micro)],
                          "min_right_eef_m": min(row["distance"]["eef_m"]["right"] for row in rows)}
    original, restored = (run / branch / f"{anchor:04d}" for branch in ("original", "restored"))
    with np.load(original / "observation.npz") as lhs, np.load(restored / "observation.npz") as rhs:
        result["first_observation_rgb_changed_components"] = {key: int(np.count_nonzero(lhs[key] != rhs[key]))
                                                              for key in lhs.files if key.startswith("video.")}
        result["first_observation_proprio_max_abs_difference"] = {key: float(np.abs(lhs[key] - rhs[key]).max())
                                                                 for key in lhs.files if key.startswith("state.")
                                                                 and not np.array_equal(lhs[key], rhs[key])}
    with np.load(original / "executed.npz") as lhs, np.load(restored / "executed.npz") as rhs:
        result["first_action_chunk_max_abs_difference"] = float(np.abs(lhs["robot_r1"] - rhs["robot_r1"]).max())
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run", type=Path)
    print(json.dumps(audit(parser.parse_args().run), indent=2))
