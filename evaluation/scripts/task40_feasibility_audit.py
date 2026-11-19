"""Audit saved Task40 rollout traces without starting the simulator."""
import argparse
import json
import subprocess
from collections import Counter, defaultdict
from pathlib import Path


def read_jsonl(path):
    with path.open() as stream:
        return [json.loads(line) for line in stream if line.strip()]


def identity(row):
    return tuple(row[key] for key in ("env_idx", "instance_id", "seed"))


def event_values(row):
    microwave, = row["microwaves"]
    bag, = row["popcorn"]
    states = row["microwaves"][microwave]
    return {"Open": states["Open"], "ToggledOn": states["ToggledOn"],
            "Inside": row["popcorn"][bag]["inside"][microwave],
            "Grasped": any(value == "TRUE" for key, value in row["grasp_states"].items()
                           if key.rsplit("/", 1)[-1] == bag)}


def audit_run(run):
    stages = run / "stages"
    episodes = read_jsonl(run / "episodes.jsonl")
    summary = {"run": str(run), "episodes": len(episodes), "envs": {}}
    for env_idx in (0, 1):
        state_rows = read_jsonl(stages / f"task40_gpu0_env{env_idx}.jsonl")
        action_rows = read_jsonl(stages / f"actions_task40_gpu0_env{env_idx}.jsonl")
        video_rows = read_jsonl(stages / f"video_gpu0_env{env_idx}.jsonl")
        previous = {}
        counts = Counter()
        transitions = defaultdict(list)
        for row in state_rows:
            key = identity(row)
            values = event_values(row)
            counts.update(name for name, value in values.items() if value)
            if key in previous:
                old = previous[key]
                for name, value in values.items():
                    if value != event_values(old)[name]:
                        transitions[name].append({"instance": key[1], "to": value,
                                                  "steps": [old["action_steps"], row["action_steps"]],
                                                  "seconds": [old["sim_time_s"], row["sim_time_s"]]})
            previous[key] = row
        action_steps = {(identity(row), row["action_start"]): row for row in action_rows}
        exact_video_matches = sum((identity(row), row["action_steps"]) in action_steps for row in video_rows)
        action_fields = {"action_start", "action_end", "action", "sim_time_s", "seed"}
        state_fields = {"microwaves", "popcorn", "grasped", "robots", "sim_time_s"}
        missing_actions = sorted(action_fields - action_rows[0].keys()) if action_rows else sorted(action_fields)
        missing_state = sorted(state_fields - state_rows[0].keys()) if state_rows else sorted(state_fields)
        summary["envs"][str(env_idx)] = {
            "state_rows": len(state_rows),
            "action_rows": len(action_rows),
            "video_frame_rows": len(video_rows),
            "instances": sorted({row["instance_id"] for row in state_rows}),
            "state_sample_steps": [state_rows[0]["action_steps"], state_rows[-1]["action_steps"]]
            if state_rows else [],
            "video_sample_steps": [video_rows[0]["action_steps"], video_rows[-1]["action_steps"]]
            if video_rows else [],
            "action_instance_ids": sorted({row["instance_id"] for row in action_rows}),
            "exact_video_policy_start_matches": exact_video_matches,
            "first_action_shape": {name: [len(value), len(value[0])]
                                   for name, value in action_rows[0]["action"].items()},
            "true_counts": {name: counts[name] for name in event_values(state_rows[0])},
            "transitions": dict(transitions),
            "missing_action_fields": missing_actions,
            "missing_state_fields": missing_state,
            "sample_microwave": next(iter(state_rows[0]["microwaves"].values()), {}) if state_rows else {},
            "sample_popcorn": next(iter(state_rows[0]["popcorn"].values()), {}) if state_rows else {},
        }
    episode = next(row for row in episodes if identity(row) == (0, 0, 22000))
    video = run / episode["video_file"]
    frames = [row for row in read_jsonl(stages / "video_gpu0_env0.jsonl") if identity(row) == identity(episode)]
    decoded = subprocess.check_output(["ffmpeg", "-v", "error", "-i", str(video), "-frames:v", "3",
                                       "-threads", "1", "-f", "rawvideo", "-pix_fmt", "rgb24", "-"])
    stream = json.loads(subprocess.check_output(["ffprobe", "-v", "error", "-select_streams", "v:0",
        "-show_entries", "stream=width,height,nb_frames,r_frame_rate", "-of", "json", str(video)]))["streams"][0]
    assert int(stream["nb_frames"]) == len(frames)
    assert stream["width"] == 768 and stream["height"] == 280
    assert len(decoded) == 3 * 768 * 280 * 3
    assert [row["frame_index"] for row in frames] == list(range(len(frames)))
    assert all(old["sim_time_s"] < new["sim_time_s"] for old, new in zip(frames, frames[1:]))
    summary["decoded_sample"] = {"file": str(video), "stream": stream,
                                 "decoded_frames": 3, "steps": [row["action_steps"] for row in frames[:3]],
                                 "seconds": [row["sim_time_s"] for row in frames[:3]]}
    summary["snapshot_or_observation_candidates"] = [str(path.relative_to(run)) for path in sorted(run.rglob("*"))
                                                     if path.suffix in {".npy", ".npz", ".pt", ".h5", ".parquet"}]
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run", type=Path, help="Task40 evaluation run directory")
    args = parser.parse_args()
    print(json.dumps(audit_run(args.run), indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
