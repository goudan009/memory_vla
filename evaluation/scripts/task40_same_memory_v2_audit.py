"""Audit and render three independently restored Task40 branches."""
import argparse
from itertools import combinations
import json
from pathlib import Path
import subprocess

import numpy as np
from PIL import Image, ImageDraw, ImageFont

from task40_same_memory_audit import compare_arrays, compare_decision, load, phase


CAMERAS = ("left_wrist", "right_wrist", "head")
MICROWAVE = "microwave_hjjxmi_0"
FONT = "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"


def rgb_stats(left, right):
    return {name: compare_arrays(left[name], right[name]) for name in CAMERAS}


def summarize_trial(run, number, reference, intervention=False):
    metrics = run / "metrics"
    trial = load(metrics / f"trial_{number:02d}.json")
    memory_path = metrics / f"memory_compare_{number}.json"
    memory = load(memory_path) if memory_path.exists() else None
    if memory is not None:
        assert memory["equal"]
    restore = load(metrics / f"restore_compare_{number}.json")
    assert restore["controllers_equal"]
    folder = metrics / trial["branch"]
    rows = [json.loads(line) for line in (folder / "truth.jsonl").read_text().splitlines()]
    anchor, start = trial["anchor_decision"], trial["anchor_step"]
    assert [row["step"] for row in rows] == list(range(start + 1, start + 1 + trial["steps"]))
    assert len(rows) and trial["decisions"] == (trial["steps"] + 7) // 8
    retrieval = []
    for offset in range(trial["decisions"]):
        index = anchor + offset
        directory = folder / f"{index:04d}"
        meta = load(directory / "decision.json")
        expected = [start + 8 * offset, min(start + 8 * (offset + 1), start + trial["steps"])]
        assert meta["decision"] == index and meta["steps"] == expected
        assert meta["memory_after"]["next_decision"] == index + 1
        assert all(meta["memory_after"][bank]["last_written"] == index for bank in ("perceptual", "cognitive"))
        with np.load(directory / "executed.npz", allow_pickle=False) as arrays:
            actions = arrays["robot_r1"]
            assert actions.shape == (expected[1] - expected[0], 23) and np.isfinite(actions).all()
        if intervention:
            record, = load(directory / "retrieval.json")["observations"]
            assert record["timestep"] == index and record["history_read_enabled"]
            assert all(record[bank]["history_slots"] == min(memory[bank]["capacity"], offset + (0 if number == 2 else anchor))
                       for bank in ("perceptual", "cognitive"))
            retrieval.append({"decision": index, "retrieval": record,
                              "action_expert_input": load(directory / "action_expert_input.json")})
    with np.load(metrics / f"refresh_{number:02d}.npz", allow_pickle=False) as archive:
        refresh = [{name: archive[f"{index:02d}_{name}"] for name in CAMERAS} for index in range(1, 7)]
    with np.load(folder / f"{anchor:04d}" / "observation.npz", allow_pickle=False) as actual:
        for name in CAMERAS:
            assert np.array_equal(reference[name] if intervention else refresh[-1][name],
                                  actual[f"video.observation.images.rgb.{name}_256_256"][0, 0])
    closest = {}
    contacts = {}
    for arm in ("left", "right"):
        nearest_row = min(rows, key=lambda row: row["interaction"]["finger_origin_to_door_aabb_m"][arm])
        closest[arm] = {"step": nearest_row["step"],
                        "distance_m": nearest_row["interaction"]["finger_origin_to_door_aabb_m"][arm]}
        all_contact_steps = [row for row in rows if row["interaction"]["microwave_contacts"][arm]]
        door_steps = [row["step"] for row in all_contact_steps if any(
            contact["body"] == row["interaction"]["door_body"] for contact in row["interaction"]["microwave_contacts"][arm])]
        contacts[arm] = {"microwave_steps": [row["step"] for row in all_contact_steps], "door_steps": door_steps,
                         "bodies": sorted({contact["body"].split("/")[-1] for row in all_contact_steps
                                           for contact in row["interaction"]["microwave_contacts"][arm]})}
    anchor_row = restore["after"]["truth"]
    prior = {**anchor_row, "task_progress": rows[0]["task_progress"]}
    edges = [{"step": row["step"], "decision": row["decision"], "phase": phase(row)}
             for previous, row in zip([prior, *rows], rows) if phase(previous) != phase(row)]
    joint = [row["microwaves"][MICROWAVE]["joint_positions"][0] for row in [anchor_row, *rows]]
    result = {**trial, "memory_equal": True, "initial_phase": phase(prior), "phase_changes": edges,
              "closest_finger_origin_to_door_aabb": closest, "contacts": contacts,
              "door_joint_rad_min_max": [min(joint), max(joint)],
              "min_right_eef_to_microwave_origin_m": min(row["distance"]["eef_m"]["right"] for row in rows),
              "final_distance": rows[-1]["distance"],
              "door_opened_steps": [row["step"] for row in rows if row["microwaves"][MICROWAVE]["Open"]],
              "render_pass_vs_live_anchor": [rgb_stats(item, reference) for item in refresh],
              "render_pass_vs_previous": [rgb_stats(right, left) for left, right in zip(refresh, refresh[1:])]}
    if intervention:
        result.update(retrieval=retrieval, initial_history=load(metrics / f"history_{trial['branch']}.json"))
        refresh[-1] = reference
    return result, rows, refresh


def compare_trials(run, first, second, first_rows, second_rows, first_frames, second_frames):
    common = list(zip(first_rows, second_rows))
    anchor = first["anchor_decision"]
    paths = [run / "metrics" / item["branch"] for item in (first, second)]
    comparisons = {}
    for index in range(anchor, anchor + min(first["decisions"], second["decisions"])):
        left, right = (path / f"{index:04d}" for path in paths)
        first_meta, second_meta = (load(path / "decision.json") for path in (left, right))
        assert all(first_meta[key] == second_meta[key] for key in ("decision", "episode", "session", "language"))
        if first_meta["steps"] == second_meta["steps"]:
            comparisons[index] = compare_decision(left, right)
        else:
            with np.load(left / "executed.npz") as lhs, np.load(right / "executed.npz") as rhs:
                shared = min(len(lhs["robot_r1"]), len(rhs["robot_r1"]))
                comparisons[index] = {"shared_action_steps": shared, "executed_actions": {
                    key: compare_arrays(lhs[key][:shared], rhs[key][:shared]) for key in lhs.files}}

    def xyz(row, part):
        robot = row["robots"]["robot_r1"]
        return robot["position"][:2] if part == "base" else robot["end_effectors"]["right"][0]

    return {"common_steps": len(common),
            "phase_disagreement_steps": [left["step"] for left, right in common if phase(left) != phase(right)],
            "first_pass_rgb": rgb_stats(first_frames[0], second_frames[0]),
            "policy_first_rgb": rgb_stats(first_frames[-1], second_frames[-1]),
            "max_control_difference": max(item["executed_actions"]["robot_r1"]["max"] for item in comparisons.values()),
            "max_position_distance_m": {part: max(float(np.linalg.norm(np.asarray(xyz(left, part)) - xyz(right, part)))
                                                 for left, right in common) for part in ("right_eef", "base")},
            "decisions": comparisons}


def first_frame_image(run, trials, refresh):
    canvas = Image.new("RGB", (768, 42 + 3 * 284), "#101820")
    draw = ImageDraw.Draw(canvas)
    font = ImageFont.truetype(FONT, 17)
    for column, label in enumerate(("LEFT WRIST", "RIGHT WRIST", "HEAD")):
        draw.text((column * 256 + 10, 11), label, fill="white", font=font)
    for index, trial in enumerate(trials):
        top = 42 + index * 284
        draw.text((10, top + 3), f"{trial['branch']}  |  decision {trial['anchor_decision']}  |  render pass 6", fill="white", font=font)
        for column, name in enumerate(CAMERAS):
            canvas.paste(Image.fromarray(refresh[index][-1][name]), (column * 256, top + 28))
    canvas.save(run / "first_frame_comparison.png")


def comparison_video(run, trials, logs):
    width, height = 1536, 816
    frame_bytes = 768 * 256 * 3
    font = ImageFont.truetype(FONT, 21)
    detail = ImageFont.truetype(FONT, 16)
    readers = [subprocess.Popen(["ffmpeg", "-v", "error", "-threads", "1", "-i", str(run / "videos" / f"{trial['branch']}.mp4"),
                                "-f", "rawvideo", "-pix_fmt", "rgb24", "-threads", "1", "pipe:1"], stdout=subprocess.PIPE)
               for trial in trials]
    output = subprocess.Popen(["ffmpeg", "-v", "error", "-y", "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", f"{width}x{height}",
                               "-r", "30", "-i", "pipe:0", "-an", "-c:v", "libx264", "-crf", "18", "-pix_fmt", "yuv420p",
                               "-movflags", "+faststart", str(run / "comparison.mp4")], stdin=subprocess.PIPE)
    last_frames = [None] * len(readers)
    try:
        for index in range(max(map(len, logs))):
            canvas = Image.new("RGB", (width, height), "#101820")
            draw = ImageDraw.Draw(canvas)
            for column, (reader, rows, trial) in enumerate(zip(readers, logs, trials, strict=True)):
                active = index < len(rows)
                if active:
                    raw = reader.stdout.read(frame_bytes)
                    if len(raw) != frame_bytes:
                        raise RuntimeError(f"Video cut short: {trial['branch']} frame {index}")
                    last_frames[column] = Image.frombytes("RGB", (768, 256), raw)
                frame = last_frames[column]
                xcoord = 512 * column
                row = rows[min(index, len(rows) - 1)]
                microwave = row["microwaves"][MICROWAVE]
                opened = microwave["Open"]
                label = f"{trial['branch']}  d={row['decision']}  Open={opened}  On={microwave['ToggledOn']}"
                draw.text((xcoord + 10, 3), label, font=font, fill="#76edac" if opened else "white")
                draw.text((xcoord + 10, 27), f"step {row['step']}   " + ("live" if active else f"stopped: {trial['stop_reason']}"),
                          font=detail, fill="#b6c5d0")
                canvas.paste(frame.crop((512, 0, 768, 256)).resize((512, 512), Image.BICUBIC), (xcoord, 48))
                canvas.paste(frame.crop((0, 0, 512, 256)), (xcoord, 560))
            output.stdin.write(canvas.tobytes())
        for reader in readers:
            assert not reader.stdout.read(1) and reader.wait() == 0
    finally:
        output.stdin.close()
        result = output.wait()
        for reader in readers:
            reader.stdout.close()
            if reader.poll() is None:
                reader.terminate()
                reader.wait()
    if result:
        raise RuntimeError("Comparison video encoding failed")


def main(run, intervention=False):
    with np.load(run / "checkpoints" / "anchor_live_observation.npz", allow_pickle=False) as archive:
        reference = {name: archive[f"video.observation.images.rgb.{name}_256_256"][0, 0] for name in CAMERAS}
    trials, logs, refresh = zip(*(summarize_trial(run, number, reference, intervention) for number in (1, 2, 3)))
    policy_logs = list((run / "logs").glob("policy*.log"))
    memory_matches = sum(path.read_text().count("TASK40_MEMORY_MATCH 9") for path in policy_logs)
    assert memory_matches == 3, f"Expected three successful memory rebuilds, found {memory_matches}"
    pairs = {f"{trials[left]['branch']}_vs_{trials[right]['branch']}":
             compare_trials(run, trials[left], trials[right], logs[left], logs[right], refresh[left], refresh[right])
             for left, right in combinations(range(3), 2)}
    first_decision = {}
    if intervention:
        for left, right in combinations(trials, 2):
            directories = [run / "metrics" / trial["branch"] / f"{trial['anchor_decision']:04d}" for trial in (left, right)]
            result = {}
            for filename, field in (("observation.npz", "observation"), ("proposed.npz", "proposed_actions"),
                                    ("action_expert_input.npz", "action_expert_input")):
                with np.load(directories[0] / filename) as lhs, np.load(directories[1] / filename) as rhs:
                    assert lhs.files == rhs.files
                    result[field] = {key: compare_arrays(lhs[key], rhs[key]) for key in lhs.files}
            assert all(value["max"] == 0 for value in result["observation"].values())
            first_decision[f"{left['branch']}_vs_{right['branch']}"] = result
    (run / "metrics" / "comparison.json").write_text(json.dumps({"memory_matches": memory_matches,
        "memory_comparison_files": sorted(path.name for path in (run / "metrics").glob("memory_compare_*.json")),
        "trials": trials, "pairs": pairs, "first_decision": first_decision}, indent=2, allow_nan=False) + "\n")
    first_frame_image(run, trials, refresh)
    comparison_video(run, trials, logs)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run", type=Path)
    parser.add_argument("--memory-intervention", action="store_true")
    args = parser.parse_args()
    main(args.run, args.memory_intervention)
