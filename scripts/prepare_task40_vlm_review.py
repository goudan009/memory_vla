#!/usr/bin/env python3
"""Build a Task40 human-review set from recorded physics/action evidence."""
import csv
import hashlib
import json
import random
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "outputs/overnight_memory_task40_20261114/verification/task40_vlm_blind_review"
sys.path.insert(0, str(ROOT / "evaluation/scripts"))
import task_stage_diagnostics as task40  # noqa: E402


# Failed cases come from runs with per-frame video/action indexes. The control
# is the first completed successful Task40 run with recorded physics/actions.
FAILURE_RUNS = (
    "V_1/memory/a6000_task40_eval09",
    "V_1/memory/a6000_task40_eval10",
    "V_1/memory/task40_eval14",
)
CONTROL_RUN = "V_1/memory/task40_eval11"
FACT_NAMES = {
    "door_open": "微波炉门开",
    "bag_held": "机器人持袋",
    "bag_inside": "爆米花袋在炉内",
    "microwave_on": "微波炉启动",
}

def read_episodes(run):
    path = run / "episodes.jsonl"
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def file_hash(path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def action_rows(run, episode):
    path = run / "stages" / f"actions_task40_gpu{episode.get('gpu_index')}_env{episode.get('env_idx')}.jsonl"
    rows = task40.read_jsonl(path)
    rows = [row for row in rows if all(row.get(key) == episode.get(key)
            for key in ("instance_id", "seed", "env_idx"))]
    identities = {row.get("episode_number") for row in rows}
    if len(identities) > 1:
        return [], "ambiguous_action_trace_identity"
    return sorted(rows, key=lambda row: row.get("action_start", -1)), None


def vec_summary(value, dims):
    if not isinstance(value, list) or not value:
        return None
    vectors = value if isinstance(value[0], list) else [value]
    vectors = [v for v in vectors if isinstance(v, list) and len(v) >= dims]
    if not vectors:
        return None
    return [round(sum(float(v[i]) for v in vectors) / len(vectors), 4) for i in range(dims)]


def compact_action(row):
    action = row.get("action", {})
    result = {"action_start": row.get("action_start"), "action_end": row.get("action_end"),
              "sim_time_s": row.get("sim_time_s")}
    for key, dims in (("action.base", 3), ("action.left_arm", 3), ("action.right_arm", 3),
                      ("action.left_gripper", 1), ("action.right_gripper", 1)):
        summary = vec_summary(action.get(key), dims)
        if summary is not None:
            result[key] = summary
    return result


def stage_for_transition(changed, before, after):
    if "door_open" in changed:
        return "S4" if after.get("door_open") is True else "S8"
    if "bag_inside" in changed:
        return "S7"
    if "bag_held" in changed:
        return "S6" if after.get("bag_held") is True else "S7"
    if "microwave_on" in changed:
        return "S9"
    return "unknown"


def physical_events(run, episode, report, kind):
    traces, warning = task40._select_traces(run, episode)
    if warning or not traces:
        return []
    output = []
    previous = None
    for row in traces:
        facts = task40.physics(row)
        if previous is not None:
            changed = [name for name in FACT_NAMES if facts.get(name) != previous.get(name)]
            if changed:
                output.append(dict(
                    source_kind="physics_transition", action_step=int(row["action_steps"]),
                    stage_id=stage_for_transition(changed, previous, facts),
                    changed_facts=changed, previous_facts=previous, facts=facts,
                    explanation="B1K 物理状态采样发生变化；这是结果证据，不代表视频动作或因果已确认。"))
        previous = facts
    if kind == "failure":
        failure = report.get("first_observed_failure")
        if not failure:
            return []
        step = int(failure["action_steps"])
        point = min(traces, key=lambda row: abs(row["action_steps"] - step))
        facts = task40.physics(point)
        return [dict(source_kind="first_logged_anomaly", action_step=step,
                     stage_id=failure["stage_id"], changed_facts=[], previous_facts=None,
                     facts=facts, error_type=failure["error_type"],
                     explanation=failure["explanation"],
                     rule_evidence=failure.get("evidence", {}))]
    return output


def load_review_set():
    selected = []
    run_names = list(FAILURE_RUNS) + [CONTROL_RUN]
    for run_name in run_names:
        run = ROOT / "evaluation" / run_name
        if not (run / "episodes.jsonl").is_file():
            raise FileNotFoundError(run / "episodes.jsonl")
        results = json.loads((run / "results.json").read_text(encoding="utf-8"))
        scale = (results.get("configuration", {}).get("video_steps_per_render") or 24) * (
            results.get("configuration", {}).get("video_fps") or 20)
        is_control = run_name == CONTROL_RUN
        for episode in read_episodes(run):
            if episode.get("valid") is not True:
                continue
            report = task40.analyze_episode(run, episode)
            if is_control:
                if report["official"].get("b1k_success") is not True:
                    continue
                discovered = physical_events(run, episode, report, "control")
            else:
                if report["official"].get("b1k_success") is True:
                    continue
                discovered = physical_events(run, episode, report, "failure")
            if not discovered:
                continue
            video = run / episode["video_file"]
            if not video.is_file():
                continue
            frames = report["time_mapping"].get("frames", [])
            if not is_control and not frames:
                raise ValueError(f"失败样本缺少逐帧视频索引：{run_name} episode {episode['episode_index']}")
            actions, action_warning = action_rows(run, episode)
            if not actions:
                raise ValueError(f"缺少可唯一关联的动作记录：{run_name} episode {episode['episode_index']} ({action_warning})")
            for event in discovered:
                step = event["action_step"]
                seconds, approximate = task40.video_position(report["time_mapping"], step)
                nearby = [row for row in actions if row.get("action_end", -1) >= step - 320
                          and row.get("action_start", float("inf")) <= step + 320]
                traces, _ = task40._select_traces(run, episode)
                nearby_trace = [row for row in traces if abs(row["action_steps"] - step) <= 640]
                selected.append(dict(
                    run=run, run_name=run_name, episode=episode, video=video,
                    report=report, scale=scale, event=event, event_seconds=seconds,
                    time_approximate=approximate, actions=[compact_action(row) for row in nearby],
                    timeline=[dict(action_steps=int(row["action_steps"]),
                                   video_seconds=task40.video_position(report["time_mapping"], row["action_steps"])[0],
                                   approximate=task40.video_position(report["time_mapping"], row["action_steps"])[1],
                                   facts=task40.physics(row)) for row in nearby_trace],
                    control=is_control))
    failures = [row for row in selected if not row["control"]]
    controls = [row for row in selected if row["control"]]
    if len(failures) != 17 or len(controls) != 6:
        raise ValueError(f"Expected 17 first-anomaly events + 6 successful-state transitions; got {len(failures)} + {len(controls)}")
    # Blind event IDs and episode order while retaining reproducibility.
    random.Random(40261115).shuffle(selected)
    return selected


def probe_duration(path):
    value = subprocess.check_output([
        "ffprobe", "-v", "error", "-show_entries", "format=duration",
        "-of", "default=noprint_wrappers=1:nokey=1", str(path),
    ], text=True)
    return float(value.strip())


def encode_review_video(source, destination, start=None, duration=None):
    command = ["ffmpeg", "-hide_banner", "-loglevel", "error", "-y"]
    if start is not None:
        command += ["-ss", f"{start:.6f}"]
    command += ["-i", str(source)]
    if duration is not None:
        command += ["-t", f"{duration:.6f}"]
    command += [
        "-vf", "crop=iw:ih-20:0:0", "-an", "-c:v", "libx264", "-preset",
        "ultrafast", "-crf", "22", "-pix_fmt", "yuv420p", "-movflags",
        "+faststart", str(destination),
    ]
    subprocess.run(command, check=True, timeout=300)


def main():
    from task40_sequence_review import ACTIONS, build_sequence_page

    OUT.mkdir(parents=True, exist_ok=True)
    for folder in ("segments", "full_episodes"):
        (OUT / folder).mkdir(parents=True, exist_ok=True)
    selected = load_review_set()
    groups = {}
    for row in selected:
        key = (row["run_name"], row["episode"]["episode_index"])
        groups.setdefault(key, []).append(row)
    merged = []
    for key, rows in groups.items():
        source = rows[0]["video"]
        source_duration = probe_duration(source)
        windows = sorted([
            {"start": max(0.0, row["event_seconds"] - 3),
             "end": min(source_duration, row["event_seconds"] + 3),
             "events": [row]} for row in rows
        ], key=lambda window: window["start"])
        joined = []
        for window in windows:
            if joined and window["start"] <= joined[-1]["end"]:
                joined[-1]["end"] = max(joined[-1]["end"], window["end"])
                joined[-1]["events"].extend(window["events"])
            else:
                joined.append(window)
        for item in joined:
            item.update(key=key, source=source, source_duration=source_duration)
        merged.extend(joined)
    random.Random(40261115).shuffle(merged)
    segments, audit_segments, csv_rows = [], [], []
    for index, item in enumerate(merged, 1):
        sid = f"C{index:02d}"
        span = item["end"] - item["start"]
        short_video = OUT / "segments" / f"{sid}.mp4"
        full_video = OUT / "full_episodes" / f"sequence_{sid}.mp4"
        encode_review_video(item["source"], short_video, item["start"], span)
        encode_review_video(item["source"], full_video)
        lead = item["events"][0]
        mapping = lead["report"]["time_mapping"]
        fps = (mapping.get("frames") or [{"video_fps": 20}])[0]["video_fps"]
        events = [dict(row["event"], event_seconds=row["event_seconds"],
                       time_approximate=row["time_approximate"])
                  for row in item["events"]]
        run, episode_index = item["key"]
        episode = lead["episode"]
        source_hash = file_hash(item["source"])
        segments.append({
            "segment_id": sid, "start": item["start"], "end": item["end"],
            "duration": probe_duration(short_video),
            "full_duration": probe_duration(full_video),
            "video_url": f"segments/{sid}.mp4",
            "full_video_url": f"full_episodes/sequence_{sid}.mp4",
            "source_episode_id": f"{run}:episode{episode_index}",
            "source_video_sha256": source_hash,
            "fps": fps, "scale": lead["scale"],
            "frames": [{"frame_index": f["frame_index"], "action_steps": f["action_steps"]}
                       for f in mapping.get("frames", [])],
        })
        audit_segments.append({
            "segment_id": sid, "source_run": run, "episode_index": episode_index,
            "instance_id": episode.get("instance_id"), "seed": episode.get("seed"),
            "source_video": str(item["source"].relative_to(ROOT)),
            "source_video_sha256": source_hash,
            "source_video_duration": item["source_duration"],
            "clip_start_seconds": item["start"], "clip_end_seconds": item["end"],
            "clip_duration_seconds": segments[-1]["duration"],
            "full_video_url": segments[-1]["full_video_url"],
            "anchor_events": [{
                "source_kind": e["source_kind"], "action_step": e["action_step"],
                "video_seconds": e["event_seconds"],
                "time_mapping_approximate": e["time_approximate"],
                "logged_stage_not_human_label": e.get("stage_id"), "facts": e["facts"],
            } for e in events],
        })
        csv_rows.append({
            "video_id": sid, "source_episode_id": segments[-1]["source_episode_id"],
            "coverage": "", "quality": "unreviewed", "steps_json": "[]", "reviewed": False,
        })
    audit = {
        "schema_version": 4, "taxonomy_version": "task40-observed-sequence-v1",
        "selection_protocol": {
            "failed_cases": "3 existing runs; first logged anomaly per eligible failed episode.",
            "control_cases": "task40_eval11 successful episode physics transitions.",
            "clip_protocol": "Overlapping +/-3s windows are merged. Full episodes are the default annotation view; clips are navigation aids.",
            "human_label_protocol": "Record visible actions in actual chronological order; choose one outcome per action, optionally a visible detail and source timestamp. Repeated actions and revisits are allowed. No fixed phase order or inferred cause is imposed.",
            "model_predictions_hidden": True,
        },
        "segment_count": len(segments), "segments": audit_segments,
    }
    (OUT / "source_key.json").write_text(json.dumps(audit, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    (OUT / "sequence_schema.json").write_text(json.dumps({
        "taxonomy_version": audit["taxonomy_version"], "actions": ACTIONS,
        "step_fields": ["order", "action", "state", "detail", "source_time_seconds", "notes"],
        "time_axis": "source video seconds; source_time_seconds is optional",
    }, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    with (OUT / "annotation.csv").open("w", newline="", encoding="utf-8-sig") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(csv_rows[0]))
        writer.writeheader()
        writer.writerows(csv_rows)
    (OUT / "README.md").write_text("""# Task40 按画面顺序标注

默认看完整回合，短片只用于定位。只记录实际看到的行为，不根据日志或正常流程补全动作。

1. 选择视频是从头开始、从中途开始、还是看不出。
2. 点击“添加下一步”，选择动作和对应结果。例如：到达厨房 → 前往袋子 → 抓袋（反复后成功）→ 返回微波炉 → 放入（反复仍失败；门关闭挡住）。
3. 可以重复添加同一动作，插入遗漏动作，上下移动步骤或删除。每步可选记下当前原视频时间，也可只排顺序。
4. 关门受阻、打不开、拿不起来、拿起后掉落、一次成功、多次后成功、失败次数不明等均有选项。“其他动作”写一句说明；看不清就选未知。
5. 排好之后勾选“这段实际看到的动作已按顺序记完”，导出顺序标注 JSON。未完成的草稿也可以导出。

动作与结果选项见 sequence_schema.json；人工结果使用 schema_version=4，steps 按真实顺序保存。不要求人工推断记忆错误或动作错误，后续只能结合这份观察序列、物理状态和动作日志分析。旧阶段标签不包含顺序，不能自动转换成本版序列；旧浏览器草稿保留在原键下。

18 个回合来自已有缓存：17 个失败回合及 1 个成功回合，23 个日志锚点合并成 18 个定位片段。非代表性随机样本，日志阶段仅保留在 source_key.json 中。没有新增训练或仿真。
""", encoding="utf-8")
    (OUT / "index.html").write_text(build_sequence_page(segments), encoding="utf-8")
    print(f"Prepared {len(segments)} full-episode sequence reviews / {len(selected)} event anchors at {OUT}")


if __name__ == "__main__":
    main()
