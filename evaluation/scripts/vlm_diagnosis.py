"""VLM 视觉诊断：把评估录像抽帧送给 A6000 上的 Qwen3-VL 服务，拿结构化标签。

支持单集诊断与整轮批量诊断；结果写到 run 目录下的 vlm_diagnosis/，不改动 results.json。
"""
from __future__ import annotations

import base64
import json
import re
import subprocess
import tempfile
import threading
import time
import urllib.request
from collections import Counter
from pathlib import Path

import task_prompt

DEFAULT_ENDPOINT = "http://100.100.2.30:8901"
DEFAULT_NOTES = ("第三人称视角，机械臂在房间里操作；录像为多倍速加速回放，"
                 "画面动作比真实快很多，不要据此判断为乱动或失控")


SEG_FRAMES = 40
MAX_TOTAL = 300


def _extract_all(video: Path, max_total: int = MAX_TOTAL, width: int = 768) -> list[str]:
    """把整条录像的帧尽量全取出来：帧数超过 max_total 才等间隔抽样。"""
    probe = subprocess.run(
        ["ffprobe", "-v", "error", "-select_streams", "v:0",
         "-show_entries", "stream=nb_frames,r_frame_rate",
         "-show_entries", "format=duration", "-of", "default=nw=1:nk=1", str(video)],
        capture_output=True, text=True, timeout=60,
    )
    values = [v.strip() for v in probe.stdout.split() if v.strip()]
    total = 0
    for value in values:
        try:
            total = max(total, int(float(value)))
        except ValueError:
            continue
    step = max(1, -(-total // max_total)) if total else 1
    with tempfile.TemporaryDirectory() as tmp:
        out = Path(tmp)
        subprocess.run(
            ["ffmpeg", "-v", "error", "-i", str(video), "-vsync", "0",
             "-vf", f"select='not(mod(n\\,{step}))',scale={width}:-2",
             "-q:v", "4", "-y", str(out / "f%04d.jpg")],
            capture_output=True, timeout=900,
        )
        files = sorted(out.glob("f*.jpg"))
        frames = [base64.b64encode(f.read_bytes()).decode() for f in files]
    if not frames:
        raise RuntimeError("抽帧失败，没有可用的画面")
    return frames


def _post_json(url: str, payload: dict, timeout: int = 600) -> dict:
    data = json.dumps(payload).encode()
    request = urllib.request.Request(
        url, data=data, headers={"Content-Type": "application/json"}, method="POST"
    )
    # This is a direct connection to the diagnostic machine on the private LAN.
    # urllib does not understand the CIDR entry in the shell's NO_PROXY setting.
    with urllib.request.build_opener(urllib.request.ProxyHandler({})).open(request, timeout=timeout) as response:
        return json.loads(response.read().decode())


def _get_json(url: str, timeout: int = 10) -> dict:
    with urllib.request.build_opener(urllib.request.ProxyHandler({})).open(url, timeout=timeout) as response:
        return json.loads(response.read().decode())


def health(endpoint: str = DEFAULT_ENDPOINT) -> dict:
    try:
        value = _get_json(endpoint.rstrip("/") + "/health")
        return {"ok": bool(value.get("ok")), "model": value.get("model"),
                "gpu_mem_gb": value.get("gpu_mem_gb"), "endpoint": endpoint}
    except Exception as error:  # noqa: BLE001 - 网络失败原样上报给界面
        return {"ok": False, "error": f"{type(error).__name__}: {error}", "endpoint": endpoint}


def _instruction(task_name: str, given: str) -> str:
    if given and given != "unknown":
        return given
    if task_name:
        return task_prompt.instruction_for(task_name)
    return "unknown"


def _grab_line(text: str, prefix: str) -> str:
    for line in (text or "").splitlines():
        line = line.strip()
        if line.startswith(prefix):
            return line[len(prefix):].strip()
    return ""

def _segments(frames: list[str], size: int = SEG_FRAMES) -> list[list[str]]:
    return [frames[i:i + size] for i in range(0, len(frames), size)]


def _analyze(endpoint: str, payload: dict, timeout: int = 1800) -> dict:
    return _post_json(endpoint.rstrip("/") + "/analyze", payload, timeout=timeout)


def diagnose(run_dir: Path, video: Path, endpoint: str = DEFAULT_ENDPOINT,
             task: str = "", episode: str = "", instruction: str = "",
             frames: int = 0, notes: str = "", task_name: str = "",
             force: bool = False) -> dict:
    """全帧分段细看，再用分段记录做整体判断。frames 参数保留兼容，0 表示不限制。"""
    if not force:
        hit = cached(run_dir, video)
        if hit is not None:
            return hit
    if not video.is_file():
        raise ValueError(f"录像不存在：{video.name}")

    started = time.time()
    images = _extract_all(video, max_total=frames if frames and frames >= 60 else MAX_TOTAL)
    instruction_text = _instruction(task_name, instruction)
    notes_text = notes or DEFAULT_NOTES
    chunks = _segments(images)
    segments: list[dict] = []
    for index, chunk in enumerate(chunks):
        label = f"第 {index + 1}/{len(chunks)} 段（第 {index * SEG_FRAMES + 1}-{index * SEG_FRAMES + len(chunk)} 帧）"
        value: dict = {"_ok": False, "error": "未执行"}
        for attempt in range(2):
            try:
                value = _analyze(endpoint, {
                    "task": task, "episode": episode, "group": "vlm-segment",
                    "mode": "segment", "label": label,
                    "instruction": instruction_text, "notes": notes_text,
                    "frames": chunk,
                })
            except Exception as error:  # noqa: BLE001 - 单段失败不中断
                value = {"_ok": False, "error": f"{type(error).__name__}: {error}"}
            if value.get("_ok"):
                break
        raw = (value.get("raw") or "").strip()
        actions = _grab_line(raw, "动作：") or raw
        contact = _grab_line(raw, "接触：")
        judge = _grab_line(raw, "判断：")
        delta = None
        numbers = [int(x) for x in re.findall(r"\d{1,3}", judge) if int(x) <= 100]
        if numbers:
            delta = numbers[-1]
        segments.append({
            "label": label, "frames": len(chunk),
            "actions": actions, "contact": contact, "judge": judge,
            "progress_delta": delta,
            "ok": bool(value.get("_ok")),
            "error": None if value.get("_ok") else str(value.get("error"))[:200],
        })

    history = "\n".join(
        f"{seg['label']}：{seg.get('actions') or '（这段没有拿到描述）'}"
        f"｜接触={seg.get('contact') or 'none'}"
        f"｜判断={seg.get('judge') or '无'}"
        for seg in segments
    )

    # 汇总时给每段的代表帧：段首、段中、段尾
    reps: list[str] = []
    for chunk in chunks:
        picks = {0, len(chunk) // 2, len(chunk) - 1}
        reps.extend([chunk[i] for i in sorted(picks) if 0 <= i < len(chunk)])
    reps = reps[:36]

    result = _analyze(endpoint, {
        "task": task, "episode": episode, "group": "vlm",
        "instruction": instruction_text, "notes": notes_text,
        "history": history, "frames": reps,
    })
    result["_video"] = video.name
    result["_frames"] = len(images)
    result["_segments_count"] = len(chunks)
    result["_segments"] = segments
    result["_task_name"] = task_name
    result["_cost_s"] = round(time.time() - started, 1)

    out_dir = run_dir / "vlm_diagnosis"
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / f"{video.stem.replace('/', '_')}.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return result


def cached(run_dir: Path, video: Path) -> dict | None:
    path = run_dir / "vlm_diagnosis" / f"{video.stem.replace('/', '_')}.json"
    if not path.is_file():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None


# ---------------------------------------------------------------- 批量与汇总

BATCH_LOCK = threading.Lock()


def read_episodes(run_dir: Path) -> list[dict]:
    path = run_dir / "episodes.jsonl"
    if not path.is_file():
        return []
    items: list[dict] = []
    for line in path.read_text(errors="ignore").splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            row = json.loads(line)
        except ValueError:
            continue
        if row.get("video_file"):
            items.append(row)
    return items


def _batch_path(run_dir: Path) -> Path:
    return run_dir / "vlm_diagnosis" / "_batch.json"


def batch_state(run_dir: Path) -> dict:
    path = _batch_path(run_dir)
    if not path.is_file():
        return {"state": "idle", "total": 0, "done": 0}
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {"state": "idle", "total": 0, "done": 0}


def _write_state(run_dir: Path, state: dict) -> None:
    out = run_dir / "vlm_diagnosis"
    out.mkdir(parents=True, exist_ok=True)
    _batch_path(run_dir).write_text(
        json.dumps(state, ensure_ascii=False, indent=1), encoding="utf-8")


def start_batch(run_dir: Path, endpoint: str = DEFAULT_ENDPOINT, frames: int = 0,
                task_name: str = "", force: bool = False) -> dict:
    with BATCH_LOCK:
        current = batch_state(run_dir)
        if current.get("state") == "running" and not force:
            return current
    episodes = read_episodes(run_dir)
    if not episodes:
        raise ValueError("该运行目录没有 episodes.jsonl 或没有录像")
    state = {"state": "running", "total": len(episodes), "done": 0, "failed": 0,
             "current": "", "started_at": time.strftime("%Y-%m-%d %H:%M:%S"),
             "finished_at": "", "summary": None, "items": []}
    _write_state(run_dir, state)
    threading.Thread(
        target=_batch_worker,
        args=(run_dir, episodes, endpoint, frames, task_name, force),
        daemon=True,
    ).start()
    return state


def _batch_worker(run_dir: Path, episodes: list[dict], endpoint: str, frames: int,
                  task_name: str, force: bool) -> None:
    for row in episodes:
        state = batch_state(run_dir)
        if state.get("state") != "running":
            return
        video_rel = row.get("video_file", "")
        video = run_dir / video_rel
        name = row.get("task") or task_name
        state["current"] = video_rel
        _write_state(run_dir, state)
        try:
            value = diagnose(
                run_dir, video, endpoint=endpoint,
                task=str(row.get("task_id", "")),
                episode=str(row.get("episode_index", "")),
                task_name=name, frames=frames, force=force,
            )
            record = {
                "episode": row.get("episode_index"),
                "video": video_rel,
                "official_progress": row.get("task_progress"),
                "official_success": row.get("success"),
                "q_score": row.get("q_score"),
                "behavior_class": value.get("behavior_class"),
                "visual_progress": value.get("visual_progress"),
                "execution_quality": value.get("execution_quality"),
                "failure_stage": value.get("failure_stage"),
                "target_touched": value.get("target_touched"),
                "stuck": value.get("stuck"),
                "evidence": value.get("evidence"),
                "frames": value.get("_frames"),
                "segments": value.get("_segments_count"),
                "cost_s": value.get("_cost_s"),
                "ok": value.get("_ok", False),
            }
        except Exception as error:  # noqa: BLE001 - 单集失败不中断整轮
            state = batch_state(run_dir)
            state["failed"] = int(state.get("failed", 0)) + 1
            state["done"] = int(state.get("done", 0)) + 1
            state.setdefault("items", []).append({
                "episode": row.get("episode_index"), "video": video_rel,
                "error": f"{type(error).__name__}: {error}", "ok": False,
            })
            _write_state(run_dir, state)
            continue
        state = batch_state(run_dir)
        state["done"] = int(state.get("done", 0)) + 1
        state.setdefault("items", []).append(record)
        _write_state(run_dir, state)

    state = batch_state(run_dir)
    if state.get("state") != "running":
        return
    state["state"] = "complete"
    state["current"] = ""
    state["finished_at"] = time.strftime("%Y-%m-%d %H:%M:%S")
    state["summary"] = summarize(state.get("items") or [], task_name)
    _write_state(run_dir, state)
    try:
        (run_dir / "vlm_diagnosis" / "_summary.json").write_text(
            json.dumps(state["summary"], ensure_ascii=False, indent=1), encoding="utf-8")
    except OSError:
        pass


def _pearson(xs: list[float], ys: list[float]) -> float | None:
    if len(xs) < 3 or len(xs) != len(ys):
        return None
    n = len(xs)
    mx, my = sum(xs) / n, sum(ys) / n
    numerator = sum((x - mx) * (y - my) for x, y in zip(xs, ys))
    dx = sum((x - mx) ** 2 for x in xs) ** 0.5
    dy = sum((y - my) ** 2 for y in ys) ** 0.5
    if dx == 0 or dy == 0:
        return None
    return round(numerator / (dx * dy), 3)


def _mean(values: list) -> float | None:
    numbers = [v for v in values if isinstance(v, (int, float))]
    if not numbers:
        return None
    return round(sum(numbers) / len(numbers), 2)


def summarize(items: list[dict], task_name: str = "") -> dict:
    good = [item for item in items if item.get("ok")]
    behaviors = Counter(item.get("behavior_class") or "unknown" for item in good)
    stages = Counter(item.get("failure_stage") or "unknown" for item in good)
    pairs = [(item.get("official_progress"), item.get("visual_progress")) for item in good]
    pairs = [(o, v) for o, v in pairs
             if isinstance(o, (int, float)) and isinstance(v, (int, float))]
    stats = {
        "task_name": task_name,
        "episodes": len(items),
        "diagnosed": len(good),
        "failed": len(items) - len(good),
        "official_success": sum(1 for item in good if item.get("official_success")),
        "official_progress_mean": _mean([item.get("official_progress") for item in good]),
        "visual_progress_mean": _mean([item.get("visual_progress") for item in good]),
        "execution_quality_mean": _mean([item.get("execution_quality") for item in good]),
        "target_touched_ratio": (round(sum(1 for item in good if item.get("target_touched")) / len(good), 2)
                                 if good else None),
        "stuck_ratio": (round(sum(1 for item in good if item.get("stuck")) / len(good), 2)
                        if good else None),
        "behavior_distribution": dict(behaviors.most_common()),
        "failure_stage_distribution": dict(stages.most_common()),
        "progress_correlation": _pearson([p[0] for p in pairs], [p[1] for p in pairs]),
    }
    stats["conclusion"] = _conclusion(stats, len(pairs))
    return stats


def _conclusion(stats: dict, pair_count: int) -> str:
    if not stats.get("diagnosed"):
        return "没有可用的诊断结果，先确认 A6000 上的 VLM 服务在线。"
    parts: list[str] = []
    success = stats.get("official_success") or 0
    progress = stats.get("official_progress_mean")
    visual = stats.get("visual_progress_mean")
    quality = stats.get("execution_quality_mean")
    parts.append(f"共 {stats['diagnosed']} 集有诊断结果，官方判定成功 {success} 集")
    if progress is not None:
        parts.append(f"官方平均进度 {progress}%")
    if visual is not None:
        parts.append(f"视觉估计平均进度 {visual}%")
    if quality is not None:
        parts.append(f"视觉平均执行质量 {quality}/100")
    if stats.get("target_touched_ratio") is not None:
        parts.append(f"碰到目标物 {int(stats['target_touched_ratio'] * 100)}% 的回合")
    if stats.get("stuck_ratio"):
        parts.append(f"{int(stats['stuck_ratio'] * 100)}% 的回合被判为卡住")
    top_behavior = next(iter(stats.get("behavior_distribution", {})), None)
    if top_behavior:
        parts.append(f"最常见行为是 {top_behavior}")
    top_stage = next((key for key in stats.get("failure_stage_distribution", {})
                      if key not in ("none", "unknown")), None)
    if top_stage:
        parts.append(f"主要卡在 {top_stage} 阶段")
    correlation = stats.get("progress_correlation")
    if correlation is not None and pair_count >= 3:
        if correlation >= 0.5:
            parts.append(f"视觉进度与官方进度一致性较高（r={correlation}）")
        elif correlation >= 0.2:
            parts.append(f"视觉进度与官方进度弱相关（r={correlation}）")
        else:
            parts.append(f"视觉进度与官方进度基本不相关（r={correlation}），"
                         "视觉判断只能当辅助线索，不能替代官方进度")
    line = "，".join(parts) + "。"
    if success == 0 and (progress or 0) < 20:
        line += "这一轮没有任何成功回合，官方进度也低，属于策略整体没做起来，不是个别回合失误。"
    elif success == 0:
        line += "这一轮没有成功回合，但进度不算低，说明能推进到中途却完不成最终目标。"
    elif success < stats["diagnosed"]:
        line += "成功和失败都有，可以对比两类回合的视觉差异来定位问题。"
    return line
