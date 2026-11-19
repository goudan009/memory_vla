"""Versioned JSON results and episode journals shared by GPU queues."""

import fcntl
import json
import os
from pathlib import Path
import re
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent))
import b1k_metrics


def allocate_output(parent, model_name, task_id, *, dry_run=False):
    """Reserve a numbered directory atomically, including simultaneous launches."""
    parent = Path(parent)
    if not dry_run:
        parent.mkdir(parents=True, exist_ok=True)
    # One counter per task/version/method, across old model label spellings.
    pattern = re.compile(rf'(?:task{task_id}_eval|(?:groot|grootmeory)_task{task_id}_)([0-9]+)$')
    numbers = [int(match[1]) for path in parent.iterdir() if path.is_dir() and (match := pattern.fullmatch(path.name))] if parent.exists() else []
    number = max(numbers, default=0) + 1
    while True:
        output = parent / f"task{task_id}_eval{number:02d}"
        if dry_run:
            if not output.exists():
                return output
        else:
            try:
                output.mkdir()
                return output
            except FileExistsError:
                pass
        number += 1


def read_records(path):
    if not path.exists():
        return []
    records = []
    for line in path.read_text(encoding="utf-8").splitlines():
        try:
            records.append(json.loads(line))
        except json.JSONDecodeError:
            continue
    return records


def name_episode_video(source, episode, output=None):
    """Rename a closed recorder file using its explicitly recorded identity."""
    source = Path(source)
    name = (
        f"episode_{episode['episode_index']:03d}_instance{episode['instance_id']}"
        f"_seed{episode['seed']}_gpu{episode.get('gpu_index', 0)}"
        f"_env{episode['env_idx']}_s{int(episode['success'])}.mp4"
    )
    destination = source.with_name(name)
    try:
        source.rename(destination)
    except OSError as error:
        # A video failure must not discard the native episode outcome.
        print(f"VIDEO_NAME_ERROR source={source} error={error}", flush=True)
        return ""
    if output is None:
        return f"videos/{name}"
    attempt = source.relative_to(Path(output) / ".runtime").parts[0]
    return f"videos/{attempt}/{name}"


def update_results(output, *, updates=None, episode=None, video_source=None):
    """Append/fsync each episode and update aggregate counts under one run lock.

    Queues publish directly here before the next reset. The launcher can merge
    configuration, status and performance without losing concurrent episodes.
    A closed video is named under this same lock, using the public episode
    number. Its assigned video_file is also copied into the caller's episode.
    """
    output = Path(output)
    runtime = output / ".runtime"
    runtime.mkdir(exist_ok=True)
    with (runtime / "results.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        result_path = output / "results.json"
        result = json.loads(result_path.read_text()) if result_path.exists() else {}
        result.update(updates or {})
        journal = output / "episodes.jsonl"
        records = read_records(journal)
        if episode is not None:
            original_episode = episode
            # Detailed per-chunk samples belong to optional diagnostics, not the
            # normal per-episode journal. All outcome/identity fields are kept.
            episode = {key: value for key, value in episode.items()
                       if key not in ("env_chunk_times_s", "video_encoding_s")}
            episode["gpu_episode_index"] = episode.pop("episode_index")
            episode["episode_index"] = len(records) + 1
            if video_source is not None:
                episode["video_file"] = name_episode_video(video_source, episode, output)
                original_episode["video_file"] = episode["video_file"]
            with journal.open("a", encoding="utf-8") as handle:
                handle.write(json.dumps(episode, allow_nan=False) + "\n")
                handle.flush()
                os.fsync(handle.fileno())
            records.append(episode)
            if episode.get("failure_reason"):
                result.setdefault("failure_reason", episode["failure_reason"])
        else:
            journal.touch(exist_ok=True)
        valid = [record for record in records if record["valid"]]
        progress = [record["task_progress"] for record in valid if record["task_progress"] is not None]
        q_scores = [record["q_score"] for record in valid if record.get("q_score") is not None]
        successes = sum(record["success"] for record in valid)
        official = result.get("evaluation_mode") == "official-eval"
        result.update(
            recorded_episodes=len(records), completed_episodes=len(valid),
            invalid_episodes=len(records) - len(valid), successful_episodes=successes,
            official_episodes=len(valid) if official else 0,
            success_rate=successes / len(valid) if official and valid else None,
            mean_task_progress=sum(progress) / len(progress) if progress else None,
            mean_q_score=sum(q_scores) / len(q_scores) if official and q_scores else None,
            q_score_episodes=len(q_scores),
            missing_q_score_episodes=len(valid) - len(q_scores),
            b1k_metrics=b1k_metrics.summarize(records, result.get('evaluation_mode'), result.get('requested_episodes')),
            total_action_steps=sum(record["sim_steps"] for record in records),
            episodes="episodes.jsonl",
        )
        temporary = output / "results.json.tmp"
        with temporary.open("w", encoding="utf-8") as handle:
            json.dump(result, handle, indent=2, allow_nan=False)
            handle.write("\n")
        temporary.replace(result_path)
        return result
