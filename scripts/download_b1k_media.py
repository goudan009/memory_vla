#!/usr/bin/env python3
"""
Download BEHAVIOR-1K 2025 Challenge media files (RGB videos only)
For the 38 available tasks based on existing parquet files
"""
import os
import sys
import json
import time
from pathlib import Path
from collections import defaultdict
import subprocess
import pyarrow.parquet as pq

# Unset all proxy variables and use HF mirror
for k in ['http_proxy', 'https_proxy', 'HTTP_PROXY', 'HTTPS_PROXY', 'ALL_PROXY', 'all_proxy']:
    os.environ.pop(k, None)
os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'

from huggingface_hub import hf_hub_download, HfFileSystem

# Configuration
REPO_ID = "behavior-1k/2025-challenge-demos"
REPO_TYPE = "dataset"
REVISION = "main"
PARQUET_ROOT = Path("/nvme3/projects/SERF-VLA/datasets/2025-BEHAVIOR-1K-CHALLENGE/data")
DOWNLOAD_ROOT = Path("/nvme3/shared/b1k2025/demos")
PROJECT_ROOT = Path("/nvme3/projects/memory_vla")
OUTPUT_DIR = PROJECT_ROOT / "outputs/b1k_download"
STATUS_FILE = OUTPUT_DIR / "status.json"
LOG_FILE = OUTPUT_DIR / "download.log"

# RGB camera keys from repository structure: observation.images.rgb.{camera}
VIDEO_KEYS = ["observation.images.rgb.head", "observation.images.rgb.left_wrist", "observation.images.rgb.right_wrist"]
VIDEO_KEY_SHORT = ["head", "left_wrist", "right_wrist"]

def log(msg):
    """Log to both console and file"""
    timestamp = time.strftime("%Y-%m-%d %H:%M:%S")
    log_msg = f"[{timestamp}] {msg}"
    print(log_msg)
    with open(LOG_FILE, "a") as f:
        f.write(log_msg + "\n")

def get_available_tasks():
    """Get list of 38 tasks that have parquet data"""
    tasks = []
    for task_dir in sorted(PARQUET_ROOT.iterdir()):
        if task_dir.is_dir() and task_dir.name.startswith("task-"):
            task_idx = int(task_dir.name.split("-")[1])
            tasks.append(task_idx)
    return tasks

def get_episodes_for_task(task_idx):
    """Get list of episode indices from parquet files for a task"""
    task_dir = PARQUET_ROOT / f"task-{task_idx:04d}"
    episodes = []
    for pq_file in sorted(task_dir.glob("episode_*.parquet")):
        # Extract episode index from filename like "episode_00000010.parquet"
        ep_idx = int(pq_file.stem.split("_")[1])
        episodes.append(ep_idx)
    return episodes

def query_actual_size(task_idx, episodes):
    """Query actual file sizes from HuggingFace for a task sample"""
    try:
        fs = HfFileSystem()
        sample_eps = episodes[:min(5, len(episodes))]
        total_size = 0
        count = 0
        for ep_idx in sample_eps:
            for video_key in VIDEO_KEYS:
                remote_path = f"datasets/{REPO_ID}/videos/task-{task_idx:04d}/{video_key}/episode_{ep_idx:08d}.mp4"
                try:
                    info = fs.info(remote_path)
                    total_size += info['size']
                    count += 1
                except:
                    pass
        if count > 0:
            avg_size = total_size / count
            return avg_size
    except:
        pass
    return None

def estimate_download_size(tasks, episodes_per_task):
    """Estimate total download size by querying actual remote files"""
    total_episodes = sum(len(eps) for eps in episodes_per_task.values())

    # Try to get actual size from first task sample
    if tasks:
        avg_size = query_actual_size(tasks[0], episodes_per_task[tasks[0]])
        if avg_size:
            total_gb = (avg_size * total_episodes * 3) / (1024**3)
            log(f"Estimated download size: {total_gb:.1f} GB based on remote file sizes (avg {avg_size/1024/1024:.1f} MB per video)")
            return total_gb

    # Fallback rough estimate
    estimated_mb = total_episodes * 20
    estimated_gb = estimated_mb / 1024
    log(f"Estimated download size: ~{estimated_gb:.1f} GB (rough estimate)")
    return estimated_gb

def download_video(task_idx, episode_idx, video_key, local_path):
    """Download a single video file from HuggingFace"""
    remote_path = f"videos/task-{task_idx:04d}/{video_key}/episode_{episode_idx:08d}.mp4"

    try:
        downloaded_path = hf_hub_download(
            repo_id=REPO_ID,
            repo_type=REPO_TYPE,
            revision=REVISION,
            filename=remote_path,
            local_dir=DOWNLOAD_ROOT,
            local_dir_use_symlinks=False,
        )
        return True, downloaded_path
    except Exception as e:
        return False, str(e)

def load_status():
    """Load download status from JSON"""
    if STATUS_FILE.exists():
        with open(STATUS_FILE) as f:
            return json.load(f)
    return {
        "started_at": None,
        "last_update": None,
        "completed_tasks": [],
        "current_task": None,
        "downloads": {},  # {task_idx: {episode_idx: {video_key: status}}}
        "errors": [],
        "total_downloaded": 0,
        "total_failed": 0,
    }

def save_status(status):
    """Save download status to JSON"""
    status["last_update"] = time.time()
    with open(STATUS_FILE, "w") as f:
        json.dump(status, f, indent=2)

def download_task_videos(task_idx, episodes, status):
    """Download all RGB videos for a task"""
    log(f"\n{'='*60}")
    log(f"Starting task {task_idx:04d} with {len(episodes)} episodes")
    log(f"{'='*60}")

    status["current_task"] = task_idx
    task_key = str(task_idx)

    if task_key not in status["downloads"]:
        status["downloads"][task_key] = {}

    task_success_count = 0
    task_fail_count = 0

    for ep_idx in episodes:
        ep_key = str(ep_idx)
        if ep_key not in status["downloads"][task_key]:
            status["downloads"][task_key][ep_key] = {}

        for i, video_key in enumerate(VIDEO_KEYS):
            video_key_short = VIDEO_KEY_SHORT[i]

            # Skip if already downloaded
            if video_key in status["downloads"][task_key][ep_key]:
                if status["downloads"][task_key][ep_key][video_key].get("success"):
                    task_success_count += 1
                    continue

            # Create local directory
            local_dir = DOWNLOAD_ROOT / "videos" / f"task-{task_idx:04d}" / video_key
            local_dir.mkdir(parents=True, exist_ok=True)
            local_path = local_dir / f"episode_{ep_idx:08d}.mp4"

            # Skip if file exists with valid size
            if local_path.exists():
                file_size = local_path.stat().st_size
                if file_size > 100000:  # > 100KB considered valid
                    status["downloads"][task_key][ep_key][video_key] = {
                        "success": True,
                        "path": str(local_path),
                        "size": file_size,
                        "skipped": True,
                    }
                    status["total_downloaded"] += 1
                    task_success_count += 1
                    continue

            # Download
            log(f"Downloading task-{task_idx:04d} episode-{ep_idx:08d} {video_key_short}...")
            success, result = download_video(task_idx, ep_idx, video_key, local_path)

            if success:
                file_size = Path(result).stat().st_size if Path(result).exists() else 0
                status["downloads"][task_key][ep_key][video_key] = {
                    "success": True,
                    "path": str(result),
                    "size": file_size,
                }
                status["total_downloaded"] += 1
                task_success_count += 1
                log(f"  ✓ Downloaded {file_size / 1024 / 1024:.2f} MB")
            else:
                status["downloads"][task_key][ep_key][video_key] = {
                    "success": False,
                    "error": result,
                }
                status["total_failed"] += 1
                task_fail_count += 1
                status["errors"].append({
                    "task": task_idx,
                    "episode": ep_idx,
                    "video_key": video_key,
                    "error": result,
                })
                log(f"  ✗ Failed: {result}")

            # Save status every 10 downloads
            if (status["total_downloaded"] + status["total_failed"]) % 10 == 0:
                save_status(status)

    # Mark task as completed only if no failures
    expected_videos = len(episodes) * len(VIDEO_KEYS)
    if task_fail_count == 0 and task_success_count == expected_videos:
        status["completed_tasks"].append(task_idx)
        log(f"Task {task_idx:04d} completed successfully ({task_success_count}/{expected_videos} videos)")
    else:
        log(f"Task {task_idx:04d} finished with issues: {task_success_count} success, {task_fail_count} failed (expected {expected_videos})")

    status["current_task"] = None
    save_status(status)

    return task_fail_count == 0

def verify_first_task(task_idx, episodes):
    """Verify videos from the first task against parquet frame counts"""
    log(f"\n{'='*60}")
    log(f"Verifying task {task_idx:04d} videos against parquet")
    log(f"{'='*60}")

    verification_failed = False
    sample_episodes = episodes[:min(10, len(episodes))]

    for ep_idx in sample_episodes:
        # Read parquet to get expected frame count
        parquet_path = PARQUET_ROOT / f"task-{task_idx:04d}" / f"episode_{ep_idx:08d}.parquet"
        try:
            table = pq.read_table(parquet_path)
            expected_frames = len(table)
        except Exception as e:
            log(f"  ✗ Cannot read parquet {parquet_path}: {e}")
            verification_failed = True
            continue

        for i, video_key in enumerate(VIDEO_KEYS):
            video_key_short = VIDEO_KEY_SHORT[i]
            video_path = DOWNLOAD_ROOT / "videos" / f"task-{task_idx:04d}" / video_key / f"episode_{ep_idx:08d}.mp4"

            if not video_path.exists():
                log(f"  ✗ Missing: episode {ep_idx:08d} {video_key_short}")
                verification_failed = True
                continue

            size_mb = video_path.stat().st_size / 1024 / 1024

            # Get video frame count using ffprobe
            try:
                result = subprocess.run(
                    ["ffprobe", "-v", "error", "-select_streams", "v:0", "-count_packets",
                     "-show_entries", "stream=nb_read_packets,width,height", "-of", "json", str(video_path)],
                    capture_output=True,
                    text=True,
                    timeout=10,
                )
                if result.returncode == 0:
                    info = json.loads(result.stdout)
                    if "streams" in info and len(info["streams"]) > 0:
                        stream = info["streams"][0]
                        width = stream.get("width", "?")
                        height = stream.get("height", "?")
                        actual_frames = int(stream.get("nb_read_packets", 0))

                        if actual_frames == expected_frames:
                            log(f"  ✓ episode {ep_idx:08d} {video_key_short}: {size_mb:.1f}MB, {width}×{height}, {actual_frames} frames (matches parquet)")
                        else:
                            log(f"  ✗ episode {ep_idx:08d} {video_key_short}: frame mismatch! video={actual_frames}, parquet={expected_frames}")
                            verification_failed = True
                    else:
                        log(f"  ✗ episode {ep_idx:08d} {video_key_short}: ffprobe returned no stream info")
                        verification_failed = True
                else:
                    log(f"  ✗ episode {ep_idx:08d} {video_key_short}: ffprobe failed")
                    verification_failed = True
            except Exception as e:
                log(f"  ✗ episode {ep_idx:08d} {video_key_short}: ffprobe error: {e}")
                verification_failed = True

    if verification_failed:
        log(f"\n⚠ Verification FAILED for task {task_idx:04d}")
        return False
    else:
        log(f"\n✓ Verification PASSED for task {task_idx:04d}")
        return True

def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    DOWNLOAD_ROOT.mkdir(parents=True, exist_ok=True)

    log(f"Starting B1K-2025 media download")
    log(f"PID: {os.getpid()}")
    log(f"Parquet root: {PARQUET_ROOT}")
    log(f"Download root: {DOWNLOAD_ROOT}")
    log(f"Status file: {STATUS_FILE}")

    # Get available tasks
    tasks = get_available_tasks()
    log(f"Found {len(tasks)} tasks with parquet data: {tasks}")

    # Get episodes per task
    episodes_per_task = {}
    for task_idx in tasks:
        episodes = get_episodes_for_task(task_idx)
        episodes_per_task[task_idx] = episodes
        log(f"Task {task_idx:04d}: {len(episodes)} episodes")

    # Estimate size
    estimated_gb = estimate_download_size(tasks, episodes_per_task)

    # Check disk space
    stat = os.statvfs(DOWNLOAD_ROOT)
    available_gb = (stat.f_bavail * stat.f_frsize) / (1024**3)
    log(f"Available disk space: {available_gb:.1f} GB")

    if estimated_gb > available_gb * 0.8:
        log(f"WARNING: Estimated download size ({estimated_gb:.1f} GB) is close to available space ({available_gb:.1f} GB)")

    # Load or initialize status
    status = load_status()
    if status["started_at"] is None:
        status["started_at"] = time.time()

    # Download task by task
    first_task_done = False
    for task_idx in tasks:
        # Skip completed tasks
        if task_idx in status["completed_tasks"]:
            log(f"Task {task_idx:04d} already completed, skipping")
            if task_idx == tasks[0]:
                first_task_done = True
            continue

        episodes = episodes_per_task[task_idx]
        task_success = download_task_videos(task_idx, episodes, status)

        # Verify first task
        if task_idx == tasks[0] and task_success:
            verify_success = verify_first_task(task_idx, episodes)
            if not verify_success:
                log(f"ERROR: First task verification failed, stopping")
                sys.exit(1)
            first_task_done = True
            log(f"\nFirst task verified successfully, continuing with remaining tasks in background...")
            # Continue with other tasks in same process

    # Final summary
    log(f"\n{'='*60}")
    log(f"Download completed!")
    log(f"Total downloaded: {status['total_downloaded']}")
    log(f"Total failed: {status['total_failed']}")
    log(f"Completed tasks: {len(status['completed_tasks'])}/{len(tasks)}")
    log(f"{'='*60}")

    save_status(status)

if __name__ == "__main__":
    main()
