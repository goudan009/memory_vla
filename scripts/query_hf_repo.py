#!/usr/bin/env python3
"""Query HuggingFace repository structure without proxy"""
import os
import sys

# Remove proxy settings
for key in ['http_proxy', 'https_proxy', 'HTTP_PROXY', 'HTTPS_PROXY', 'all_proxy', 'ALL_PROXY']:
    os.environ.pop(key, None)

from huggingface_hub import HfApi

api = HfApi()

try:
    files = api.list_repo_files("behavior-1k/2025-challenge-demos", repo_type="dataset")
    print(f"Total files: {len(files)}")

    # Group by type
    video_files = [f for f in files if f.endswith('.mp4')]
    parquet_files = [f for f in files if f.endswith('.parquet')]
    json_files = [f for f in files if f.endswith('.json') or f.endswith('.jsonl')]

    print(f"\nVideo files: {len(video_files)}")
    print(f"Parquet files: {len(parquet_files)}")
    print(f"JSON/JSONL files: {len(json_files)}")

    # Check structure - look for task directories
    task_dirs = set()
    for f in files:
        if '/' in f:
            first_part = f.split('/')[0]
            if first_part.startswith('task-'):
                task_dirs.add(first_part)

    print(f"\nTask directories: {len(task_dirs)}")
    print(f"Task indices: {sorted([int(d.split('-')[1]) for d in task_dirs])}")

    # Sample video structure
    print("\nSample video files:")
    for f in sorted(video_files)[:30]:
        print(f"  {f}")

except Exception as e:
    print(f"Error: {e}", file=sys.stderr)
    import traceback
    traceback.print_exc()
    sys.exit(1)
