#!/usr/bin/env python3
"""Read-only structural audit for the native B1K-2026 LeRobot v3 subset.

This command only checks the downloaded Task0/Task40 data and annotations.  It
does not rewrite metadata, calculate a training checkpoint, or submit a job.
Run it with the pinned N1.7 environment because that environment supplies the
Parquet and video readers used by the native loader.
"""
import argparse
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from v1_2_groot_baseline.b1k2026_data import audit_2026, probe_native_loader


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("dataset", type=Path)
    parser.add_argument("--native-view", type=Path)
    parser.add_argument("--upstream", type=Path)
    args = parser.parse_args()
    if bool(args.native_view) != bool(args.upstream):
        parser.error("--native-view and --upstream must be supplied together")
    try:
        result = audit_2026(args.dataset)
        if args.native_view:
            result['native_loader'] = probe_native_loader(args.native_view, args.upstream)
    except Exception as error:
        print(json.dumps({"status": "FAIL", "error_type": type(error).__name__,
                          "error": str(error)}, ensure_ascii=False, indent=2))
        return 1
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
