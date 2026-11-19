#!/usr/bin/env python3
"""通用机器盘点：只读，输出 JSON（供 4090 实验台账跨机汇总）

用法:
    python3 inventory_scan.py                # 人类可读
    python3 inventory_scan.py --json         # JSON（远端调用用）
    python3 inventory_scan.py --json --max 12

设计原则:
  * 只读，绝不修改/删除任何文件
  * 一律限制深度与条目数，避免在多用户大盘上跑成 IO 风暴
  * 不做机器专属假设：home 一层 + /nvme3 一层 + /data 一层，有就扫、没有跳过
"""
from __future__ import annotations

import argparse
import json
import os
import re
import shutil
import socket
import subprocess
import time
from pathlib import Path

HOME = Path.home()
CANDIDATE_ROOTS = [HOME, Path("/nvme3"), Path("/data"), Path("/mnt/data")]
SKIP_NAMES = {".cache", ".local", ".config", ".nvm", ".codex", ".conda", "snap", "miniforge3",
              "miniconda3", "anaconda3", "lost+found", "backup", "cuda_12.8.1_570.124.06_linux.run"}

DATASET_HINT = re.compile(r"(data|dataset|b1k|behavior|libero|parquet|demonstr|episode|lerobot_runs)", re.I)
WEIGHT_HINT = re.compile(r"(model|hf_|huggingface|ckpt|checkpoint|weights|safetensors|train_model)", re.I)
EXP_HINT = re.compile(r"(train|output|run|eval|exp|benchmark|log)", re.I)


def run(cmd: list[str], timeout: float = 60.0) -> str:
    try:
        p = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)
        return (p.stdout or "").strip()
    except Exception:
        return ""


def du_gb(path: Path, timeout: float = 90.0) -> float:
    out = run(["du", "-sb", "-d0", str(path)], timeout=timeout)
    m = out.split()
    if m and m[0].isdigit():
        return int(m[0]) / 1024 ** 3
    return 0.0


def classify(name: str, path: Path) -> str:
    n = str(path)
    if WEIGHT_HINT.search(name) or "/huggingface" in n or "/models" in n:
        return "weights"
    if DATASET_HINT.search(name):
        return "dataset"
    if EXP_HINT.search(name):
        return "experiment"
    return "other"


def scan_dirs(max_items: int) -> list[dict]:
    """home / nvme3 / data 的一层子目录（深度 1，限制条数）"""
    out = []
    for root in CANDIDATE_ROOTS:
        if not root.is_dir():
            continue
        try:
            children = [c for c in root.iterdir() if c.is_dir() and not c.name.startswith(".")]
        except OSError:
            continue
        for c in children:
            if c.name in SKIP_NAMES:
                continue
            gb = du_gb(c)
            if gb < 0.05:
                continue
            out.append({
                "name": c.name,
                "path": str(c),
                "root": str(root),
                "size_gb": round(gb, 2),
                "kind": classify(c.name, c),
                "mtime": time.strftime("%Y-%m-%d", time.localtime(c.stat().st_mtime)),
            })
    out.sort(key=lambda x: -x["size_gb"])
    return out[:max_items]


def scan_repos(max_depth: int = 4, max_items: int = 20) -> list[dict]:
    repos = []
    roots = [str(HOME), "/nvme3/projects"]
    for root in roots:
        if not Path(root).is_dir():
            continue
        found = run(["find", root, "-maxdepth", str(max_depth), "-name", ".git", "-type", "d"], timeout=90)
        for line in found.splitlines():
            repo = line[:-5] if line.endswith("/.git") else line
            if any(s in repo for s in ("/.codex/", "/.nvm", "plugins.disabled")):
                continue
            if repo in repos:
                continue
            repos.append(repo)
            if len(repos) >= max_items:
                break
    out = []
    for repo in repos:
        branch = run(["git", "-C", repo, "rev-parse", "--abbrev-ref", "HEAD"], timeout=20) or "-"
        dirty = len(run(["git", "-C", repo, "status", "--porcelain"], timeout=40).splitlines())
        remote = ""
        rem = run(["git", "-C", repo, "remote", "-v"], timeout=20).splitlines()
        if rem:
            parts = rem[0].split()
            remote = parts[1] if len(parts) > 1 else ""
        last = run(["git", "-C", repo, "log", "-1", "--format=%cd|%h", "--date=short"], timeout=20)
        cdate, chash = (last.split("|") + ["", ""])[:2] if "|" in last else ("", "")
        out.append({
            "path": repo, "branch": branch, "dirty": dirty, "remote": remote,
            "last_commit": cdate, "hash": chash,
        })
    return out


def scan_gpu() -> str:
    return run(["nvidia-smi", "--query-gpu=index,memory.used,memory.total,utilization.gpu",
                "--format=csv,noheader,nounits"], timeout=20)


def scan_disk() -> list[dict]:
    out = []
    for line in run(["df", "-B1", "-x", "tmpfs", "-x", "devtmpfs"], timeout=20).splitlines()[1:]:
        f = line.split()
        if len(f) < 6:
            continue
        if not (f[5].startswith("/nvme") or f[5] == "/" or f[5].startswith("/home")
                or f[5].startswith("/mnt")):
            continue
        total, used, avail = int(f[1]), int(f[2]), int(f[3])
        out.append({"mount": f[5], "total_gb": round(total / 1024**3, 1),
                    "used_gb": round(used / 1024**3, 1), "avail_gb": round(avail / 1024**3, 1),
                    "pct": int(f[4].rstrip("%"))})
    return out


def scan_ports() -> list[dict]:
    wanted = {8765: "实验台账/结果工作台", 8901: "VLM 诊断服务", 8000: "vLLM/API",
              8080: "web", 3000: "web dev", 8888: "jupyter"}
    listening = run(["ss", "-ltnH"], timeout=20)
    ports = set()
    for line in listening.splitlines():
        m = re.search(r":(\d+)\s", line)
        if m:
            ports.add(int(m.group(1)))
    return [{"port": p, "name": n, "up": p in ports} for p, n in wanted.items() if p in ports]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--json", action="store_true")
    ap.add_argument("--max", type=int, default=16, help="目录条目上限")
    args = ap.parse_args()

    t0 = time.time()
    data = {
        "host": os.environ.get("INV_HOST_LABEL") or socket.gethostname(),
        "hostname": socket.gethostname(),
        "scanned_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "gpu": scan_gpu(),
        "disk": scan_disk(),
        "dirs": scan_dirs(args.max),
        "repos": scan_repos(),
        "ports": scan_ports(),
        "scan_s": round(time.time() - t0, 1),
    }
    if args.json:
        print(json.dumps(data, ensure_ascii=False))
        return
    print(f"== {data['host']} ({data['hostname']}) 扫描 {data['scan_s']}s")
    print("GPU:", data["gpu"].replace("\n", " | "))
    for d in data["disk"]:
        print(f"  盘 {d['mount']:<22} {d['used_gb']:>7.1f}/{d['total_gb']:>7.1f}G  {d['pct']}%")
    print("目录:")
    for d in data["dirs"][:12]:
        print(f"  {d['size_gb']:>8.1f}G [{d['kind']:<10}] {d['path']}")
    print("代码仓库:")
    for r in data["repos"]:
        print(f"  {r['path']:<60} {r['branch']:<16} 改动{r['dirty']:<4} {r['last_commit']} {r['remote'][:40]}")
    print("端口:", ", ".join(f"{p['port']}({p['name']})" for p in data["ports"]) or "无")


if __name__ == "__main__":
    main()
