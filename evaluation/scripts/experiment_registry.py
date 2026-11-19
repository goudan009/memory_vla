#!/usr/bin/env python3
"""实验台账：扫描训练 run / 评估 run / 权重 / 数据集，产出 registry.json

用法:
    python3 experiment_registry.py                # 扫描并写 registry.json
    python3 experiment_registry.py --json         # 只打印 JSON
    python3 experiment_registry.py --out X.json   # 指定输出

设计目标：只读。不改动、不删除任何文件。
"""
from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import time
from datetime import datetime, timezone, timedelta
from pathlib import Path

ROOT = Path("/nvme3/projects/memory_vla")
DEFAULT_OUT = ROOT / "outputs" / "registry.json"

# 跨机盘点：4090 通过 ssh 调用各机 ~/inventory_scan.py --json（只读）
REMOTE_HOSTS = [
    ("A6000", "6000PRO-WYH", "~/inventory_scan.py"),
    ("A100", "A100-WYH", "~/inventory_scan.py"),
]
REMOTE_TIMEOUT = 240
TZ = timezone(timedelta(hours=8))

# 历史路径 -> 当前 canonical 路径
PATH_ALIASES = [
    ("/nvme3/data_wyh/outputs/memory_vla", str(ROOT)),
]


# ---------------------------------------------------------------- helpers
def canon(p: str | None) -> str:
    """规范化路径（修复历史别名）"""
    if not p:
        return ""
    p = str(p)
    for old, new in PATH_ALIASES:
        if p.startswith(old):
            p = new + p[len(old):]
    return p


def read_json(path: Path, default=None):
    try:
        with path.open() as f:
            return json.load(f)
    except Exception:
        return default


def read_jsonl_tail(path: Path, n: int = 200):
    """读 jsonl 尾部 n 行"""
    try:
        with path.open("rb") as f:
            f.seek(0, os.SEEK_END)
            size = f.tell()
            block = min(size, 200_000)
            f.seek(size - block)
            lines = f.read().decode("utf-8", "ignore").strip().split("\n")
        out = []
        for line in lines[-n:]:
            line = line.strip()
            if not line:
                continue
            try:
                out.append(json.loads(line))
            except Exception:
                pass
        return out
    except Exception:
        return []


def du_bytes(paths: list[Path]) -> dict:
    """批量 du -sb，返回 {path_str: bytes}"""
    res = {}
    paths = [p for p in paths if p.exists()]
    if not paths:
        return res
    for i in range(0, len(paths), 40):
        chunk = [str(p) for p in paths[i:i + 40]]
        try:
            out = subprocess.run(["du", "-sb", "--"] + chunk,
                                 capture_output=True, text=True, timeout=300).stdout
            for line in out.splitlines():
                parts = line.split("\t", 1)
                if len(parts) == 2:
                    res[parts[1]] = int(parts[0])
        except Exception:
            continue
    return res


def mtime_str(p: Path) -> str:
    try:
        return datetime.fromtimestamp(p.stat().st_mtime, TZ).strftime("%Y-%m-%d %H:%M")
    except Exception:
        return ""


def newest_mtime(*paths: Path) -> float:
    best = 0.0
    for p in paths:
        try:
            if p.is_file():
                best = max(best, p.stat().st_mtime)
            elif p.is_dir():
                for f in p.rglob("*"):
                    if f.is_file():
                        best = max(best, f.stat().st_mtime)
        except Exception:
            pass
    return best


def gb(n: int | float) -> float:
    return round((n or 0) / 1024 ** 3, 2)


def short_task(name: str, cfg: dict | None = None) -> str:
    """从 run 名/任务名提取 task 编号"""
    if cfg:
        if cfg.get("task") is not None:
            return str(cfg["task"])
    import re
    m = re.search(r"task[_\-]?(\d+)", name)
    return m.group(1) if m else ""


def human_kind(path: Path) -> str:
    """猜测 run 的组织版本，如 V_1 / agent"""
    parts = path.parts
    for p in parts:
        if p.startswith("V_") and "_" in p:
            return p
    if "agent_runs" in parts:
        return "agent"
    return ""


# ---------------------------------------------------------------- 训练 run
def scan_training() -> list[dict]:
    items = []

    # 1) outputs/<ver_dir>/training/<run>/  （自研训练框架）
    for train_dir in sorted(ROOT.glob("outputs/*/training")):
        ver_dir = train_dir.parent.name
        for run in sorted(train_dir.iterdir()):
            if not run.is_dir():
                continue
            cfg = read_json(run / "config" / "run_config.json", {}) or {}
            arms = []
            eval_st = read_json(run / "evaluation" / "status.json", {}) or {}
            # 子目录 = 训练臂（memory_v1 / baseline / ...）；evaluation 是闭环验证环节，不算臂
            for sub in sorted(run.iterdir()):
                if not sub.is_dir() or sub.name in ("config", "logs", "metrics", "plots",
                                                    "evaluation", "validation", "diagnostics"):
                    continue
                st = read_json(sub / "status.json", {}) or {}
                if not st and not (sub / "checkpoints").exists():
                    continue
                arms.append((sub, st))
            if not arms:
                arms = [(run, {})]

            for sub, st in arms:
                # 最新 metrics
                mfile = sub / "metrics" / "metrics.jsonl"
                tail = read_jsonl_tail(mfile, 3)
                last = tail[-1] if tail else {}
                ckpts = sorted((sub / "checkpoints").glob("step-*")) if (sub / "checkpoints").is_dir() else []
                best = st.get("best") or {}
                status = st.get("status") or ("unknown")
                step = st.get("step") or last.get("step") or 0
                total = st.get("total_steps") or cfg.get("steps") or 0
                item = {
                    "id": f"train:{ver_dir}/{run.name}/{sub.name}" if sub is not run else f"train:{ver_dir}/{run.name}",
                    "kind": "train",
                    "version": human_kind(run),
                    "name": run.name + (f"/{sub.name}" if sub is not run else ""),
                    "arm": "memory" if "mem" in (sub.name + run.name).lower() else "baseline",
                    "task": short_task(run.name, cfg),
                    "task_name": cfg.get("task_name", ""),
                    "status": status,
                    "step": step,
                    "total_steps": total,
                    "metrics": {
                        k: v for k, v in {
                            "loss": last.get("loss") or last.get("train_loss"),
                            "lr": last.get("lr") or last.get("learning_rate"),
                            "val_loss": best.get("validation_loss") or st.get("validation_loss"),
                        }.items() if v is not None
                    },
                    "best_step": best.get("step"),
                    "dataset": canon(cfg.get("dataset", "")),
                    "pretrained": canon(cfg.get("pretrained_checkpoint", "")),
                    "gpus": cfg.get("gpus") or ([cfg["world_size"]] if cfg.get("world_size") else []),
                    "memory": bool(cfg.get("memory_enabled") or "mem" in (sub.name + run.name).lower()),
                    "checkpoints": [canon(str(c)) for c in ckpts],
                    "n_checkpoints": len(ckpts),
                    "path": canon(str(sub if sub is not run else run)),
                    "updated": mtime_str(sub),
                    "source": "outputs",
                    "eval_status": eval_st.get("status") or "",
                    "eval_error": (eval_st.get("error") or "")[:160],
                }
                item["size_gb"] = 0.0  # 后面统一算
                items.append(item)

    # 2) outputs/official_baseline_*  （官方脚本产出）
    for run in sorted(ROOT.glob("outputs/official_baseline_*")):
        if not run.is_dir():
            continue
        cfg = read_json(run / "config.json", {}) or {}
        log = run / "train.log"
        last_step = 0
        try:
            with log.open("rb") as f:
                f.seek(0, os.SEEK_END)
                size = f.tell()
                f.seek(max(0, size - 200_000))
                txt = f.read().decode("utf-8", "ignore")
            import re
            steps = re.findall(r"step[^\d]{0,4}(\d+)", txt)
            if steps:
                last_step = int(steps[-1])
        except Exception:
            pass
        ckpts = sorted(run.glob("checkpoint-*"))
        weights = sorted(run.glob("model-*.safetensors"))
        items.append({
            "id": f"train:official/{run.name}",
            "kind": "train",
            "version": "official",
            "name": run.name,
            "arm": "official",
            "task": short_task(run.name),
            "task_name": "",
            "status": "completed" if (run / "model.safetensors.index.json").exists() else "unknown",
            "step": last_step,
            "total_steps": 1000,
            "metrics": {},
            "best_step": None,
            "dataset": "",
            "pretrained": "",
            "gpus": [],
            "memory": False,
            "checkpoints": [canon(str(c)) for c in ckpts],
            "n_checkpoints": len(ckpts),
            "weights": [canon(str(w)) for w in weights],
            "path": canon(str(run)),
            "updated": mtime_str(run),
            "source": "official_script",
            "size_gb": 0.0,
        })

    # 3) agent_runs/train/<arm>/
    for run in sorted((ROOT / "agent_runs" / "train").glob("*")):
        if not run.is_dir():
            continue
        st = read_json(run / "status.json", {}) or {}
        subs = [d for d in run.iterdir() if d.is_dir() and (d / "status.json").exists()]
        if subs:
            for sub in subs:
                st = read_json(sub / "status.json", st) or st
                items.append(_agent_train_item(run, sub, st))
        else:
            items.append(_agent_train_item(run, run, st))

    return items


def _agent_train_item(run: Path, sub: Path, st: dict) -> dict:
    ckpts = sorted((sub / "checkpoints").glob("step-*")) if (sub / "checkpoints").is_dir() else []
    return {
        "id": f"train:agent/{run.name}/{sub.name}" if sub is not run else f"train:agent/{run.name}",
        "kind": "train",
        "version": "agent",
        "name": run.name + (f"/{sub.name}" if sub is not run else ""),
        "arm": "memory" if "mem" in (run.name + sub.name).lower() else "baseline",
        "task": short_task(run.name + sub.name),
        "task_name": "",
        "status": st.get("status") or "unknown",
        "step": st.get("step") or 0,
        "total_steps": st.get("total_steps") or 0,
        "metrics": {},
        "best_step": None,
        "dataset": "",
        "pretrained": "",
        "gpus": [],
        "memory": "mem" in (run.name + sub.name).lower(),
        "checkpoints": [canon(str(c)) for c in ckpts],
        "n_checkpoints": len(ckpts),
        "error": (st.get("error") or "")[:200],
        "path": canon(str(sub)),
        "updated": mtime_str(sub),
        "source": "agent_runs",
        "size_gb": 0.0,
    }


# ---------------------------------------------------------------- 评估 run
def scan_evals() -> list[dict]:
    items = []
    seen = set()
    for ver_dir in sorted(ROOT.glob("evaluation/V_*")):
        if not ver_dir.is_dir():
            continue
        for rj in sorted(ver_dir.rglob("results.json")):
            rel = rj.relative_to(ver_dir)
            if len(rel.parts) > 4:
                continue
            key = str(rj)
            if key in seen:
                continue
            seen.add(key)
            d = read_json(rj)
            if not isinstance(d, dict):
                d = {}
            run_dir = rj.parent
            group = str(rel.parts[0]) if len(rel.parts) > 1 else ""
            # 补漏：老结果没有 instruction 时代码会退化，这里记一下视频数
            videos = []
            try:
                videos = [str(p.relative_to(run_dir)) for p in run_dir.rglob("*.mp4")]
            except Exception:
                pass
            diag_dir = run_dir / "vlm_diagnosis"
            n_diag = len(list(diag_dir.glob("*.json"))) if diag_dir.is_dir() else 0
            items.append({
                "id": f"eval:{ver_dir.name}/{rel.parent}",
                "kind": "eval",
                "version": ver_dir.name,
                "group": group,
                "name": str(rel.parent),
                "method": d.get("method") or d.get("model_name") or "",
                "model_name": d.get("model_name") or "",
                "task": str(d.get("task_id")) if d.get("task_id") is not None else short_task(str(rel.parent)),
                "task_name": d.get("task_name") or "",
                "status": d.get("status") or "unknown",
                "ok": d.get("ok"),
                "episodes": d.get("requested_episodes") or d.get("recorded_episodes"),
                "valid_episodes": d.get("valid_episodes"),
                "success_rate": d.get("success_rate"),
                "mean_progress": d.get("mean_task_progress"),
                "mean_q_score": d.get("mean_q_score"),
                "elapsed_s": d.get("elapsed_wall_s"),
                "started_at": d.get("started_at") or "",
                "finished_at": d.get("finished_at") or "",
                "policy": (d.get("policy_identity") or {}) if isinstance(d.get("policy_identity"), dict) else {},
                "n_videos": len(videos),
                "n_vlm_diag": n_diag,
                "path": canon(str(run_dir)),
                "results": canon(str(rj)),
                "updated": mtime_str(run_dir),
                "size_gb": 0.0,
            })
    return items


# ---------------------------------------------------------------- 权重
def scan_weights() -> list[dict]:
    items = []
    # checkpoints/
    for w in sorted((ROOT / "checkpoints").iterdir()) if (ROOT / "checkpoints").is_dir() else []:
        if not w.is_dir():
            continue
        files = sorted(list(w.glob("model-*.safetensors")) + list(w.glob("params/*")) + list(w.glob("*.safetensors")))
        items.append({
            "id": f"weights:checkpoints/{w.name}",
            "kind": "weights",
            "version": "shared",
            "name": w.name,
            "status": "ready" if files else "unknown",
            "files": [canon(str(f)) for f in files][:6],
            "n_files": len(files),
            "path": canon(str(w)),
            "updated": mtime_str(w),
            "size_gb": 0.0,
        })
    return items


# ---------------------------------------------------------------- 数据集
def scan_datasets() -> list[dict]:
    items = []
    ds_root = ROOT / "assets" / "datasets"
    if not ds_root.is_dir():
        return items
    for d in sorted(ds_root.iterdir()):
        if not d.is_dir():
            continue
        n_parquet = sum(1 for _ in d.rglob("*.parquet"))
        items.append({
            "id": f"dataset:{d.name}",
            "kind": "dataset",
            "version": "data",
            "name": d.name,
            "task": short_task(d.name),
            "n_parquet": n_parquet,
            "path": canon(str(d)),
            "updated": mtime_str(d),
            "size_gb": 0.0,
        })
    return items


# ---------------------------------------------------------------- 其它大块产物
def scan_other(items: list[dict]) -> list[dict]:
    """outputs/ 与 agent_runs/ 下未被具体条目覆盖的大目录（benchmarks、diagnostics 等）"""
    covered = [i.get("path") or "" for i in items]
    out = []
    for root, label in [(ROOT / "outputs", "outputs"), (ROOT / "agent_runs", "agent_runs")]:
        if not root.is_dir():
            continue
        for d in sorted(root.iterdir()):
            if not d.is_dir() or d.name in ("registry.json",):
                continue
            sp = str(d)
            if any(c.startswith(sp) for c in covered):
                continue
            if not any(d.iterdir()) if d.is_dir() else True:
                continue
            out.append({
                "id": f"other:{label}/{d.name}",
                "kind": "other",
                "version": label,
                "name": d.name,
                "status": "",
                "path": sp,
                "updated": mtime_str(d),
                "size_gb": 0.0,
                "flags": ["未归类"],
            })
    return out


# ---------------------------------------------------------------- 组装
def scan_remote() -> dict:
    """ssh 到各机跑 inventory_scan.py --json，返回 {host: data}（只读，失败标 alive=False）"""
    out = {}
    for label, ssh_alias, remote_py in REMOTE_HOSTS:
        t0 = time.time()
        try:
            proc = subprocess.run(
                ["ssh", "-o", "BatchMode=yes", "-o", "ConnectTimeout=12", ssh_alias,
                 f"INV_HOST_LABEL={label} python3 {remote_py} --json"],
                capture_output=True, text=True, timeout=REMOTE_TIMEOUT)
            raw = proc.stdout.strip()
            if not raw:
                raise RuntimeError((proc.stderr or "empty output").strip()[:200])
            data = json.loads(raw.splitlines()[-1])
            data["alive"] = True
        except Exception as error:
            data = {"host": label, "alive": False, "error": str(error)[:200], "dirs": [], "repos": []}
        data["query_s"] = round(time.time() - t0, 1)
        out[label] = data
    return out


def remote_items(scans: dict) -> list[dict]:
    """把远端目录变成台账条目（kind 归一到 dataset/weights/other）"""
    items = []
    for label, scan in scans.items():
        if not scan.get("alive"):
            continue
        for d in scan.get("dirs") or []:
            kind = d.get("kind")
            if kind == "experiment":
                kind = "other"          # 远端训练目录先归到「其他」，不冒充本地实验链
            items.append({
                "id": f"{label}:{d['path']}",
                "kind": kind or "other",
                "host": label,
                "name": d["name"],
                "path": f"{label}:{d['path']}",
                "fs_path": d["path"],
                "size_gb": round(float(d.get("size_gb") or 0), 2),
                "updated": d.get("mtime") or "",
                "status": "remote",
                "flags": [],
            })
    return items


def local_infra() -> dict:
    """本机 GPU/磁盘/端口/代码仓库（复用 inventory_scan 的只读函数）"""
    try:
        import inventory_scan as inv
    except Exception:
        return {"gpu": "", "disk": [], "ports": [], "repos": []}
    try:
        return {
            "gpu": inv.scan_gpu(),
            "disk": inv.scan_disk(),
            "ports": inv.scan_ports(),
            "repos": inv.scan_repos(),
        }
    except Exception:
        return {"gpu": "", "disk": [], "ports": [], "repos": []}


_ARM_ALIAS = {"bas": "baseline", "base": "baseline", "baseline": "baseline",
              "mem": "memory", "memory": "memory",
              "bas_ex1": "baseline_ex1", "mem_ex1": "memory_ex1"}


def _train_key(item: dict):
    """V_1_mem_task21/memory_v1 -> ('V_1','memory',21)"""
    run = (item.get("name") or "").split("/")[0]
    m = re.match(r"(V_\d+)_([A-Za-z0-9]+?)_task(\d+)$", run)
    if not m:
        return None
    ver, arm_raw, task = m.group(1), m.group(2).lower(), int(m.group(3))
    return ver, _ARM_ALIAS.get(arm_raw, arm_raw), task


def build_experiments(items: list[dict]) -> list[dict]:
    """实验版本管理：把训练 run 与评估 run 按 (版本, 实验臂, 任务) 串成一条链"""
    trains = {}
    for it in items:
        if it["kind"] != "train":
            continue
        key = _train_key(it)
        if key:
            trains[key] = it
    evals = {}
    for it in items:
        if it["kind"] != "eval":
            continue
        ver = it.get("version") or ""
        arm = _ARM_ALIAS.get((it.get("group") or "").lower(), (it.get("group") or "").lower())
        task_raw = str(it.get("task") or "")
        if not (ver and arm and task_raw.isdigit()):
            continue
        evals.setdefault((ver, arm, int(task_raw)), []).append(it)

    chains = []
    for key in set(trains) | set(evals):
        ver, arm, task = key
        tr = trains.get(key)
        evs = sorted(evals.get(key, []), key=lambda e: e.get("name") or "")
        ck_known = [e.get("checkpoint_exists") for e in evs]
        ck_lost = any(v is False for v in ck_known)
        ck_unrecorded = bool(evs) and all(v is None for v in ck_known)
        problems = []
        if tr is None:
            problems.append("训练记录缺失")
        elif tr.get("n_checkpoints", 0) == 0:
            problems.append("训练权重不在盘")
        if not evs:
            problems.append("未评估")
        if ck_lost:
            problems.append("评估权重已丢失")
        elif ck_unrecorded:
            problems.append("未记录权重来源")
        chains.append({
            "version": ver, "arm": arm, "task": task,
            "task_name": (tr or {}).get("task_name") or (evs[0].get("task_name") if evs else ""),
            "train": None if tr is None else {
                "name": tr["name"], "status": tr.get("status"),
                "step": tr.get("step"), "total_steps": tr.get("total_steps"),
                "best_step": tr.get("best_step"), "metrics": tr.get("metrics") or {},
                "n_checkpoints": tr.get("n_checkpoints", 0),
                "size_gb": tr.get("size_gb"), "path": tr.get("path"),
                "updated": tr.get("updated"),
            },
            "evals": [{
                "name": e["name"], "group": e.get("group"),
                "success_rate": e.get("success_rate"), "mean_progress": e.get("mean_progress"),
                "episodes": e.get("episodes"), "n_vlm_diag": e.get("n_vlm_diag", 0),
                "checkpoint_exists": e.get("checkpoint_exists"), "size_gb": e.get("size_gb"),
                "updated": e.get("updated"), "path": e.get("path"),
            } for e in evs],
            "problems": problems,
        })
    chains.sort(key=lambda c: (c["version"], c["arm"], c["task"]))
    return chains


def build() -> dict:
    t0 = time.time()
    items = scan_training() + scan_evals() + scan_weights() + scan_datasets()
    items += scan_other(items)

    # 大小（批量 du，只算顶层路径，避免重复遍历）
    for i in items:
        i.setdefault("host", "4090")
    dirs = [Path(i["path"]) for i in items
            if i.get("host") == "4090" and i.get("path") and Path(i["path"]).is_dir()]
    sizes = du_bytes(dirs)
    for i in items:
        if i.get("host") != "4090":
            continue
        p = i.get("path")
        i["size_gb"] = gb(sizes.get(p, 0)) if p else 0.0

    # 血缘：评估 run -> 训练 run（policy_identity.training_id 精确匹配）
    train_items = [i for i in items if i["kind"] == "train"]
    by_run_name = {}
    for tr in train_items:
        rn = tr["name"].split("/")[0]
        if rn:
            by_run_name.setdefault(rn, tr)
    for ev in (i for i in items if i["kind"] == "eval"):
        pol = ev.get("policy") or {}
        tid = pol.get("training_id") or ""
        tr = by_run_name.get(tid)
        ev["training_id"] = tid
        ev["checkpoint"] = canon(pol.get("checkpoint") or "")
        ev["checkpoint_step"] = pol.get("checkpoint_step") or ""
        ev["linked_train"] = tr["id"] if tr else ""
        # 评估用的权重是否还在盘上
        ck = ev.get("checkpoint")
        if ck:
            ev["checkpoint_exists"] = Path(ck).exists()
        else:
            ev["checkpoint_exists"] = None

    # flags
    for i in items:
        flags = []
        if i["kind"] == "train":
            if i.get("status") == "failed":
                flags.append("失败")
            if i.get("status") == "training":
                flags.append("进行中")
            if i.get("n_checkpoints", 0) >= 2 and i.get("source") == "official_script":
                flags.append("中间权重可删")
            evs = [e for e in items if e["kind"] == "eval" and e.get("linked_train") == i["id"]]
            if not evs and i.get("status") == "completed":
                flags.append("未评估")
            elif evs:
                flags.append(f"已评估×{len(evs)}")
        if i["kind"] == "eval":
            if not i.get("linked_train"):
                flags.append("训练产物已删" if i.get("training_id") else "无血缘信息")
            if i.get("checkpoint_exists") is False:
                flags.append("权重已丢失")
            elif i.get("checkpoint_exists") is True:
                flags.append("权重在盘")
        if i["kind"] == "train" and i.get("eval_status") == "failed":
            flags.append("闭环验证失败")
        if i["kind"] == "eval" and i.get("n_vlm_diag", 0) == 0:
            flags.append("无VLM诊断")
        elif i["kind"] == "eval":
            flags.append("有VLM诊断")
        i["flags"] = flags

    # ---- 跨机盘点（A6000 / A100）----
    scans = scan_remote()
    items += remote_items(scans)
    hosts = []
    for label, scan in scans.items():
        hosts.append({
            "host": label,
            "hostname": scan.get("hostname") or label,
            "alive": bool(scan.get("alive")),
            "error": scan.get("error", ""),
            "gpu": scan.get("gpu", ""),
            "disk": scan.get("disk") or [],
            "ports": scan.get("ports") or [],
            "scanned_at": scan.get("scanned_at", ""),
            "query_s": scan.get("query_s"),
        })
    infra_local = local_infra()
    hosts.insert(0, {
        "host": "4090", "hostname": "server-4090", "alive": True, "error": "",
        "gpu": infra_local.get("gpu", ""), "disk": infra_local.get("disk") or [],
        "ports": infra_local.get("ports") or [],
        "scanned_at": datetime.now(TZ).strftime("%Y-%m-%d %H:%M:%S"), "query_s": 0,
    })

    repos = []
    for r in infra_local.get("repos") or []:
        repos.append({**r, "host": "4090"})
    for label, scan in scans.items():
        for r in scan.get("repos") or []:
            repos.append({**r, "host": label})

    total = sum(i["size_gb"] for i in items)
    ck_paths = [Path(p) for i in items if i.get("source") == "official_script"
                for p in (i.get("checkpoints") or [])]
    ck_sizes = du_bytes(ck_paths)
    reclaimable = sum(gb(v) for v in ck_sizes.values())
    train_evals = [i for i in items if i["kind"] == "eval" and i.get("linked_train")]
    lost = [i for i in items if i["kind"] == "eval" and i.get("checkpoint_exists") is False]
    summary = {
        "train_runs": sum(1 for i in items if i["kind"] == "train"),
        "eval_runs": sum(1 for i in items if i["kind"] == "eval"),
        "weights": sum(1 for i in items if i["kind"] == "weights"),
        "datasets": sum(1 for i in items if i["kind"] == "dataset"),
        "other": sum(1 for i in items if i["kind"] == "other"),
        "total_gb": round(total, 1),
        "reclaimable_gb": round(reclaimable, 1),
        "eval_linked": len(train_evals),
        "eval_orphan": sum(1 for i in items if i["kind"] == "eval") - len(train_evals),
        "eval_ckpt_lost": len(lost),
        "remote_gb": round(sum(i["size_gb"] for i in items if i.get("host") != "4090"), 1),
        "remote_hosts_alive": sum(1 for h in hosts if h["alive"] and h["host"] != "4090"),
        "repos_dirty": sum(1 for r in repos if (r.get("dirty") or 0) > 0),
        "repos_no_remote": sum(1 for r in repos if not r.get("remote")),
        "scan_s": round(time.time() - t0, 1),
    }
    return {
        "generated_at": datetime.now(TZ).strftime("%Y-%m-%d %H:%M:%S"),
        "root": str(ROOT),
        "summary": summary,
        "hosts": hosts,
        "repos": repos,
        "experiments": build_experiments(items),
        "items": items,
    }


_CACHE = {"t": 0.0, "data": None}


def cached(force: bool = False, ttl: float = 0.0) -> dict:
    """读盘优先：默认直接返回 outputs/registry.json（秒开、不重扫）；
    force=True 才跑一次完整扫描并写盘；ttl>0 时文件超过 ttl 秒会自动重扫。"""
    now = time.time()
    if not force:
        if _CACHE["data"] is not None and (ttl <= 0 or now - _CACHE["t"] < ttl):
            return _CACHE["data"]
        try:
            data = json.loads(DEFAULT_OUT.read_text())
            if isinstance(data, dict) and "items" in data:
                if ttl <= 0 or now - DEFAULT_OUT.stat().st_mtime < ttl:
                    _CACHE.update({"t": now, "data": data})
                    return data
        except (OSError, ValueError):
            pass
    reg = build()
    _CACHE.update({"t": now, "data": reg})
    try:
        DEFAULT_OUT.write_text(json.dumps(reg, ensure_ascii=False, indent=1))
    except OSError:
        pass
    return reg


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--json", action="store_true", help="打印 JSON")
    ap.add_argument("--out", default=str(DEFAULT_OUT))
    ap.add_argument("--brief", action="store_true", help="打印摘要表")
    a = ap.parse_args()
    reg = build()
    Path(a.out).write_text(json.dumps(reg, ensure_ascii=False, indent=1))
    print(f"[registry] -> {a.out}  {reg['summary']}")
    if a.brief or a.json:
        for i in sorted(reg["items"], key=lambda x: -x["size_gb"]):
            print(f"{i['size_gb']:>8.2f}G  {i['kind']:<8} {str(i.get('status','')):<10} "
                  f"{i['id']:<55} {'/'.join(i.get('flags') or [])}")
    if a.json:
        print(json.dumps(reg, ensure_ascii=False, indent=1))


if __name__ == "__main__":
    main()
