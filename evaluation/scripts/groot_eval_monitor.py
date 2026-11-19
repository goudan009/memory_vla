"""Sample process, host, disk, and GPU resources during evaluation diagnostics."""

import argparse
import csv
import os
from pathlib import Path
import subprocess
import time


def proc_sample(pid):
    try:
        stat = Path(f"/proc/{pid}/stat").read_text()
        fields = stat[stat.rfind(")") + 2 :].split()
        parent_pid = int(fields[1])
        cpu_ticks = int(fields[11]) + int(fields[12])
        status = Path(f"/proc/{pid}/status").read_text()
        rss_kib = next(int(line.split()[1]) for line in status.splitlines() if line.startswith("VmRSS:"))
        io = {}
        for line in Path(f"/proc/{pid}/io").read_text().splitlines():
            key, value = line.split(":", 1)
            io[key] = int(value)
        return parent_pid, cpu_ticks, rss_kib * 1024, io.get("read_bytes", 0), io.get("write_bytes", 0)
    except (FileNotFoundError, StopIteration, ValueError, PermissionError):
        return None


def process_tree_sample(root_pid):
    processes = {}
    try:
        pids = [int(entry.name) for entry in os.scandir("/proc") if entry.name.isdigit()]
    except OSError:
        return None
    for pid in pids:
        sample = proc_sample(pid)
        if sample is not None:
            processes[pid] = sample
    if root_pid not in processes:
        return None
    members = {root_pid}
    changed = True
    while changed:
        changed = False
        for pid, sample in processes.items():
            if sample[0] in members and pid not in members:
                members.add(pid)
                changed = True
    totals = [sum(processes[pid][index] for pid in members) for index in range(1, 5)]
    return tuple(totals)


def host_sample():
    mem_available_kib = next(
        int(line.split()[1]) for line in Path("/proc/meminfo").read_text().splitlines()
        if line.startswith("MemAvailable:")
    )
    read_sectors = 0
    write_sectors = 0
    for line in Path("/proc/diskstats").read_text().splitlines():
        fields = line.split()
        name = fields[2]
        if name.startswith("nvme") and "p" not in name or name.startswith("sd") and name[-1].isdigit() is False:
            read_sectors += int(fields[5])
            write_sectors += int(fields[9])
    return mem_available_kib * 1024, read_sectors * 512, write_sectors * 512


def gpu_sample(gpu_index):
    try:
        result = subprocess.run(
            ["nvidia-smi", "--id", str(gpu_index),
             "--query-gpu=utilization.gpu,memory.used", "--format=csv,noheader,nounits"],
            check=True, capture_output=True, text=True, timeout=4,
        )
        util, memory = (int(value.strip()) for value in result.stdout.strip().split(","))
        return util, memory * 1024 * 1024
    except (OSError, subprocess.SubprocessError, ValueError):
        return None, None


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--pid", type=int, required=True)
    parser.add_argument("--gpu", type=int, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--interval", type=float, default=5.0)
    args = parser.parse_args()

    args.output.parent.mkdir(parents=True, exist_ok=True)
    fields = [
        "elapsed_s", "cpu_pct", "rss_mib", "process_read_mib", "process_write_mib",
        "mem_available_mib", "disk_read_mib_total", "disk_write_mib_total",
        "gpu_util_pct", "gpu_mem_mib",
    ]
    start = time.monotonic()
    previous = None
    previous_time = start
    with args.output.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        handle.flush()
        while True:
            now = time.monotonic()
            process = process_tree_sample(args.pid)
            if process is None:
                break
            host = host_sample()
            gpu_util, gpu_mem = gpu_sample(args.gpu)
            cpu_pct = None
            if previous is not None:
                cpu_pct = max(0.0, process[0] - previous[0]) * 100.0 / os.sysconf("SC_CLK_TCK") / max(now - previous_time, 1e-9)
            writer.writerow({
                "elapsed_s": round(now - start, 3),
                "cpu_pct": None if cpu_pct is None else round(cpu_pct, 1),
                "rss_mib": round(process[1] / 1024**2, 1),
                "process_read_mib": round(process[2] / 1024**2, 1),
                "process_write_mib": round(process[3] / 1024**2, 1),
                "mem_available_mib": round(host[0] / 1024**2, 1),
                "disk_read_mib_total": round(host[1] / 1024**2, 1),
                "disk_write_mib_total": round(host[2] / 1024**2, 1),
                "gpu_util_pct": gpu_util,
                "gpu_mem_mib": None if gpu_mem is None else round(gpu_mem / 1024**2, 1),
            })
            handle.flush()
            previous, previous_time = process, now
            time.sleep(args.interval)


if __name__ == "__main__":
    main()
