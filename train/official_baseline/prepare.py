"""Download NVIDIA's unchanged N1.6 sources and converted BEHAVIOR data."""
import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import hashlib
import json
from pathlib import Path
import shutil
import subprocess
import tarfile
import tempfile

CODE_REVISION = "9b37aa1ce69c73c6d165233fa88128283bba4508"
DATA_REVISION = "ea7ac0b68f87da62f1e726771bba0fe74300802f"
DATA_REPO = "nvidia/PhysicalAI-Robotics-GR00T-X-Embodiment-Sim"
TASKS = [
    "sim_behavior_r1_pro.task-0000_turning_on_radio",
    "sim_behavior_r1_pro.task-0034_hanging_pictures",
    "sim_behavior_r1_pro.task-0040_make_microwave_popcorn",
]
ENDPOINT = "https://hf-mirror.com"


def curl(url, target=None):
    command = ["curl", "-fsSL", "--connect-timeout", "8", "--max-time", "180",
               "--retry", "4", "--retry-all-errors"]
    if target is not None:
        command += ["-o", str(target)]
    return subprocess.check_output(command + [url])


def download(entry, root):
    target = root / entry["path"]
    target.parent.mkdir(parents=True, exist_ok=True)
    is_lfs = "lfs" in entry
    expected = entry["lfs"]["oid"] if is_lfs else entry["oid"]
    def verified(path):
        if not path.exists() or path.stat().st_size != entry["size"]:
            return False
        digest = hashlib.sha256() if is_lfs else hashlib.sha1()
        if not is_lfs:
            digest.update(f"blob {entry['size']}\0".encode())
        with path.open("rb") as data:
            for block in iter(lambda: data.read(1024 * 1024), b""):
                digest.update(block)
        return digest.hexdigest() == expected
    if not verified(target):
        partial = target.with_name(target.name + ".download")
        curl(f"{ENDPOINT}/datasets/{DATA_REPO}/resolve/{DATA_REVISION}/{entry['path']}", partial)
        if not verified(partial):
            raise ValueError(f"Official file checksum mismatch: {target}")
        partial.replace(target)
    return entry["size"]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, default=Path("/nvme3/datasets/groot_official"))
    parser.add_argument("--code-root", type=Path, default=Path("/nvme3/projects/GR00T/official-baseline-9b37aa1"))
    args = parser.parse_args()
    if not args.code_root.exists():
        args.code_root.parent.mkdir(parents=True, exist_ok=True)
        with tempfile.TemporaryDirectory(prefix="groot-official-") as temporary:
            archive = Path(temporary) / "source.tar.gz"
            subprocess.run(["curl", "-fL", "--connect-timeout", "8",
                            "--max-time", "120", "--retry", "2", "-o", str(archive),
                            f"https://codeload.github.com/NVIDIA/Isaac-GR00T/tar.gz/{CODE_REVISION}"], check=True)
            print("Source archive SHA256:", hashlib.sha256(archive.read_bytes()).hexdigest(), flush=True)
            with tarfile.open(archive) as source:
                source.extractall(temporary, filter="data")
            shutil.move(str(Path(temporary) / f"Isaac-GR00T-{CODE_REVISION}"), args.code_root)
    args.data_root.mkdir(parents=True, exist_ok=True)
    entries = []
    for task in TASKS:
        listing = json.loads(curl(f"{ENDPOINT}/api/datasets/{DATA_REPO}/tree/{DATA_REVISION}/{task}?recursive=true&limit=1000"))
        if len(listing) >= 1000:
            raise ValueError("Dataset listing needs pagination; refusing an incomplete download")
        entries.extend(entry for entry in listing if entry["type"] == "file")
    print("Downloading", len(entries), "unchanged official files; bytes", sum(e["size"] for e in entries), flush=True)
    with ThreadPoolExecutor(max_workers=12) as pool:
        pending = [pool.submit(download, entry, args.data_root) for entry in entries]
        total = 0
        for count, future in enumerate(as_completed(pending), 1):
            total += future.result()
            if count % 50 == 0 or count == len(entries):
                print(f"Verified {count}/{len(entries)} files, {total / 1024**3:.2f} GiB", flush=True)
    (args.data_root / "official_source.json").write_text(json.dumps({
        "repo": DATA_REPO, "revision": DATA_REVISION, "files": entries,
    }, indent=2))
    for task in TASKS:
        info = json.loads((args.data_root / task / "meta/info.json").read_text())
        print(task, {key: info.get(key) for key in ("fps", "total_episodes", "total_frames")}, flush=True)
    print("READY", args.code_root, args.data_root, flush=True)


if __name__ == "__main__":
    main()
