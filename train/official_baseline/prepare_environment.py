"""Reuse installed dependency versions in an isolated single-GPU environment."""
from pathlib import Path
import subprocess


destination = Path("/nvme3/projects/GR00T/official-baseline-env")
source = Path("/nvme3/projects/GR00T/Isaac-GR00T-n1d6-9b37aa1/.venv/lib/python3.10/site-packages")
subprocess.run(["/usr/bin/python3.10", "-m", "venv", "--without-pip", str(destination)], check=True)
packages = destination / "lib/python3.10/site-packages"
for entry in source.iterdir():
    # DeepSpeed is optional for single-GPU training. The existing installation
    # imports CUDA-build checks despite no CUDA toolkit on this workstation.
    # Keep the shared environment untouched and omit it from this private one.
    if entry.name.startswith(("deepspeed", "gr00t", "__editable__")) or entry.suffix == ".pth":
        continue
    target = packages / entry.name
    if not target.exists():
        target.symlink_to(entry, target_is_directory=entry.is_dir())
print(destination / "bin/python")
