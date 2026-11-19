#!/usr/bin/env python3
"""Organize legacy bundles without relocating paths used by live evaluations."""

import argparse
import json
from pathlib import Path
import re


def organize(root, version="V_1"):
    root = Path(root)
    if not re.fullmatch(r'\w[\w.-]*', version):
        raise ValueError('version must be a single directory name')
    version_root = root / version
    version_root.mkdir(parents=True, exist_ok=True)

    def active(source):
        paths = [source / 'results.json'] if (source / 'results.json').is_file() else source.glob('*/results.json')
        return any(json.loads(path.read_text()).get('status') == 'running' for path in paths)

    def move(source, target):
        if source.is_symlink():
            return
        if active(source):
            print(f'ACTIVE {source}; original writer path retained')
            return
        if target.exists():
            # Merge group directories only, never merge or overwrite a run.
            if (source / 'results.json').is_file():
                raise FileExistsError(f'Different run already exists: {target}')
            children = list(source.iterdir())
            for child in children:
                if (target / child.name).exists():
                    raise FileExistsError(f'Different entry already exists: {target / child.name}')
            for child in children:
                child.rename(target / child.name)
            source.rmdir()
        else:
            source.rename(target)
        print(f'MOVED {source} -> {target}')

    # Move the former version directories intact, including historical Markdown.
    for source in sorted(root.iterdir()):
        if source.is_dir() and (source.name == 'baseline' or re.fullmatch(r'exper(?:i)?ment_[1-9][0-9]*', source.name)):
            group = source.name.replace('experiment_', 'experment_', 1)
            move(source, version_root / group)

    baseline = version_root / 'baseline'
    baseline.mkdir(exist_ok=True)
    (version_root / 'experment_1').mkdir(exist_ok=True)
    for source in sorted(root.iterdir()):
        if not re.fullmatch(r"(?:groot|grootmeory)_task[0-9]+_[0-9]+", source.name):
            continue
        result_path = source / "results.json"
        if source.is_symlink() or not result_path.is_file():
            continue
        target = baseline / source.name
        move(source, target)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--version", default="V_1", help="target version directory (default V_1)")
    args = parser.parse_args()
    organize(args.root, args.version)
