"""Fetch pinned official native demonstrations and publish verified episode manifests."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import time
from urllib.parse import quote
from urllib.request import urlopen

import h5py
import numpy as np

from .protocol import (CONTRACT, DATA_REPO, DATA_REVISION, DATA_ROOT, HORIZONS,
                       atomic_json, dataset_path, manifest_path, read_json, task_names)


def inspect_dataset(file):
    episodes, rejected = [], []
    with h5py.File(file, 'r') as data:
        convention = str(data['data'].attrs.get('macros_image_convention', 'opengl'))
        if convention != 'opengl':
            raise ValueError(f'Unexpected camera convention: {convention}')
        for key in sorted(data['data'], key=lambda k: int(k.split('_')[-1])):
            demo = data['data'][key]
            actions, states, gripper = (demo[k][:] for k in ('actions', 'obs/ee_states', 'obs/gripper_states'))
            n = len(actions)
            if not n or actions.shape != (n, 7) or states.shape != (n, 6) or gripper.shape != (n, 2):
                raise ValueError(f'Invalid dimensions in {file.name}/{key}')
            if not all(np.isfinite(a).all() for a in (actions, states, gripper)):
                raise ValueError(f'Nonfinite demonstration {key}')
            for camera in ('agentview_rgb', 'eye_in_hand_rgb'):
                images = demo['obs/'+camera]
                if len(images) != n or images.shape[-1] != 3 or images.dtype != np.uint8:
                    raise ValueError(f'Invalid camera {key}/{camera}')
            # Official create_dataset.py assigns final rewards/dones=1, so these
            # are format checks, not evidence that failed trajectories were removed.
            success = demo.attrs.get('success', demo.attrs.get('successful', None))
            if success is not None and not bool(success):
                rejected.append(key)
                continue
            episodes.append(dict(name=key, length=n))
    if not episodes:
        raise ValueError('No usable demonstrations')
    return dict(episodes=episodes, rejected=rejected, camera_convention=convention,
                success_filter='Explicit failed flags excluded; original final rewards are synthetic, not a success audit')


def prepare_task(suite, task, data_root=DATA_ROOT, endpoint=None):
    endpoint = (endpoint or os.environ.get('LIBERO_HF_ENDPOINT', 'https://hf-mirror.com')).rstrip('/')
    names = task_names()
    if suite not in HORIZONS or not 0 <= task < len(names[suite]):
        raise ValueError('Invalid LIBERO task')
    file = dataset_path(suite, task, data_root)
    marker = manifest_path(suite, task, data_root)
    ready = read_json(marker, {})
    if ready.get('revision') == DATA_REVISION and file.is_file() and file.stat().st_size == ready.get('bytes'):
        return ready
    relative = suite+'/'+file.name
    with urlopen(f'{endpoint}/api/datasets/{DATA_REPO}/tree/{DATA_REVISION}/{suite}', timeout=30) as response:
        entries = json.load(response)
    entry = next(e for e in entries if e.get('path') == relative)
    expected = entry['lfs']['oid']
    file.parent.mkdir(parents=True, exist_ok=True)
    if not file.exists():
        temporary = file.with_suffix('.download')
        url = f'{endpoint}/datasets/{DATA_REPO}/resolve/{DATA_REVISION}/{quote(relative)}?download=true'
        size, last = 0, time.monotonic()
        with urlopen(url, timeout=60) as response, temporary.open('wb') as stream:
            while chunk := response.read(4*1024*1024):
                stream.write(chunk); size += len(chunk)
                if time.monotonic()-last > 3:
                    print(json.dumps(dict(download=relative, bytes=size, total_bytes=entry['size'])), flush=True)
                    last = time.monotonic()
        temporary.replace(file)
    digest = hashlib.sha256()
    with file.open('rb') as stream:
        while chunk := stream.read(8*1024*1024): digest.update(chunk)
    if digest.hexdigest() != expected:
        raise ValueError(f'Dataset checksum mismatch: {file}; file retained for inspection')
    result = dict(suite=suite, task=task, name=names[suite][task], file=str(file),
                  repo=DATA_REPO, revision=DATA_REVISION, sha256=expected, bytes=file.stat().st_size,
                  contract=CONTRACT, **inspect_dataset(file))
    atomic_json(marker, result)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--suite', choices=HORIZONS, required=True)
    parser.add_argument('--tasks', type=int, nargs='+', required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--data-root', type=Path, default=DATA_ROOT)
    args = parser.parse_args()
    results = [prepare_task(args.suite, task, args.data_root) for task in args.tasks]
    atomic_json(args.output, dict(type='libero_dataset', passed=True, suite=args.suite,
                                 datasets=results, episodes=sum(len(r['episodes']) for r in results)))
    print(json.dumps(dict(passed=True, tasks=args.tasks, episodes=sum(len(r['episodes']) for r in results))), flush=True)


if __name__ == '__main__': main()
