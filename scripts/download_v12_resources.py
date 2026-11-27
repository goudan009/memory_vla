"""Pinned CPU-only V1.2 downloads. No training, simulation, or global configuration changes."""
import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import fcntl
import hashlib
import json
import os
from pathlib import Path
import subprocess
import threading
import time
from urllib.parse import quote

ROOT = Path('/nvme3/outputs/memory_vla/v12-downloads')
MANIFEST = Path(__file__).resolve().parents[1] / 'docs/v1_2/downloads/manifest.json'
SOURCES = [
    ('n17', 'nvidia/GR00T-N1.7-3B', 'model', '373d34e2723d928870157d07dc216a19a624b5bb', '/nvme3/models/GR00T-N1.7-3B'),
    ('demos', 'behavior-1k/2026-challenge-demos', 'dataset', '4f50b44796641a4d526a19d9aeadc8aa51e2f2c2', '/nvme3/datasets/b1k2026-task0-task40'),
    ('assets', 'behavior-1k/zipped-datasets', 'dataset', '9f0d57d465726976ed98138d3f8b8ca3e2186775', '/nvme3/datasets/b1k2026-assets/archives'),
]
ARCHIVES = {'behavior-1k-assets-3.9.0.zip', 'omnigibson-robot-assets-3.8.2.zip', '2026-challenge-task-instances.zip'}


def select_demo(path):
    parts = path.split('/')
    if parts[0] == 'annotations':
        return len(parts) == 3 and parts[1] in {'task-0000', 'task-0040'}
    if parts[0] == 'data':
        return len(parts) == 3 and parts[1] in {'chunk-000', 'chunk-040'}
    if parts[0] == 'meta':
        return (len(parts) == 2 or (len(parts) == 4 and parts[1] == 'episodes' and parts[2] in {'chunk-000', 'chunk-040'}))
    return (len(parts) == 4 and parts[0] == 'videos' and parts[1] in {
        'observation.rgb.left_realsense_link_camera_0',
        'observation.rgb.right_realsense_link_camera_0',
        'observation.rgb.zed_link_camera_0',
    } and parts[2] in {'chunk-000', 'chunk-040'})


def atomic_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(json.dumps(value, indent=2, ensure_ascii=False) + '\n')
    temporary.replace(path)


def prepare():
    from huggingface_hub import HfApi
    resources = []
    for name, repo, kind, revision, target in SOURCES:
        info = HfApi().repo_info(repo, repo_type=kind, revision=revision, files_metadata=True)
        assert info.sha == revision
        files = []
        for f in info.siblings:
            if name == 'demos' and not select_demo(f.rfilename):
                continue
            if name == 'assets' and f.rfilename not in ARCHIVES:
                continue
            files.append({'path': f.rfilename, 'bytes': f.size,
                          'sha256': f.lfs.sha256 if f.lfs else None, 'git_blob_sha1': f.blob_id})
        resources.append(dict(name=name, repo=repo, repo_type=kind, revision=revision,
                              target=target, total_bytes=sum(f['bytes'] for f in files), files=files))
    atomic_json(MANIFEST, {'schema': 1, 'resources': resources,
        'cosmos': {'repo': 'nvidia/Cosmos-Reason2-2B', 'revision': '9ce19a195e423419c349abfc86fd07178b230561',
                   'status': 'blocked_gated_http401', 'target': '/nvme3/models/Cosmos-Reason2-2B'},
        'radio': {'url': 'https://drive.google.com/file/d/1OXNm3SPLvWOSJR1e8In6xHMHxOYDp789/view',
                  'target': '/nvme3/checkpoints/memory_vla/official-n17-radio'}})
    print(json.dumps([{k: r[k] for k in ('name', 'total_bytes', 'target')} | {'files': len(r['files'])} for r in resources]))


def verify(path, entry):
    if path.stat().st_size != entry['bytes']:
        raise ValueError('size mismatch: ' + str(path))
    sha256 = hashlib.sha256()
    blob = hashlib.sha1(('blob ' + str(entry['bytes']) + '\0').encode())
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b''):
            sha256.update(block)
            blob.update(block)
    if entry['sha256']:
        assert sha256.hexdigest() == entry['sha256'], str(path)
    else:
        assert blob.hexdigest() == entry['git_blob_sha1'], str(path)
    return sha256.hexdigest()


def resume_large_file(resource, entry, path):
    """Use one stable partial; hub 2.4.0 intentionally discards process-unique partials."""
    path.parent.mkdir(parents=True, exist_ok=True)
    partial = path.with_suffix(path.suffix + '.part')
    if not partial.exists() and entry['sha256']:
        candidates = list((Path(resource['target']) / '.cache/huggingface/download').rglob('*.' + entry['sha256'] + '.*.incomplete'))
        if candidates:
            max(candidates, key=lambda p: p.stat().st_size).replace(partial)
    prefix = 'datasets/' if resource['repo_type'] == 'dataset' else ''
    url = 'https://huggingface.co/' + prefix + resource['repo'] + '/resolve/' + resource['revision'] + '/' + quote(entry['path'])
    result = subprocess.run(['curl', '--fail', '--location', '--silent', '--show-error', '--connect-timeout', '30',
                             '--max-time', '900', '--speed-time', '60', '--speed-limit', '1024',
                             '--continue-at', '-', '--output', str(partial), url])
    if result.returncode:
        raise RuntimeError('curl exit ' + str(result.returncode) + '; partial preserved: ' + str(partial))
    verify(partial, entry)
    partial.replace(path)
    return path


def download():
    from huggingface_hub import hf_hub_download
    ROOT.mkdir(parents=True, exist_ok=True)
    lock = (ROOT / 'download.lock').open('w')
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    state = {'pid': os.getpid(), 'started_at': time.time(), 'status': 'running', 'files': {}}
    mutex = threading.Lock()
    done = threading.Event()
    manifest = json.loads(MANIFEST.read_text())
    def save():
        with mutex:
            state['updated_at'] = time.time()
            state['verified_bytes'] = sum(x.get('bytes', 0) for x in state['files'].values() if x['status'] == 'verified')
            state['partial_bytes'] = sum(p.stat().st_size for r in manifest['resources'] for p in Path(r['target']).rglob('*.part') if p.is_file())
            atomic_json(ROOT / 'progress.json', state)
    def heartbeat():
        while not done.wait(30):
            save()
    threading.Thread(target=heartbeat, daemon=True).start()
    def fetch(resource, entry):
        key = resource['name'] + '/' + entry['path']
        with mutex:
            state['files'][key] = {'status': 'downloading', 'bytes': entry['bytes']}
        for attempt in range(1, 4):
            try:
                path = Path(resource['target']) / entry['path']
                reused = False
                if path.is_file():
                    try:
                        verify(path, entry)
                        reused = True
                    except (ValueError, AssertionError):
                        pass
                if not reused:
                    if entry['bytes'] >= 10 * 1024 * 1024:
                        path = resume_large_file(resource, entry, path)
                    else:
                        path = hf_hub_download(resource['repo'], entry['path'], repo_type=resource['repo_type'],
                                               revision=resource['revision'], local_dir=resource['target'])
                digest = verify(Path(path), entry)
                with mutex:
                    state['files'][key] = {'status': 'verified', 'bytes': entry['bytes'], 'sha256': digest, 'path': str(path), 'reused': reused}
                print('VERIFIED', key, entry['bytes'], flush=True)
                save()
                return
            except Exception as error:
                with mutex:
                    state['files'][key] = {'status': 'retrying' if attempt < 3 else 'failed', 'attempt': attempt,
                                           'error': type(error).__name__ + ': ' + str(error)[:600]}
                save()
                print('ERROR', key, type(error).__name__, flush=True)
                if attempt < 3:
                    time.sleep(5 * attempt)
    save()
    def fetch_resource(resource):
        with ThreadPoolExecutor(max_workers=2) as pool:
            futures = [pool.submit(fetch, resource, entry) for entry in resource['files']]
            for future in as_completed(futures):
                future.result()
    with ThreadPoolExecutor(max_workers=len(manifest['resources'])) as pool:
        list(pool.map(fetch_resource, manifest['resources']))
    done.set()
    state['status'] = 'complete' if all(f['status'] == 'verified' for f in state['files'].values()) else 'partial_failure'
    save()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action', choices=['prepare', 'download'])
    args = parser.parse_args()
    prepare() if args.action == 'prepare' else download()
