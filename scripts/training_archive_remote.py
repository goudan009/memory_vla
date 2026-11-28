"""Read-only collector executed over the workbench device's SSH connection.

REQUEST is supplied by training_archive.py. stdout is a gzip JSON envelope;
model/optimizer tensors, datasets, videos and machine credentials are excluded.
"""
import base64
import gzip
import hashlib
import json
from pathlib import Path
import sys


def collect(request):
    root = Path(request['output'])
    arm = request['arm']
    source = Path(request['source']) if request.get('source') else None
    files, origins = {}, {}

    def add(path, name, tail=None):
        if not path.is_file() or path.is_symlink():
            return
        if tail:
            size = path.stat().st_size
            with path.open('rb') as stream:
                stream.seek(max(0, size - tail))
                data = stream.read()
            extra = dict(selection='last bytes', source_bytes=size, offset=max(0, size-tail))
        else:
            data = path.read_bytes()
            extra = dict(selection='complete file')
        files[name] = base64.b64encode(data).decode()
        origins[name] = dict(path=str(path), bytes=len(data), sha256=hashlib.sha256(data).hexdigest(), **extra)

    metric = next((p for p in (root/arm/'metrics/metrics.jsonl', root/'run/metrics.jsonl') if p.is_file()), None)
    if metric:
        add(metric, 'metrics.jsonl')
    patterns = ['config/*.json', 'config/web_resumes/resume-*/*.json', arm+'/config/*.json',
                'status.json', arm+'/status.json', 'run/status.json', 'run/preflight.json',
                'run/stop.json', 'run/validation-*.json', 'run/native_contract.json',
                'run/native_runtime.json', 'run/data_split.json', 'run/prepared/PREPARED.json',
                'run/prepared/normalization_audit.json', 'run/prepared/train/meta/stats.json',
                'run/prepared/train/meta/relative_stats.json', 'run/prepared/train/meta/modality.json',
                'metrics/stop_training.json', 'dataset/meta/info.json',
                'dataset/meta/tasks.jsonl']
    for pattern in patterns:
        for path in sorted(root.glob(pattern)):
            add(path, 'recorded/'+str(path.relative_to(root)))
    for checkpoint_root in (root/arm/'checkpoints', root/'run/checkpoints'):
        for pattern in ('latest.json', '*/*.json', '*/training_state/*.json'):
            for path in sorted(checkpoint_root.glob(pattern)):
                add(path, 'recorded/'+str(path.relative_to(root)))
    for path in sorted((root/'logs').glob('*.log')):
        add(path, 'recorded/logs/'+path.name, tail=256*1024)
    if source and source.is_dir():
        add(source/'web_source_manifest.json', 'source/web_source_manifest.json')
        # Preserve the implementations used by the two supported training backends.
        for package in ('v1_groot_memoryvla', 'v2_1_groot_memoryvla'):
            for path in sorted((source/'src'/package).rglob('*')):
                if path.suffix in ('.py', '.json', '.yaml', '.yml') and not {'tests', '__pycache__'} & set(path.parts):
                    add(path, 'source/'+str(path.relative_to(source)))
        for name in ('train/run_groot_train.sh', 'train/web_training_resume.py', 'train/run_identity.py',
                     'train/v21_workbench.py', 'scripts/stage1_native_train.py',
                     'scripts/stage1_native_prepare.py', 'pyproject.toml'):
            add(source/name, 'source/'+name)
    return dict(files=files, origins=origins, metric_found=metric is not None)


if __name__ == '__main__':
    sys.stdout.buffer.write(gzip.compress(json.dumps(collect(REQUEST)).encode()))
