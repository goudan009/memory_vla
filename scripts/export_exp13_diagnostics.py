"""Export existing Exp13 evidence for GitHub readers; never starts GPU work.

Run once from the 4090 repository. --verify-only checks the exported snapshot
without SSH, workbench access, model libraries, or the original training data.
"""
import argparse
from collections import defaultdict, deque
import csv
from datetime import datetime
import hashlib
import io
import json
from pathlib import Path
import subprocess
import tarfile
from urllib.request import ProxyHandler, build_opener

ROOT = Path(__file__).resolve().parents[1]
RUN = 'v1__test01__exp13__run001'
STEP = 16000
RUN_ROOT = '/nvme3/data_wyh/outputs/memory_vla/v1_groot_memoryvla/training/' + RUN
SOURCE = '/nvme3/projects/memory_vla/deployments/' + RUN + '__resume002'
BUNDLE = ROOT/'Output/test_01/experiment_13/diagnostics/step016000'
ALLOWED = {'.py', '.json', '.jsonl', '.csv', '.yaml', '.yml', '.toml', '.lock', '.md', '.sh', '.log'}


def sha(data):
    return hashlib.sha256(data).hexdigest()


def dump(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + '\n')


def get(action):
    opener = build_opener(ProxyHandler({}))
    with opener.open('http://127.0.0.1:8765/api/' + action, timeout=30) as response:
        return json.load(response)


def ssh(host, command, data=None):
    return subprocess.check_output(['ssh', '-o', 'BatchMode=yes', '-o', 'ConnectTimeout=5',
        host, command], input=data, timeout=90)


def csv_file(path, rows, fields):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('w', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=fields, extrasaction='ignore')
        writer.writeheader()
        writer.writerows(rows)


def verify():
    manifest = json.loads((BUNDLE/'MANIFEST.json').read_text())
    for relative, info in manifest['files'].items():
        content = (BUNDLE/relative).read_bytes()
        assert len(content) == info['bytes'] and sha(content) == info['sha256'], relative
    metrics = b''.join((BUNDLE/part).read_bytes() for part in manifest['metrics']['parts'])
    assert sha(metrics) == manifest['metrics']['source_sha256']
    rows = [json.loads(line) for line in metrics.splitlines()]
    assert [r['step'] for r in rows if r.get('kind') == 'train'] == list(range(1, STEP+1))
    assert manifest['training_job']['step'] == STEP
    print(json.dumps(dict(verified_files=len(manifest['files']), train_steps=STEP,
        raw_metrics_sha256=sha(metrics), total_bytes=sum(f['bytes'] for f in manifest['files'].values()))))


def export():
    if BUNDLE.exists():
        raise SystemExit('Snapshot already exists; use --verify-only. Do not overwrite published evidence.')
    job = next(j for j in get('training/jobs?device=a100')['jobs'] if j['run'] == RUN)
    assert job['step'] == STEP and job['training_complete'], job
    assert job['source'] == SOURCE
    detail = get('training/detail?device=a100&run=' + RUN + '&arm=memory_v1')
    device = json.loads((ROOT/'configs/workbench_devices.json').read_text())['a100']
    host = device['host']
    # Enumerate only training metadata and the actual saved implementation.
    # No weights, optimizer tensors, auth files, or machine routing are read.
    remote_code = '''from pathlib import Path
import json
r=Path(RUN_ROOT); s=Path(SOURCE); selected={}
def add(path, target):
 if path.is_file(): selected[str(path)]=target
for pattern in ('config/*.json','config/web_resumes/resume-*/request.json',
                'config/web_resumes/resume-*/engineering_revision.json',
                'memory_v1/config/*.json','memory_v1/metrics/*.jsonl',
                'logs/controller.log','logs/memory_v1_torchrun.log',
                'logs/*step016000*.log','status.json','memory_v1/status.json',
                'metrics/stop_training.json'):
 for p in r.glob(pattern): add(p,'training/recorded/'+str(p.relative_to(r)))
checkpoint=r/'memory_v1/checkpoints'/CHECKPOINT
for p in checkpoint.glob('*.json'): add(p,'training/checkpoint/'+p.name)
for p in (checkpoint/'training_state').glob('*.json'): add(p,'training/checkpoint/training_state/'+p.name)
base=Path('/nvme3/huggingface/train_model/GR00T/GR00T-N1.6-3B')
for name in ('config.json','processor_config.json','statistics.json','embodiment_id.json'):
 add(base/name,'training/base_checkpoint/'+name)
for name in ('info.json','episodes.jsonl','tasks.jsonl'):
 add(r/'dataset/meta'/name,'training/data_metadata/'+name)
pkg=s/'src/v1_groot_memoryvla'
for directory in ('experiment','model','data','configs','policy','utils'):
 for p in (pkg/directory).rglob('*'):
  if p.suffix in ('.py','.json','.yaml','.yml') and '__pycache__' not in p.parts:
   add(p,'source/'+str(p.relative_to(s)))
for name in ('src/v1_groot_memoryvla/__init__.py','src/v1_groot_memoryvla/serve.py',
             'src/v1_groot_memoryvla/LICENSE','train/run_groot_train.sh',
             'train/web_training_resume.py','train/run_identity.py','pyproject.toml','uv.lock',
             'web_source_manifest.json'):
 add(s/name,'source/'+name)
print(json.dumps(selected))
'''
    code = 'RUN_ROOT=' + repr(RUN_ROOT) + '\nSOURCE=' + repr(SOURCE) + '\nCHECKPOINT=' + repr(RUN+'__step016000') + '\n' + remote_code
    selected = json.loads(ssh(host, 'python3 -', code.encode()))
    # Fetch whitelisted files as one archive; extraction uses explicit destinations.
    archive = ssh(host, 'tar -czf - -C / --null -T -',
        b'\0'.join(p.lstrip('/').encode() for p in selected) + b'\0')
    BUNDLE.mkdir(parents=True)
    origins = {}

    def save(relative, data, origin, **extra):
        path = BUNDLE/relative
        assert path.suffix in ALLOWED or path.name == 'LICENSE', relative
        assert len(data) < 10*1024*1024, (relative, len(data))
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(data)
        origins[relative] = dict(origin=origin, **extra)

    metrics_data = None
    with tarfile.open(fileobj=io.BytesIO(archive), mode='r:gz') as tar:
        for member in tar:
            original = '/' + member.name
            relative = selected[original]
            data = tar.extractfile(member).read()
            if relative.endswith('/metrics/metrics.jsonl'):
                metrics_data = data
            else:
                save(relative, data, 'a100:' + original)
    assert metrics_data is not None
    raw_rows = [json.loads(line) for line in metrics_data.splitlines()]
    train = [r for r in raw_rows if r.get('kind') == 'train']
    assert [r['step'] for r in train] == list(range(1, STEP+1))
    assert len(raw_rows) == len(train), 'Unexpected non-training rows: preserve explicitly before exporting'
    lines = metrics_data.splitlines(keepends=True)
    parts = []
    for offset in range(0, STEP, 1000):
        relative = f'training/raw_metrics/step{offset+1:06d}-{offset+1000:06d}.jsonl'
        save(relative, b''.join(lines[offset:offset+1000]), 'a100:' + RUN_ROOT + '/memory_v1/metrics/metrics.jsonl',
             first_step=offset+1, last_step=offset+1000)
        parts.append(relative)
    csv_file(BUNDLE/'training/loss_per_step.csv', train, ['step', 'epoch', 'loss'])
    gradient_rows = []
    for row in train:
        flat = {k: row.get(k) for k in ('step', 'gradient_norm', 'gradient_norm_after_clip', 'learning_rate')}
        flat.update({'component.'+k: v for k, v in row.get('component_grad_norms', {}).items()})
        flat.update({'flow_mse.'+k: v for k, v in row.get('body_flow_mse', {}).items()})
        gradient_rows.append(flat)
    fields = ['step', 'gradient_norm', 'gradient_norm_after_clip', 'learning_rate'] + sorted(
        {k for row in gradient_rows for k in row} - {'step', 'gradient_norm', 'gradient_norm_after_clip', 'learning_rate'})
    csv_file(BUNDLE/'training/gradients_and_body_loss.csv', gradient_rows, fields)
    epochs = []
    for epoch in sorted({r['epoch'] for r in train}):
        batch = [r for r in train if r['epoch'] == epoch]
        epochs.append(dict(epoch_zero_based=epoch, first_step=batch[0]['step'], last_step=batch[-1]['step'],
            steps=len(batch), mean_loss=sum(r['loss'] for r in batch)/len(batch), complete=len(batch)==625))
    dump(BUNDLE/'training/epoch_means.json', epochs)
    dump(BUNDLE/'training/effective_parameters.json', detail['training_parameters'])
    dump(BUNDLE/'training/workbench_job.json', job)
    dump(BUNDLE/'training/workbench_detail.json', detail)
    manifest_source = json.loads((BUNDLE/'source/web_source_manifest.json').read_text())
    assert manifest_source['digest'] == job['source_digest']
    for relative in origins:
        if relative.startswith('source/'):
            name = relative.removeprefix('source/')
            if name in manifest_source['files']:
                assert sha((BUNDLE/relative).read_bytes()) == manifest_source['files'][name], relative

    evals = []
    jobs = get('launch/jobs')['jobs']
    for result in sorted((ROOT/'evaluation').glob('V_*/**/results.json')):
        data = result.read_bytes()
        value = json.loads(data)
        identity = value.get('policy_identity') or value.get('configuration', {}).get('policy_identity') or {}
        if identity.get('training_id') != RUN or not identity.get('checkpoint', '').endswith('__step016000'):
            continue
        run_dir = result.parent
        prefix = 'evaluation/' + run_dir.name + '/'
        save(prefix + 'results.json', data, str(result))
        eval_job = next((j for j in jobs if j.get('output') == str(run_dir)), {})
        dump(BUNDLE/(prefix+'workbench_job.json'), eval_job)
        for name in ('episodes.jsonl', 'gpu0.log', 'gpu1.log'):
            p = run_dir/name
            if p.is_file():
                save(prefix+name, p.read_bytes(), str(p))
        for p in sorted((run_dir/'stages').glob('*.jsonl')):
            # Logs are live: take complete lines from this read, never rewrite the source.
            content = p.read_bytes()
            complete_lines = content.splitlines(keepends=True)
            if complete_lines and not complete_lines[-1].endswith(b'\n'):
                complete_lines.pop()
            content = b''.join(complete_lines)
            if p.name.startswith('actions_'):
                first, last = defaultdict(list), defaultdict(lambda: deque(maxlen=8))
                for line in complete_lines:
                    row = json.loads(line)
                    key = (row.get('env_idx'), row.get('episode_number'), row.get('instance_id'), row.get('seed'))
                    if len(first[key]) < 8:
                        first[key].append(line)
                    last[key].append(line)
                sampled = b''.join(line for key in sorted(first) for line in
                    first[key] + [line for line in last[key] if line not in first[key]])
                save(prefix+'stages/'+p.stem+'_sample.jsonl', sampled, str(p),
                    selection='first 8 and last 8 recorded chunks per (env_idx, episode_number, instance_id, seed)',
                    source_complete_lines=len(complete_lines), source_snapshot_sha256=sha(content),
                    source_snapshot_bytes=len(content))
            else:
                save(prefix+'stages/'+p.name, content, str(p))
        for p in sorted((run_dir/'.runtime').glob('gpu*/*/*.json')):
            if p.name in ('results.json', 'run_meta.json', 'profiling.json', 'runtime_env0.json', 'runtime_env1.json'):
                save(prefix+'runtime/'+p.parent.parent.name+'/'+p.name, p.read_bytes(), str(p))
        # The original runner's complete per-episode data include goal predicates.
        for p in sorted((run_dir/'.runtime').glob('gpu*/*/episodes.jsonl')):
            save(prefix+'runtime/'+p.parent.parent.name+'/episodes.jsonl', p.read_bytes(), str(p))
        evals.append(dict(folder=str(run_dir.relative_to(ROOT/'evaluation')), evaluation_id=value.get('evaluation_id'),
            status=value.get('status'), workbench_state=eval_job.get('state'),
            requested=value.get('requested_episodes'), completed=value.get('completed_episodes'),
            valid=value.get('official_episodes'), successful=value.get('successful_episodes'),
            checkpoint=identity.get('checkpoint'), source_digest=identity.get('source_digest')))
    assert evals, 'No evaluation is explicitly bound to the step16000 checkpoint'
    dump(BUNDLE/'evaluation/index.json', evals)
    for name in ('training_review_20261124.md', 'v1_paper_fidelity_review_20261124.md', 'optimization_20261125.md',
                 'launch_record_20261125.md', 'pause_resume_proof.json', 'MANIFEST.json', 'checkpoint_index.csv'):
        p = ROOT/'Output/test_01/experiment_13'/name
        save('historical/'+name, p.read_bytes(), str(p))
    # Manifest covers payload files; the human reading guide is maintained separately.
    files = {}
    for p in sorted(BUNDLE.rglob('*')):
        if p.is_file():
            data = p.read_bytes()
            relative = p.relative_to(BUNDLE).as_posix()
            files[relative] = dict(bytes=len(data), sha256=sha(data),
                **origins.get(relative, dict(origin='derived from exported workbench/metrics metadata')))
    manifest = dict(exported_at=datetime.now().astimezone().isoformat(), run=RUN, optimizer_step=STEP,
        repository_head_at_export=subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=ROOT, text=True).strip(),
        training_source=SOURCE, training_source_digest=job['source_digest'], training_job=job,
        metrics=dict(source_bytes=len(metrics_data), source_sha256=sha(metrics_data), parts=parts,
            training_rows=STEP, fields_available_at_sampled_steps_only=['component_grad_norms','memory_gradients']),
        evaluation_snapshots=evals, files=files,
        exclusions=['model and optimizer weights', 'dataset frames and videos', 'auth and machine routing',
            'full evaluation action chunks (sample and full-log hash exported)', 'unrelated dirty working-tree files'],
        training_submitted=False)
    dump(BUNDLE/'MANIFEST.json', manifest)
    verify()
    print(json.dumps(dict(evaluations=evals, epochs=epochs[-2:]), ensure_ascii=False))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--verify-only', action='store_true')
    args = parser.parse_args()
    verify() if args.verify_only else export()
