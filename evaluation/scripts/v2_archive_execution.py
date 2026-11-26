"""Archive measured V2 workbench evidence without launching any GPU work."""
import argparse
import csv
from datetime import datetime
import hashlib
import json
from pathlib import Path
import shutil
import subprocess

from benchmark_dashboard import BenchmarkManager, atomic_json
from v2_evaluation_summary import csv_rows


def sha256(path):
    value = hashlib.sha256()
    with path.open('rb') as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b''):
            value.update(chunk)
    return value.hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--job', required=True)
    parser.add_argument('--eval-run', default='eval001')
    parser.add_argument('--offline-job', help='Original completed diagnostics job when archiving a paired continuation')
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[2]
    experiment = root / 'Output/test_02/experiment_01'
    manager = BenchmarkManager(root)
    run = manager._directory(args.job)
    detail = manager.detail({'run': args.job})
    offline_run = manager._directory(args.offline_job) if args.offline_job else run
    if args.offline_job:
        offline_plan = json.loads((offline_run / 'plan.json').read_text())
        for field in ('checkpoint', 'protocol_sha256'):
            if offline_plan['config'][field] != detail['job']['config'][field]:
                raise ValueError('Offline and paired jobs must share '+field)
    training = manager._directory(detail['job']['config']['training_job'])
    config = json.loads((training / 'train_config.json').read_text())
    train_archive = experiment / 'train_runs/run001'
    eval_archive = experiment / 'eval_runs' / args.eval_run
    train_archive.mkdir(parents=True, exist_ok=True)
    eval_archive.mkdir(parents=True, exist_ok=True)
    atomic_json(train_archive / 'workbench_job.json', manager.detail({'run': training.name}))
    atomic_json(eval_archive / 'workbench_job.json', detail)
    for name in ('train_config.json', 'train_monitor.csv', 'preflight.json',
                 'v2_training.json', 'checkpoint_index.csv'):
        shutil.copy2(training / name, train_archive / name)
    for name in ('offline_evaluation.json', 'paired_evaluation.json', 'paired_status.json',
                 'evaluation_status.json', 'state_predictions.jsonl', 'plan.json', 'state.json'):
        source = offline_run if name in ('offline_evaluation.json', 'evaluation_status.json', 'state_predictions.jsonl') else run
        if (source / name).is_file():
            shutil.copy2(source / name, eval_archive / name)
    if args.offline_job:
        atomic_json(eval_archive / 'offline_workbench_job.json', manager.detail({'run': offline_run.name}))
    paired = json.loads((run / 'paired_status.json').read_text())
    restores, resources, samples = [], [], []
    evidence_files = []
    for result in paired.get('results', []):
        output = Path(result['output'])
        for audit in sorted((output / 'metrics').glob('*/restore.json')):
            values = json.loads(audit.read_text())
            row = dict(pair_id=result['pair_id'], version=result['version'],
                instance=result['instance'], seed=result['seed'], status=result['status'],
                reason=(result.get('validity') or {}).get('reason'),
                controllers_equal=values['controllers_equal'],
                position_max=max(values['position'].values()),
                rotation_max=max(values['rotation'].values()),
                joints_max=max(values['joints'].values()))
            row.update({camera+'_rgb_mae': mae for camera, mae in values['camera_rgb_mae_vs_anchor'].items()})
            restores.append(row)
            for path in (audit, audit.parent / 'restore_validity.json',
                         audit.parent / 'initial_correspondence.json', output / 'logs/failure.txt'):
                if path.is_file():
                    evidence_files.append(dict(path=str(path.relative_to(root)), sha256=sha256(path)))
        resource_file = output / 'resource.csv'
        if resource_file.is_file():
            rows = list(csv.DictReader(resource_file.open()))
            for row in rows:
                row.update(pair_id=result['pair_id'], version=result['version'])
                samples.append(row)
            if rows:
                used = [int(row['used_mib']) for row in rows]
                utilization = [int(row['utilization_percent']) for row in rows]
                total = int(rows[0]['used_mib']) + int(rows[0]['free_mib'])
                resources.append(dict(pair_id=result['pair_id'], version=result['version'],
                    samples=len(rows), total_mib=total, peak_device_used_mib=max(used),
                    mean_device_used_mib=sum(used)/len(used),
                    mean_device_utilization=sum(utilization)/len(utilization),
                    samples_at_or_above_70_percent=sum(v >= .7*total for v in used),
                scope='whole device usage for each source attempt; includes other existing workloads, not model allocation'))
    csv_rows(eval_archive / 'restore_audit.csv', restores)
    csv_rows(eval_archive / 'resource_summary.csv', resources)
    csv_rows(eval_archive / 'resource_samples.csv', samples)
    atomic_json(eval_archive / 'restore_evidence_index.json', evidence_files)
    checkpoint = Path(detail['job']['config']['checkpoint'])
    complete = json.loads((checkpoint / 'COMPLETE.json').read_text())
    atomic_json(train_archive / 'checkpoint_complete.json', complete)
    atomic_json(eval_archive / 'artifact_index.json', dict(
        original_job_directory=str(run), offline_job_directory=str(offline_run),
        mechanism_windows=str(offline_run / 'mechanism_windows'),
        mechanism_json_count=len(list((offline_run / 'mechanism_windows').glob('*.json'))),
        mechanism_npz_count=len(list((offline_run / 'mechanism_windows').glob('*.npz'))),
        checkpoint=str(checkpoint), checkpoint_complete_sha256=sha256(checkpoint / 'COMPLETE.json'),
        note='Large source arrays and weights retained in original workbench outputs; no duplicate copy'))
    manifest_path = experiment / 'MANIFEST.json'
    manifest = json.loads(manifest_path.read_text())
    terminal = paired.get('status') == 'complete' and detail['job']['status'] == 'complete'
    manifest.update(status=('EXECUTION_COMPLETE_WITH_SCIENTIFIC_LIMITATIONS' if terminal
                           else 'TRAINING_AND_OFFLINE_COMPLETE_PAIRED_RUNNING'),
        checked_at=datetime.now().astimezone().isoformat(),
        code_commit=config['code_commit'], training_code_commit=config['code_commit'],
        executed_training_runner_sha256=config['executed_runner_sha256'],
        final_tooling_commit=subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=root, text=True).strip(),
        paired_attempts_completed=paired.get('attempts_completed', 0),
        closed_loop_pairs_completed=paired.get('valid_pairs', 0),
        invalid_attempts=sum(r['status'] == 'invalid' for r in paired.get('results', [])),
        scientific_state_learning_passed=False,
        scientific_closed_loop_improvement_demonstrated=False)
    manifest['blockers'] = (['Some restored starts failed the recorded restore acceptance; inspect per-attempt audits']
        if paired.get('invalid_attempts') or any(r['status'] == 'invalid' for r in paired.get('results', [])) else [])
    manifest['artifacts'] = {str(path.relative_to(experiment)): sha256(path)
        for path in sorted(experiment.rglob('*'))
        if path.is_file() and path != manifest_path
        and path.suffix in ('.json', '.jsonl', '.csv', '.md', '.py', '.xml', '.png', '.pdf')}
    atomic_json(manifest_path, manifest)
    print(json.dumps(dict(job=args.job, terminal=terminal,
        attempts=manifest['paired_attempts_completed'], invalid=manifest['invalid_attempts'],
        valid_pairs=manifest['closed_loop_pairs_completed'], indexed_files=len(manifest['artifacts']))))


if __name__ == '__main__':
    main()
