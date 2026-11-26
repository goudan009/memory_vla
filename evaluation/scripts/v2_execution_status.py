"""One read-only workbench inspection, persisted for the active V2 experiment."""
import argparse
import csv
from datetime import datetime, timedelta
import json
from pathlib import Path
import subprocess
from urllib.request import urlopen

from benchmark_dashboard import BenchmarkManager, atomic_json, read_json


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--job', required=True)
    parser.add_argument('--offline-job', help='Completed diagnostics source for a paired continuation')
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[2]
    experiment = root / 'Output/test_02/experiment_01'
    manager = BenchmarkManager(root)
    directory = manager._directory(args.job)
    job = manager._row(directory)
    training_directory = (manager._directory(job['config']['training_job'])
        if job['config'].get('training_job') else directory)
    labels = read_json(training_directory / 'labels.json', {})
    training = read_json(training_directory / 'training_status.json', {})
    offline_directory = manager._directory(args.offline_job) if args.offline_job else directory
    if args.offline_job:
        offline_config = read_json(offline_directory / 'plan.json')['config']
        if any(offline_config[key] != job['config'][key] for key in ('checkpoint','protocol_sha256')):
            raise ValueError('Diagnostic source must match the active paired job')
    evaluation = read_json(offline_directory / 'evaluation_status.json', {})
    paired = read_json(directory / 'paired_status.json', {})
    now = datetime.now().astimezone()
    terminal = job['status'] in ('complete', 'failed', 'interrupted', 'stopped')
    row = dict(checked_at=now.isoformat(), job_id=args.job, status=job['status'],
        stage=job.get('stage'), decoded_episodes=len(labels.get('episodes', [])),
        decoded_frames=sum(e['frame_count'] for e in labels.get('episodes', [])),
        optimizer_steps=training.get('step', training.get('steps', 0)),
        evaluation_episodes=evaluation.get('episodes_completed', 0),
        diagnostic_windows=max(evaluation.get('windows_completed', 0),
            len(list((offline_directory / 'mechanism_windows').glob('*.json')))),
        paired_attempts=paired.get('attempts_completed', 0),
        paired_valid=paired.get('valid_pairs'),
        training=training, next_check_at=None if terminal else (now+timedelta(minutes=2)).isoformat(),
        next_check_reason='Terminal job: archive and accept measured results' if terminal else
            'Track real training/evaluation increments and stage transitions; investigate any terminal failure before resubmitting')
    with urlopen('http://127.0.0.1:8765/api/services', timeout=15) as response:
        row['services'] = json.load(response)
    resources = subprocess.check_output(['nvidia-smi',
        '--query-gpu=index,uuid,memory.used,memory.total,utilization.gpu',
        '--format=csv,noheader,nounits'], text=True)
    row['local_resources'] = resources.strip().splitlines()
    atomic_json(experiment / 'watchdog_state.json', row)
    with (experiment / 'watchdog_log.jsonl').open('a') as stream:
        stream.write(json.dumps(row, ensure_ascii=False)+'\n')
    path = experiment / 'evidence/persistent_execution_resource.csv'
    is_new = not path.exists()
    with path.open('a', newline='') as stream:
        writer = csv.writer(stream)
        if is_new:
            writer.writerow(('checked_at','job','gpu_index','uuid','used_mib','total_mib','utilization_percent'))
        for line in resources.strip().splitlines():
            writer.writerow((now.isoformat(), args.job, *[part.strip() for part in line.split(',')]))
    print(json.dumps({k:v for k,v in row.items() if k not in ('services','training')}, ensure_ascii=False))


if __name__ == '__main__':
    main()
