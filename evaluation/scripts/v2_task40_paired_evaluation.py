"""Workbench stage for the locked 12 V1/V2 physical restore comparisons."""
import argparse
import csv
import json
import os
from pathlib import Path
import signal
import socket
import subprocess
import sys
import time

from benchmark_dashboard import atomic_json
from task40_restore_worker import lease_sim_cache
from v2_lock_evaluation import digest


def simulator_environment(output):
    # Isaac's Vulkan/graphics device uses the physical index, unlike policy CUDA.
    gpu = int(os.environ.get('CUDA_VISIBLE_DEVICES', '0'))
    env = dict(os.environ, OMNIGIBSON_GPU_ID=str(gpu),
        OMNIGIBSON_APPDATA_PATH=str(output / 'appdata'),
        GROOT_EVAL_TRACE='1', GROOT_EVAL_PROCESS_STARTED_AT=str(time.perf_counter()))
    env.pop('CUDA_VISIBLE_DEVICES', None)
    return gpu, env


def simulator_command(root, item, output, seed_file, port, protocol):
    argv = [str(root / 'scripts/b1k_python.sh'), '-u', str(root / 'evaluation/scripts/task40_paired_sim.py'),
        '--output', str(output), '--anchor', str(root / item['checkpoint']), '--seeds', str(seed_file),
        '--behavior', '/nvme3/projects/GR00T/BEHAVIOR-1K-task-progress',
        '--groot', '/nvme3/projects/GR00T/Isaac-GR00T-n1d6-9b37aa1',
        '--port', str(port), '--policy-host', '127.0.0.1', '--cycles', str(protocol['closed_loop']['cycles']),
        '--phase', 'restore', '--restore-repeats', '1', '--remote-policy', '--trace-memory',
        '--collection-protocol', '--restore-checkpoint', str(root / item['checkpoint']),
        '--parent-trial', item['prefix'], '--rebuild-policy-history']
    if protocol['closed_loop'].get('require_exact_restore_rgb', False):
        argv.append('--require-exact-restore-rgb')
    return argv


def stop(process):
    if process is not None and process.poll() is None:
        process.terminate()
        try:
            process.wait(timeout=45)
        except subprocess.TimeoutExpired:
            process.kill()
            process.wait()


def invalid_parent_boundary(output, evidence_path):
    """Return an invalid marker only for the simulator's explicit boundary rejection."""
    failure_path = Path(output) / 'logs' / 'failure.txt'
    if not failure_path.is_file() or 'Restored initial truth or language disagrees with the verified parent boundary' not in failure_path.read_text():
        return None
    evidence = json.loads(Path(evidence_path).read_text())
    checks = evidence.get('truth_correspondence', {})
    fields_match = all(checks.get(name) is True for name in ('step', 'decision', 'Open', 'ToggledOn', 'Inside'))
    hinge_matches = isinstance(checks.get('hinge_abs_rad'), (int, float)) and checks['hinge_abs_rad'] <= 1e-5
    if evidence.get('language_equal') is True and fields_match and hinge_matches:
        return None
    return dict(valid=False, reason='INVALID_PARENT_BOUNDARY',
        rule='verified task-state/language boundary and hinge abs error <= 1e-5 rad',
        evidence=str(evidence_path), correspondence=checks,
        language_equal=evidence.get('language_equal'))


def load_resumable_results(path):
    """Reuse completed comparisons; archive invalid physical starts without retrying them."""
    if path is None:
        return []
    prior = json.loads(path.read_text())
    results = []
    seen = set()
    for source in prior.get('results', []):
        row = dict(source)
        key = (row['pair_id'], row['version'])
        if key in seen:
            raise ValueError(f'Duplicate resumed attempt: {key}')
        seen.add(key)
        if row['status'] == 'failed':
            correspondence_files = list((Path(row['output']) / 'metrics').glob('*/initial_correspondence.json'))
            if len(correspondence_files) != 1:
                raise ValueError(f"Cannot safely resume failed attempt {key}; inspect its simulator evidence")
            classification = invalid_parent_boundary(row['output'], correspondence_files[0])
            if classification is None:
                raise ValueError(f"Cannot safely resume failed attempt {key}; inspect its simulator failure reason")
            row['status'] = 'invalid'
            row['validity'] = classification
            row['resume_classification'] = 'existing physical-boundary rejection; not rerun'
        elif row['status'] not in ('complete', 'invalid'):
            raise ValueError(f"Cannot resume unresolved attempt {key} with status {row['status']}")
        results.append(row)
    return results


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('root', 'checkpoint', 'protocol', 'output'):
        parser.add_argument('--'+name, type=Path, required=True)
    parser.add_argument('--port', type=int, default=5598)
    parser.add_argument('--resume-status', type=Path)
    args = parser.parse_args()
    root, folder = args.root.resolve(), args.output.parent.resolve()
    protocol = json.loads(args.protocol.read_text())
    if args.checkpoint.name != protocol['v2_checkpoint_id'] or not (args.checkpoint / 'COMPLETE.json').is_file():
        raise ValueError('Paired evaluation requires the locked complete final V2 checkpoint')
    policy = simulator = None
    results = load_resumable_results(args.resume_status)

    def interrupted(signum, frame):
        stop(simulator)
        stop(policy)
        raise SystemExit(128+signum)

    signal.signal(signal.SIGTERM, interrupted)
    signal.signal(signal.SIGINT, interrupted)
    try:
        for version, checkpoint in [('v1', root / protocol['v1_checkpoint']), ('v2', args.checkpoint)]:
            policy_log = (folder / f'{version}_policy.log').open('a', buffering=1)
            policy = subprocess.Popen([sys.executable, str(root / 'evaluation/scripts/v2_task40_policy.py'),
                '--checkpoint', str(checkpoint), '--port', str(args.port)], cwd=root,
                stdout=policy_log, stderr=subprocess.STDOUT)
            deadline = time.monotonic()+300
            while time.monotonic() < deadline:
                if policy.poll() is not None:
                    raise RuntimeError(f'{version} policy exited before readiness; see {policy_log.name}')
                try:
                    with socket.create_connection(('127.0.0.1', args.port), timeout=2):
                        pass
                    break
                except OSError:
                    try:
                        policy.wait(timeout=1)
                    except subprocess.TimeoutExpired:
                        pass
            else:
                raise TimeoutError(f'{version} policy did not become ready')
            for item in protocol['pairs']:
                if any(r['pair_id'] == item['pair_id'] and r['version'] == version for r in results):
                    print(json.dumps(dict(pair=item['pair_id'], version=version, stage='already_recorded')), flush=True)
                    continue
                for path, expected in item['input_sha256'].items():
                    if digest(root / path) != expected:
                        raise ValueError('Locked paired input changed: '+path)
                output = folder / f"{item['pair_id']}_{version}"
                output.mkdir(exist_ok=False)
                for name in ('logs','metrics','videos','checkpoints'):
                    (output / name).mkdir()
                seed_file = output / 'seeds.json'
                atomic_json(seed_file, dict(trials=[dict(trial=1, policy_seed=int(item['seed']), instance=int(item['instance']))]))
                atomic_json(output / 'job.json', dict(kind='v2_paired_evaluation', checkpoint=str(checkpoint),
                    pair_id=item['pair_id'], instance=int(item['instance']), seed=int(item['seed']),
                    source=item, protocol_sha256=digest(args.protocol), policy_port=args.port))
                sim_gpu, env = simulator_environment(output)
                lease = lease_sim_cache(output / 'appdata', sim_gpu, 1)
                argv = simulator_command(root, item, output, seed_file, args.port, protocol)
                print(json.dumps(dict(pair=item['pair_id'], version=version, stage='restoring')), flush=True)
                try:
                    with (output / 'logs/simulator.log').open('a', buffering=1) as log:
                        simulator = subprocess.Popen(argv, cwd=root, env=env, stdout=log, stderr=subprocess.STDOUT)
                        with (output / 'resource.csv').open('x', newline='', buffering=1) as resource:
                            writer = csv.writer(resource)
                            writer.writerow(('timestamp','used_mib','free_mib','utilization_percent'))
                            while simulator.poll() is None:
                                usage = subprocess.check_output(['nvidia-smi','--id='+os.environ.get('CUDA_VISIBLE_DEVICES','0'),
                                    '--query-gpu=memory.used,memory.free,utilization.gpu','--format=csv,noheader,nounits'], text=True)
                                writer.writerow([time.time(), *map(int, usage.strip().split(','))])
                                try:
                                    simulator.wait(timeout=2)
                                except subprocess.TimeoutExpired:
                                    pass
                        code = simulator.wait()
                finally:
                    stop(simulator)
                    lease.close()
                validity = list((output / 'metrics').glob('*/restore_validity.json'))
                validity = json.loads(validity[0].read_text()) if validity else None
                trials = [json.loads(p.read_text()) for p in (output / 'metrics').glob('*/trial.json')]
                boundary_files = list((output / 'metrics').glob('*/initial_correspondence.json'))
                if validity is None and len(boundary_files) == 1:
                    validity = invalid_parent_boundary(output, boundary_files[0])
                row = dict(pair_id=item['pair_id'], version=version, checkpoint=str(checkpoint),
                    instance=int(item['instance']), seed=int(item['seed']), output=str(output), returncode=code,
                    status='invalid' if validity and not validity['valid'] else 'complete' if code == 0 and trials else 'failed',
                    validity=validity, trials=trials)
                results.append(row)
                atomic_json(folder / 'paired_status.json', dict(status='running', attempts_completed=len(results), results=results))
                print(json.dumps(dict(pair=item['pair_id'], version=version, status=row['status'])), flush=True)
                if row['status'] == 'failed':
                    raise RuntimeError(f"Unresolved paired runtime failure: {item['pair_id']} {version}; inspect simulator.log")
            stop(policy)
            policy_log.close()
        valid_pairs = [item['pair_id'] for item in protocol['pairs'] if all(
            any(r['pair_id']==item['pair_id'] and r['version']==v and r['status']=='complete' for r in results)
            for v in ('v1','v2'))]
        report = dict(passed=True, attempts_completed=len(results), valid_pairs=len(valid_pairs),
            valid_pair_ids=valid_pairs, invalid_attempts=sum(r['status']=='invalid' for r in results),
            acceptance='execution complete; invalid restores excluded from scientific comparisons', results=results)
        atomic_json(args.output, report)
        atomic_json(folder / 'paired_status.json', dict(status='complete', **report))
        print(json.dumps(dict(pairs_completed=len(valid_pairs), invalid_attempts=report['invalid_attempts'])), flush=True)
    finally:
        stop(simulator)
        stop(policy)


if __name__ == '__main__':
    main()
