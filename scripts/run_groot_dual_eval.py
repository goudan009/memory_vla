#!/usr/bin/env python3
"""One independent BEHAVIOR process per 4090, with disjoint instance assignments."""

import argparse
import json
import os
from pathlib import Path
import re
import signal
import subprocess
import sys
import time

from groot_eval_results import write_json


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--task', type=int, choices=[21, 22, 26], default=21)
    parser.add_argument('--instances-gpu0', type=int, nargs='+', default=[0])
    parser.add_argument('--instances-gpu1', type=int, nargs='+', default=[1])
    parser.add_argument('--episodes-per-gpu', type=int, default=1)
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--policy-host', default='172.28.62.17')
    parser.add_argument('--policy-port', type=int, default=5556)
    parser.add_argument('--run-tag', default='dual_' + time.strftime('%Y%m%d_%H%M%S'))
    parser.add_argument('--profile-cycles', type=int, choices=range(100, 201), default=150)
    parser.add_argument('--full-episodes', action='store_true', help='run through native episode termination instead of stopping at the diagnostic window')
    parser.add_argument('--disable-viewer', action='store_true')
    parser.add_argument('--appdata-gpu0', type=Path)
    parser.add_argument('--appdata-gpu1', type=Path)
    parser.add_argument('--dry-run', action='store_true')
    args = parser.parse_args()
    partitions = [args.instances_gpu0, args.instances_gpu1]
    if any(len(set(ids)) != len(ids) or any(i not in range(10) for i in ids) for ids in partitions):
        parser.error('each GPU must receive unique B1K instance IDs in 0-9')
    if set(partitions[0]) & set(partitions[1]):
        parser.error('GPU0 and GPU1 instance assignments must be disjoint')
    if args.episodes_per_gpu < 1 or any(args.episodes_per_gpu > len(ids) for ids in partitions):
        parser.error('episodes-per-gpu must fit each distinct instance assignment')
    if not args.full_episodes and args.episodes_per_gpu != 1:
        parser.error('diagnostic mode requires one episode per GPU')
    if not re.fullmatch(r'[A-Za-z0-9_.-]+', args.run_tag):
        parser.error('run-tag must use letters, digits, dot, underscore or hyphen')
    root = Path(os.environ.get('PROJECT_ROOT', Path(__file__).resolve().parents[1]))
    args.appdata_gpu0 = (args.appdata_gpu0 or root / 'evaluation/groot_runtime_gpu0/appdata').resolve()
    args.appdata_gpu1 = (args.appdata_gpu1 or root / 'evaluation/groot_runtime_gpu1/appdata').resolve()
    if args.appdata_gpu0 == args.appdata_gpu1:
        parser.error('two live simulators cannot share an AppData directory')
    return args


def main():
    args = parse_args()
    root = Path(os.environ.get('PROJECT_ROOT', Path(__file__).resolve().parents[1]))
    output = root / 'evaluation' / ('groot_dual_' + args.run_tag)
    barrier = output / 'barrier'
    commands = []
    for gpu, instances, appdata in [(0, args.instances_gpu0, args.appdata_gpu0), (1, args.instances_gpu1, args.appdata_gpu1)]:
        command = ['bash', str(root / 'scripts/run_groot_priority_eval.sh'), '--preset', 'official-eval' if args.full_episodes else 'fast-debug',
                   '--tasks', str(args.task), '--gpu-index', str(gpu), '--n-envs', '1', '--episodes', str(args.episodes_per_gpu),
                   '--instance-ids', *map(str, instances), '--seed', str(args.seed), '--policy-host', args.policy_host,
                   '--policy-port', str(args.policy_port), '--run-tag', args.run_tag]
        if not args.full_episodes:
            command += ['--profile-cycles', str(args.profile_cycles), '--diagnostic-only', '--audit-semantics']
        else:
            command += ['--monitor-resources']
        if args.disable_viewer:
            command += ['--disable-viewer']
        if appdata:
            command += ['--appdata-path', str(appdata.resolve())]
        commands.append(command)
    plan = {'commands': commands, 'instances': [args.instances_gpu0, args.instances_gpu1],
            'seed': args.seed, 'shared_policy': f'{args.policy_host}:{args.policy_port}',
            'diagnostic_only': not args.full_episodes, 'output': str(output)}
    if args.dry_run:
        print(json.dumps(plan, indent=2))
        return 0
    output.mkdir(parents=True, exist_ok=False)
    barrier.mkdir()
    write_json(output / 'plan.json', plan)
    print(f'DUAL_START output={output}', flush=True)
    # Probe uses independent live REQ sockets; no server restart or model input changes.
    probe = subprocess.run([str(root / 'scripts/b1k_python.sh'), str(root / 'scripts/groot_policy_route_probe.py'),
                            '--host', args.policy_host, '--port', str(args.policy_port), '--output', str(output / 'routing_probe.json')],
                           capture_output=True, text=True)
    if probe.returncode:
        print(probe.stdout + probe.stderr, flush=True)
        return probe.returncode
    print('LIVE_REQ_ROUTING_OK clients=2', flush=True)
    children, logs = [], []
    interrupted = False

    def stop(_signal=None, _frame=None):
        nonlocal interrupted
        interrupted = True
        for child in children:
            if child.poll() is None:
                child.terminate()  # Each queue forwards to only its own simulator group.

    signal.signal(signal.SIGINT, stop)
    signal.signal(signal.SIGTERM, stop)
    started = time.time()
    try:
        for gpu, command in enumerate(commands):
            log = (output / f'gpu{gpu}.launcher.log').open('w', buffering=1)
            logs.append(log)
            environment = os.environ.copy()
            environment['GROOT_EVAL_BARRIER_DIR'] = str(barrier)
            for flag in ('GROOT_EVAL_DETAIL_PROFILE', 'GROOT_EVAL_SIM_PROFILE', 'GROOT_EVAL_CAMERA_COUNTERS'):
                environment[flag] = '0' if args.full_episodes else '1'
            child = subprocess.Popen(command, stdout=log, stderr=subprocess.STDOUT, env=environment)
            children.append(child)
            print(f'GPU{gpu}_QUEUE pid={child.pid} instances={plan["instances"][gpu]}', flush=True)
        released = False
        actual_instances = None
        while any(child.poll() is None for child in children):
            if not released and not interrupted and all((barrier / f'gpu{gpu}.ready.json').exists() for gpu in (0, 1)):
                ready = [json.loads((barrier / f'gpu{gpu}.ready.json').read_text()) for gpu in (0, 1)]
                actual_instances = [row['audit_config']['instance_id'] for row in ready]
                expected = [args.instances_gpu0[0], args.instances_gpu1[0]]
                if actual_instances != expected:
                    print(f'ACTUAL_INSTANCE_MISMATCH expected={expected} actual={actual_instances}', flush=True)
                    stop()
                else:
                    print(f'ACTUAL_INSTANCES_OK GPU0={actual_instances[0]} GPU1={actual_instances[1]}', flush=True)
                if not interrupted:
                    write_json(barrier / 'release.json', {'released_unix': time.time()})
                    released = True
                    print('BOTH_ENVS_READY releasing synchronized diagnostic windows', flush=True)
            if not released and any(child.poll() is not None for child in children):
                stop()
            time.sleep(0.25)
        result = {'started_unix': started, 'ended_unix': time.time(), 'released': released, 'actual_instances': actual_instances,
                  'interrupted': interrupted, 'exit_codes': [child.returncode for child in children], 'workers': []}
        for gpu in (0, 1):
            matches = list((root / 'evaluation').glob(f'groot_n1d6_task{args.task:03d}_*_{args.run_tag}_gpu{gpu}'))
            worker = {'gpu': gpu, 'instances': plan['instances'][gpu]}
            if matches:
                worker['output'] = str(matches[0])
                for key, filename in [('profile', 'diagnostic_profile.json'), ('results', 'results.json')]:
                    path = matches[0] / filename
                    if path.exists():
                        worker[key] = json.loads(path.read_text())
            result['workers'].append(worker)
        profiles = [worker.get('profile') for worker in result['workers']]
        if all(profiles) and not args.full_episodes:
            beginning = min(p['ab_audit']['wall_start_unix'] for p in profiles)
            ending = max(p['ab_audit']['wall_end_unix'] for p in profiles)
            result['aggregate_actions_per_s'] = sum(p['sim_step_calls'] for p in profiles) / (ending - beginning)
            result['diagnostic_wall_s'] = ending - beginning
            result['valid_episodes_per_hour'] = None
        records = [worker.get('results', {}) for worker in result['workers']]
        if args.full_episodes:
            result['valid_episodes_per_hour'] = sum(r.get('completed_episodes', 0) for r in records) * 3600 / (result['ended_unix'] - started)
        expected_status = 'complete' if args.full_episodes else 'diagnostic_complete'
        window_valid = args.full_episodes or all(
            p and p['cycles'] == args.profile_cycles and p['sim_step_calls'] == args.profile_cycles * 8
            and p['ab_audit']['all_finite'] and p['ab_audit']['physx_crash_cycles'] == 0
            for p in profiles
        )
        result['ok'] = not interrupted and window_valid and all(r.get('status') == expected_status for r in records)
        write_json(output / 'summary.json', result)
        print(f'DUAL_FINISHED ok={result["ok"]} aggregate_actions_per_s={result.get("aggregate_actions_per_s")} summary={output / "summary.json"}', flush=True)
        return 0 if result['ok'] else 1
    finally:
        if any(child.poll() is None for child in children):
            stop()
        for child in children:
            child.wait()
        for log in logs:
            log.close()


if __name__ == '__main__':
    sys.exit(main())
