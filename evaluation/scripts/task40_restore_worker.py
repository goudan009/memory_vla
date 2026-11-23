#!/usr/bin/env python3
"""Run one Task40 checkpoint restore through the evaluation workbench."""

import argparse
import csv
import fcntl
import json
import os
from pathlib import Path
import signal
import socket
import subprocess
import sys
import time


children = []


def lease_sim_cache(appdata, gpu, slots):
    """Reuse derived Isaac caches, keeping Kit preferences and logs job-local."""
    cache_root = appdata.parents[2] / 'cache'
    cache_root.mkdir(parents=True, exist_ok=True)
    for index in range(slots):
        slot = cache_root / f'isaac-gpu{gpu}-slot{index}'
        lease = slot.with_suffix('.lock').open('a')
        try:
            fcntl.flock(lease, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            lease.close()
            continue
        try:
            for name in ('global', 'local'):
                target = slot / name
                target.mkdir(parents=True, exist_ok=True)
                link = appdata / name / 'cache'
                link.parent.mkdir(parents=True, exist_ok=True)
                link.symlink_to(target, target_is_directory=True)
            return lease
        except BaseException:
            lease.close()
            raise
    raise RuntimeError(f'No idle Isaac cache slot for GPU {gpu}')


def stop_children(signum, frame):
    for child in children:
        if child.poll() is None:
            child.terminate()
    deadline = time.monotonic() + 45
    for child in children:
        if child.poll() is None:
            try:
                child.wait(timeout=max(0.1, deadline - time.monotonic()))
            except subprocess.TimeoutExpired:
                child.kill()
    raise SystemExit(128 + signum)


def main(args):
    root = args.root.resolve()
    output = args.output.resolve()
    anchor = args.anchor.resolve()
    weights = args.weights.resolve()
    experiment = json.loads(args.experiment_plan) if args.experiment_plan else {}
    renderer_probe = experiment.get('phase') == 'renderer_probe'
    targeted = experiment.get('phase') in ('directed_restore', 'renderer_probe')
    controlled_reclosure = experiment.get('controlled_reclosure', False)
    full_task_eval = experiment.get('full_task_eval', False)
    if type(full_task_eval) is not bool or (full_task_eval and (
            targeted or controlled_reclosure or not args.collect_instances or args.restore_seeds
            or args.memory_mode != 'A_normal' or not (
                experiment.get('official_policy') or experiment.get('policy_identity', {}).get('task40_trace')))):
        raise ValueError('Full task evaluation requires fresh A_normal collection with Task40 trace')
    if type(controlled_reclosure) is not bool:
        raise ValueError('controlled_reclosure must be a boolean')
    if controlled_reclosure and (
            targeted or experiment.get('kind') != 'task40_collect' or experiment.get('sim_slots', 1) not in (1, 2)
            or not args.collect_instances or args.restore_seeds or args.restore_repeats != 1
            or args.memory_mode != 'A_normal' or not experiment.get('policy_identity', {}).get('task40_trace')):
        raise ValueError('Controlled reclosure requires fresh A_normal collection with Task40 trace and slot 1 or 2')
    if targeted:
        if (args.collect_instances or args.restore_seeds or args.restore_repeats not in (1, 2)
                or args.memory_mode != 'A_normal'):
            raise ValueError('Targeted restore requires one A_normal checkpoint and at most two repeats')
        checkpoint = (root / experiment['restore_checkpoint']).resolve(strict=True)
        parent_trial = (root / experiment['lineage']['prefix']).resolve(strict=True)
        if any(output.is_relative_to(path) for path in (checkpoint, parent_trial)):
            raise ValueError('Targeted restore output must be outside the archived inputs')
        parent = json.loads((parent_trial / 'trial.json').read_text())
    output.mkdir(parents=True, exist_ok=False)
    (output / 'logs').mkdir()
    (output / 'metrics').mkdir()
    (output / 'videos').mkdir()
    (output / 'checkpoints').mkdir()
    seeds = output / 'seeds.json'
    instances = args.collect_instances or [0]
    trials = [{'trial': index + 1, 'policy_seed': args.seed + index, 'instance': instance}
              for index, instance in enumerate(instances * args.collect_repetitions)]
    if args.restore_seeds:
        trials = [{'trial': index + 1, 'policy_seed': seed} for index, seed in enumerate(args.restore_seeds)]
    if targeted:
        trials = [{'trial': 1, 'policy_seed': parent['seed'], 'instance': parent['instance'],
                   'label_seed': args.seed}]
    seeds.write_text(json.dumps({'trials': trials}, indent=2) + '\n')
    phase = 'renderer_probe' if renderer_probe else 'collect' if args.collect_instances else 'restore'
    plan = {'kind': 'task40_restore', 'task_id': 40, 'version': 'Task40',
            'group': 'checkpoint_restore', 'output': str(output),
            'instances': {str(args.sim_gpu): [0]},
            'restore_anchor': str(anchor), 'weights': str(weights), 'seed': args.seed,
            'cycles': args.cycles, 'memory_mode': args.memory_mode,
            'policy_port': args.port, 'phase': phase, 'restore_repeats': args.restore_repeats,
            'renderer_audit': args.renderer_audit, 'renderer_aa': args.renderer_aa,
            'policy_host': args.policy_host, 'policy_device': 'remote', 'trials': trials}
    if args.experiment_plan:
        plan.update(experiment)
        if controlled_reclosure:
            plan.update(phase='controlled_reclosure', source='controlled')
        if targeted:
            plan.update(restore_checkpoint=str(checkpoint), restore_anchor=str(checkpoint),
                        lineage={**experiment['lineage'], 'prefix': str(parent_trial)},
                        source='directed_restore', seed=parent['seed'], requested_seed=args.seed,
                        restore_repeats=args.restore_repeats, renderer_audit=args.renderer_audit,
                        renderer_aa=args.renderer_aa,
                        instances={str(args.sim_gpu): [parent['instance']]}, trials=trials)
        (output / 'job.json').write_text(json.dumps(plan, ensure_ascii=False, indent=2) + '\n')
    print(json.dumps(plan, ensure_ascii=False), flush=True)

    signal.signal(signal.SIGTERM, stop_children)
    signal.signal(signal.SIGINT, stop_children)

    task_scripts = Path(os.environ.get('GROOT_TASK40_RUNNER', root / 'evaluation/scripts'))
    sim_script = (root / 'evaluation/scripts/task40_paired_sim.py' if args.experiment_plan
                  else task_scripts / 'paired_sim_diagnostic.py')
    omnigibson = Path(os.environ.get('OMNIGIBSON_ROOT',
        '/nvme3/projects/GR00T/BEHAVIOR-1K-task-progress/OmniGibson'))
    bddl = Path(os.environ.get('BDDL_ROOT', omnigibson.parent / 'bddl'))
    groot = Path(os.environ.get('GROOT_REPO', '/nvme3/projects/GR00T/Isaac-GR00T-n1d6-9b37aa1'))
    if not sim_script.is_file():
        raise FileNotFoundError('Task40 restore runtime or diagnostic scripts are missing')

    sim_log = None
    cache_lease = None
    try:
        if not renderer_probe:
            deadline = time.monotonic() + args.startup_timeout
            while time.monotonic() < deadline:
                try:
                    with socket.create_connection((args.policy_host, args.port), timeout=2):
                        break
                except OSError:
                    time.sleep(1)
            else:
                raise TimeoutError(f'Remote policy {args.policy_host}:{args.port} is unavailable')

        sim_env = os.environ.copy()
        sim_env.pop('CUDA_VISIBLE_DEVICES', None)
        if sim_env.get('BEHAVIOR_DATA'):
            sim_env['OMNIGIBSON_DATA_PATH'] = sim_env['BEHAVIOR_DATA']
        else:
            sim_env.pop('OMNIGIBSON_DATA_PATH', None)
        sim_env.update({'OMNIGIBSON_GPU_ID': str(args.sim_gpu), 'OMNIGIBSON_HEADLESS': 'true',
                        'PYTHONUNBUFFERED': '1', 'OMNIGIBSON_ROOT': str(omnigibson),
                        'BDDL_ROOT': str(bddl)})
        if args.experiment_plan:
            sim_env.update(GROOT_EVAL_TRACE='1', GROOT_EVAL_PROCESS_STARTED_AT=str(time.perf_counter()))
        if plan.get('group') in ('experiment_11', 'experiment_12') and plan.get('kind') == 'task40_collect':
            appdata = output / 'appdata'
            cache_lease = lease_sim_cache(appdata, args.sim_gpu, plan.get('sim_slots', 1))
            sim_env['OMNIGIBSON_APPDATA_PATH'] = str(appdata)
            print('ISAAC_CACHE_LEASE', cache_lease.name, flush=True)
        elif targeted or args.restore_repeats > 1 or args.restore_seeds or args.collect_instances:
            sim_env.setdefault('OMNIGIBSON_APPDATA_PATH', str(output / 'appdata'))
        sim_log = (output / 'logs' / 'simulator.log').open('ab', buffering=0)
        sim_command = ['bash', str(root / 'scripts' / 'b1k_python.sh'), '-u', str(sim_script),
            '--output', str(output), '--anchor', str(anchor), '--seeds', str(seeds),
            '--behavior', str(omnigibson.parent), '--groot', str(groot), '--port', str(args.port),
            '--cycles', str(args.cycles), '--phase', phase, '--memory-mode', args.memory_mode,
            '--policy-host', args.policy_host]
        if not renderer_probe:
            sim_command.append('--remote-policy')
        if args.collect_instances:
            sim_command += ['--instances', *map(str, instances)]
        if args.experiment_plan:
            sim_command.append('--collection-protocol')
            if not plan.get('official_policy') and (targeted or plan.get('policy_identity', {}).get('task40_trace')):
                sim_command.append('--trace-memory')
            if plan.get('rebuild_policy_history'):
                sim_command.append('--rebuild-policy-history')
            if plan.get('official_policy'):
                sim_command.append('--official-policy')
            if plan.get('stop_at_first_open'):
                sim_command.append('--stop-at-first-open')
        if controlled_reclosure:
            sim_command.append('--controlled-reclosure')
        if full_task_eval:
            sim_command.append('--full-task-eval')
        if plan.get('capture_approach'):
            sim_command.append('--capture-approach')
        if targeted:
            sim_command += ['--restore-checkpoint', str(checkpoint), '--parent-trial', str(parent_trial)]
        if renderer_probe:
            sim_command += ['--renderer-probe-actions', str(parent_trial), '--compare-accumulation-reset']
            if 'renderer_gi_history' in plan:
                sim_command += ['--renderer-gi-history', str(plan['renderer_gi_history'])]
            if plan.get('renderer_disable_indirect_diffuse'):
                sim_command.append('--renderer-disable-indirect-diffuse')
        if args.restore_repeats > 1:
            sim_command += ['--restore-repeats', str(args.restore_repeats)]
        if args.renderer_audit:
            sim_command += ['--renderer-audit']
        if args.renderer_aa is not None:
            sim_command += ['--renderer-aa', str(args.renderer_aa)]
        if args.capture_failed_attempt or args.restore_seeds:
            sim_command += ['--capture-failed-attempt']
        if args.observe_open_chunks:
            sim_command += ['--observe-open-chunks', str(args.observe_open_chunks)]
        if args.experiment_plan:
            usage = subprocess.check_output(['nvidia-smi', '-i', str(args.sim_gpu),
                '--query-gpu=memory.free', '--format=csv,noheader,nounits'], text=True)
            required_free = {1: 10240, 2: 24576}[plan.get('sim_slots', 1)]
            if int(usage.strip()) <= required_free:
                raise RuntimeError(f"{plan.get('sim_slots', 1)} 环境模式要求剩余显存大于 {required_free} MiB")
        started = time.monotonic()
        simulator = subprocess.Popen(sim_command,
            cwd=root, env=sim_env, stdin=subprocess.DEVNULL, stdout=sim_log,
            stderr=subprocess.STDOUT, start_new_session=True,
            pass_fds=(cache_lease.fileno(),) if cache_lease is not None else ())
        children.append(simulator)
        if args.experiment_plan:
            with (output / 'metrics' / 'gpu.csv').open('x', newline='', buffering=1) as stream:
                writer = csv.writer(stream)
                writer.writerow(('elapsed_seconds', 'used_mib', 'free_mib', 'utilization_percent'))
                while simulator.poll() is None:
                    usage = subprocess.check_output(['nvidia-smi', '-i', str(args.sim_gpu),
                        '--query-gpu=memory.used,memory.free,utilization.gpu', '--format=csv,noheader,nounits'],
                        text=True, timeout=5)
                    writer.writerow([round(time.monotonic()-started, 3), *map(int, usage.strip().split(','))])
                    try:
                        simulator.wait(timeout=2)
                    except subprocess.TimeoutExpired:
                        pass
        code = simulator.wait()
        if code:
            raise subprocess.CalledProcessError(code, simulator.args)
        failure = output / 'logs' / 'failure.txt'
        if failure.exists():
            raise RuntimeError('Simulator reported a failed restore or trial:\n' +
                               failure.read_text(errors='replace')[-6000:])
        if renderer_probe:
            audits = ('metrics/fixed_action_audit.json', 'metrics/accumulation_reset_audit.json')
            for audit in audits:
                comparison = json.loads((output / audit).read_text())
                if comparison['shared_steps'] != args.cycles * 8:
                    raise RuntimeError(f'Fixed-action comparison did not complete all planned steps: {audit}')
            (output / 'results.json').write_text(json.dumps({
                'status': 'completed', 'configuration': plan, 'fixed_action_audits': audits,
                'finished_at': time.strftime('%Y-%m-%d %H:%M:%S'),
                'elapsed_seconds': time.monotonic() - started}, ensure_ascii=False, indent=2) + '\n')
            return
        names = ([f"episode_i{row['instance']}_s{row['policy_seed']}_A_normal" for row in trials]
                 if args.collect_instances else [
                     f"restore_{row.get('label_seed', row['policy_seed'])}_{args.memory_mode}" for row in trials])
        if args.restore_repeats > 1:
            names = [f'restore_{args.seed}_repeat{index:02d}_{args.memory_mode}'
                     for index in range(1, args.restore_repeats + 1)]
        completed = []
        for name in names:
            trial = output / 'metrics' / name / 'trial.json'
            if not trial.is_file():
                raise RuntimeError(f'Simulator exited without a completed trial record: {trial}')
            completed.append(json.loads(trial.read_text()))
        if args.experiment_plan:
            artifacts = {}
            if controlled_reclosure:
                markers = []
                for trial in completed:
                    summary = trial['controlled_reclosure']
                    marker_file = output / summary['marker']
                    marker = json.loads(marker_file.read_text())
                    if ((marker['branch'], marker['status'], marker['valid']) !=
                            (trial['branch'], summary['status'], summary['valid'])):
                        raise RuntimeError(f'Controlled reclosure marker disagrees with trial: {marker_file}')
                    markers.append(summary)
                artifacts['controlled_reclosure'] = {'valid_samples': sum(item['valid'] for item in markers),
                                                     'markers': markers}
            if targeted and args.restore_repeats == 2:
                audit = 'metrics/repeat_audit.json'
                if not (output / audit).is_file():
                    raise RuntimeError('Targeted repeat finished without a pair audit')
                artifacts['repeat_audit'] = audit
            (output / 'results.json').write_text(json.dumps({
                'status': 'completed', 'configuration': plan, 'trials': completed, **artifacts,
                'finished_at': time.strftime('%Y-%m-%d %H:%M:%S'),
                'elapsed_seconds': time.monotonic() - started}, ensure_ascii=False, indent=2) + '\n')
        print('TASK40_RESTORE_COMPLETE', output, flush=True)
    finally:
        for child in reversed(children):
            if child.poll() is None:
                child.terminate()
        for child in reversed(children):
            if child.poll() is None:
                try:
                    child.wait(timeout=15)
                except subprocess.TimeoutExpired:
                    child.kill()
                    child.wait()
        if sim_log is not None:
            sim_log.close()
        if cache_lease is not None:
            cache_lease.close()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--anchor', type=Path, required=True)
    parser.add_argument('--weights', type=Path, required=True)
    parser.add_argument('--sim-gpu', type=int, choices=(0, 1), required=True)
    parser.add_argument('--policy-host', required=True)
    parser.add_argument('--seed', type=int, required=True)
    parser.add_argument('--cycles', type=int, required=True)
    parser.add_argument('--memory-mode', choices=('A_normal', 'B_empty'), default='A_normal')
    parser.add_argument('--port', type=int, required=True)
    parser.add_argument('--startup-timeout', type=int, default=600)
    parser.add_argument('--collect-instances', type=int, nargs='+', choices=range(10))
    parser.add_argument('--collect-repetitions', type=int, default=1)
    parser.add_argument('--restore-repeats', type=int, choices=range(1, 4), default=1)
    parser.add_argument('--renderer-audit', action='store_true')
    parser.add_argument('--renderer-aa', type=int, choices=(0, 3))
    parser.add_argument('--restore-seeds', type=int, nargs='+')
    parser.add_argument('--capture-failed-attempt', action='store_true',
                        help='save a restore checkpoint when an episode records a real closed-door attempt')
    parser.add_argument('--observe-open-chunks', type=int, choices=range(13), default=0)
    parser.add_argument('--experiment-plan')
    args = parser.parse_args()
    max_cycles = 300 if args.collect_instances else 60
    if not 1 <= args.collect_repetitions <= 10 // len(args.collect_instances or [0]):
        parser.error('A collection worker can run at most ten trajectories')
    if args.collect_repetitions != 1 and not args.collect_instances:
        parser.error('--collect-repetitions requires --collect-instances')
    if not 1 <= args.cycles <= max_cycles:
        parser.error(f'--cycles must be between 1 and {max_cycles} for this mode')
    if args.collect_instances and args.restore_repeats > 1:
        parser.error('--restore-repeats applies only to checkpoint restores')
    if args.renderer_audit and args.restore_repeats < 2:
        parser.error('--renderer-audit requires repeated checkpoint restores')
    if args.renderer_aa is not None and not args.renderer_audit:
        parser.error('--renderer-aa requires --renderer-audit')
    if args.restore_seeds and (args.collect_instances or args.restore_repeats > 1 or args.memory_mode != 'A_normal'):
        parser.error('--restore-seeds supports independent normal-memory checkpoint restores only')
    main(args)
