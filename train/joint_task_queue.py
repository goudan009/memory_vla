"""Prepare a joint dataset and run two matched jobs after an existing queue.

This is an unattended queue controller, not a second training entry point.
All optimization uses train/run_groot_train.sh from an immutable source copy.
"""

import argparse
from collections import Counter, defaultdict
from datetime import datetime
import hashlib
import json
import os
from pathlib import Path
import shlex
import shutil
import subprocess
import time

from run_identity import NAMING_SCHEMA, reserve_training, validate_run_namespace


def read(path, default=None):
    return json.loads(Path(path).read_text()) if Path(path).exists() else default


def write(path, data):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + '.pending')
    tmp.write_text(json.dumps(data, indent=2, ensure_ascii=False) + '\n')
    tmp.replace(path)


def prepare_dataset(config):
    target = Path(config['dataset'])
    ready = read(target/'preparation/ready.json')
    if ready:
        assert ready['sources'] == config['datasets']
        return read(target/'preparation/data_split.json')
    assert not target.exists(), 'Unfinished dataset exists; inspect it before retrying'
    rows, episode_tasks, splits, train, validation = [], {}, {}, [], []
    reference, total_frames = None, 0
    for source in config['datasets']:
        root, task = Path(source['dataset']), source['task']
        split = read(source['split'])
        assert split['task'] == task and split['seed'] == config['seed']
        assert len(split['train']) == 180 and len(split['validation']) == 20
        assert not set(split['train']) & set(split['validation'])
        episodes = [json.loads(line) for line in (root/'meta/episodes.jsonl').read_text().splitlines()]
        assert {e['episode_index'] for e in episodes} == set(split['train'] + split['validation'])
        info = read(root/'meta/info.json')
        assert info['fps'] == 30 and info['chunks_size'] == 10000
        for episode in episodes:
            eid = episode['episode_index']
            assert eid not in episode_tasks, 'Overlapping Episode IDs need namespacing before merging'
            assert episode['length'] >= 544, 'Need two complete windows for wrong-history validation'
            episode_tasks[eid] = task
        if reference is None:
            reference = root
            merged_info = dict(info)
        for name in ('modality.json', 'stats.json', 'relative_stats.json'):
            assert read(root/'meta'/name) == read(reference/'meta'/name), name
        assert (root/'meta/tasks.jsonl').read_bytes() == (reference/'meta/tasks.jsonl').read_bytes()
        for field in ('data_path', 'video_path', 'features', 'fps', 'chunks_size'):
            assert info.get(field) == merged_info.get(field), field
        rows.extend(episodes)
        train.extend(split['train']); validation.extend(split['validation'])
        total_frames += sum(e['length'] for e in episodes)
        splits[str(task)] = split
    split = dict(tasks=[s['task'] for s in config['datasets']], seed=config['seed'],
        train=train, validation=validation, episode_tasks=episode_tasks, per_task=splits,
        sampling_stride=8, gradient_window=32, task_sampling='equal task cycle',
        normalization='unchanged official checkpoint statistics')
    from v1_groot_memoryvla.data.episode_split import validate_split
    validate_split(split)
    (target/'meta').mkdir(parents=True)
    for name in ('modality.json', 'stats.json', 'relative_stats.json', 'tasks.jsonl'):
        shutil.copyfile(reference/'meta'/name, target/'meta'/name)
    for source in config['datasets']:
        root = Path(source['dataset'])
        for kind in ('data', 'videos'):
            for chunk in sorted((root/kind).iterdir()):
                destination = target/kind/chunk.name
                assert not destination.exists(), 'Dataset chunk collision'
                destination.parent.mkdir(parents=True, exist_ok=True)
                destination.symlink_to(chunk.resolve(), target_is_directory=True)
    merged_info.update(total_episodes=len(rows), total_frames=total_frames,
        total_videos=3*len(rows), total_chunks=len(list((target/'data').iterdir())),
        splits=dict(train_episode_ids=train, validation_episode_ids=validation))
    write(target/'meta/info.json', merged_info)
    (target/'meta/episodes.jsonl').write_text(''.join(json.dumps(e)+'\n' for e in rows))
    write(target/'preparation/data_split.json', split)
    write(target/'preparation/ready.json', dict(status='completed', sources=config['datasets'],
        train_episodes=len(train), validation_episodes=len(validation), total_frames=total_frames))
    return split


def dependency_state(config):
    directory = Path(config['after_queue'])
    state = read(directory/'status.json', {})
    if state.get('status') == 'failed':
        raise RuntimeError('Dependency queue failed: '+str(state.get('error')))
    rows = []
    migrations = read(directory.parent/'name_migration/status.json', {}).get('migrations', {})
    for job in state.get('runs', []):
        root = Path(job['output'])
        if not root.exists():
            root = root.parent/migrations.get(root.name, root.name)
        split = read(root/'config/data_split.json', {})
        train = read(root/job['arm']/'status.json', {})
        validation = read(root/job['arm']/'validation_status.json', {})
        aggregate = read(root/'status.json', {})
        if aggregate.get('status') == 'failed':
            raise RuntimeError('Dependency run failed: '+str(root))
        complete = (aggregate.get('status') == 'completed' and train.get('training_complete')
            and train.get('step') == 3000 and validation.get('status') == 'completed'
            and 3000 in validation.get('validated_steps', [])
            and (root/job['arm']/'checkpoints/step-003000/config.json').is_file())
        rows.append(dict(task=split.get('task'), arm=job['arm'], output=str(root),
                         step=train.get('step', 0), complete=bool(complete)))
    expected = {(t, a) for t in (34, 40) for a in ('baseline', 'memory_v1')}
    complete = (state.get('status') == 'completed' and len(rows) == 4 and
        {(r['task'], r['arm']) for r in rows} == expected and all(r['complete'] for r in rows))
    return complete, rows


def process_identity(pid):
    try:
        # /proc comm may contain spaces: fields after the closing ')' start at #3.
        return Path(f'/proc/{pid}/stat').read_text().rsplit(')', 1)[1].split()[19]
    except FileNotFoundError:
        return None


def compute_pids():
    data = subprocess.check_output(['nvidia-smi', '--query-compute-apps=pid',
                                    '--format=csv,noheader,nounits'], text=True)
    return sorted(set(int(line) for line in data.splitlines() if line.strip()))


def verify_source(config):
    source = Path(config['source'])
    manifest = read(source/'web_source_manifest.json')
    assert manifest['digest'] == config['source_digest']
    actual = {name: hashlib.sha256((source/name).read_bytes()).hexdigest()
              for name in manifest['files']}
    assert actual == manifest['files'], 'Immutable queue source changed'
    checkpoint = Path(config['checkpoint'])
    assert checkpoint.name == 'GR00T-N1.6-BEHAVIOR1k'
    assert read(checkpoint/'config.json'), 'Official checkpoint missing'


def command(config, job, allowed):
    source = Path(job['source'])
    return ['bash', str(source/'train/run_groot_train.sh'), '--source-dir', str(source),
        '--python-path', config['python'], '--dataset', config['dataset'],
        '--split', str(Path(job['output'])/'config/data_split.json'),
        '--checkpoint', config['checkpoint'], '--output', job['output'], '--arm', job['arm'],
        '--gpus', ','.join(map(str, job['gpus'])), '--async-validation-gpu', str(job['validation_gpu']),
        '--steps', str(config['steps']), '--global-windows-per-step', str(config['global_windows_per_step']),
        '--microbatch', str(config['microbatch']), '--learning-rate', str(config['learning_rate']),
        '--seed', str(config['seed']), '--validation-interval', '3000', '--disable-early-stopping',
        '--allow-service-pids', *map(str, allowed)]


def reserve(config):
    version, test_id, experiment_id = validate_run_namespace(
        config.get('version'), config.get('test_id'), config.get('experiment_id'))
    if version != 'V_1':
        raise ValueError('joint_task_queue仅支持V_1，其他方法请使用其登记的工作台入口')
    verify_source(config)
    split = prepare_dataset(config)
    jobs = []
    task_label = '_'.join(map(str, split['tasks']))
    for arm, method, gpus, validator in [('baseline', 'baseline', [0, 1], 0),
                                         ('memory_v1', 'memory', [2, 3], 3)]:
        name, number = reserve_training(config['training_root'], config['deployments'], version,
                                        test_id=test_id, experiment_id=experiment_id)
        output, source = Path(config['training_root'])/name, Path(config['deployments'])/name
        shutil.copytree(config['source'], source)
        job = dict(run=name, training_id=name, train_number=number, arm=arm, method=method,
                   output=str(output), source=str(source), gpus=gpus, validation_gpu=validator)
        write(output/'config/data_split.json', split)
        write(output/'config/source_manifest.json', read(source/'web_source_manifest.json'))
        write(output/'config/web_launch.json', dict(job, version=version, test_id=test_id,
            experiment_id=experiment_id, tasks=split['tasks'],
            task=task_label, naming_schema=NAMING_SCHEMA, seed=config['seed'], steps=config['steps'],
            dataset=config['dataset'], checkpoint=config['checkpoint'],
            global_windows_per_step=config['global_windows_per_step'],
            global_decisions=32*config['global_windows_per_step'],
            learning_rate=config['learning_rate'], microbatch=config['microbatch'],
            method_description='Three-task joint training from unchanged official B1K checkpoint'))
        write(output/'config/ddp_launch.json', dict(steps_each=config['steps'],
            commands={arm: dict(CUDA_VISIBLE_DEVICES=','.join(map(str, gpus)), argv=command(config, job, []))}))
        write(output/arm/'status.json', dict(status='queued', step=0, total_steps=config['steps'], training_complete=False))
        # Root status must be absent until launch(): it enforces fresh-run initialization.
        write(output/arm/'validation_queue/step-000000.json', dict(step=0, checkpoint=config['checkpoint']))
        (output/'logs').mkdir(exist_ok=True)
        jobs.append(job)
    config['jobs'] = jobs
    return jobs


def report(config):
    summary = {}
    initializations = []
    for job in config['jobs']:
        root = Path(job['output'])
        initial = read(root/job['arm']/'config/ddp.json')
        if initial:
            initializations.append(initial)
        groups, counts = defaultdict(list), Counter()
        path = root/job['arm']/'metrics/metrics.jsonl'
        evaluations = []
        for line in path.read_text().splitlines() if path.exists() else []:
            try:
                row = json.loads(line)
            except ValueError:
                continue  # Last record may still be in flight.
            if row['kind'] == 'train':
                groups[str(row['task'])].append(row)
                counts[str(row['task'])] += row['global_decisions']
            elif row['kind'] == 'eval':
                evaluations.append(row)
        values = {task: dict(updates=len(rows), decisions=counts[task],
            last_step=rows[-1]['step'], rolling_100_loss=sum(r['loss'] for r in rows[-100:])/len(rows[-100:]),
            mean_seconds=sum(r['seconds'] for r in rows[-100:])/len(rows[-100:]),
            peak_allocated_gib=max(r['peak_allocated_gib'] for r in rows),
            last_memory_gradients=rows[-1]['memory_gradients']) for task, rows in groups.items()}
        summary[job['run']] = dict(per_task=values,
            all_tasks_rolling_loss_below_target=(len(values) == 3 and all(
                v['updates'] >= 100 and v['rolling_100_loss'] <= config['loss_observation_target'] for v in values.values())),
            validation=evaluations[-1] if evaluations else None,
            status=read(root/job['arm']/'status.json', {}))
    if len(initializations) == 2:
        for key in ('checkpoint', 'dataset', 'seed', 'steps', 'learning_rate', 'microbatch',
                    'global_windows_per_step', 'frozen_eagle', 'action_head_sha256'):
            assert initializations[0][key] == initializations[1][key], key
    write(Path(config['queue_directory'])/'summary.json', dict(updated_at=datetime.now().isoformat(),
        target=config['loss_observation_target'], target_is_observation_not_guarantee=True, runs=summary))


def run(config_path):
    config = read(config_path)
    directory = Path(config['queue_directory'])
    assert not (directory/'status.json').exists(), 'Queue already submitted'
    verify_source(config)
    state = dict(status='waiting_for_four_runs', pid=os.getpid(), jobs=config['jobs'])
    write(directory/'status.json', state)
    print(json.dumps(dict(event='waiting_for_four_runs', pid=os.getpid(),
                          dependency_queue=config['after_queue'])), flush=True)
    children = []
    try:
        while True:
            done, dependencies = dependency_state(config)
            state.update(dependencies=dependencies, updated_at=datetime.now().isoformat())
            write(directory/'status.json', state)
            if done:
                break
            time.sleep(10)
        allowed = []
        while True:
            active = compute_pids()
            allowed = [pid for pid in active
                if process_identity(pid) == config['allowed_processes'].get(str(pid))]
            unexpected = sorted(set(active)-set(allowed))
            if not unexpected:
                break
            state.update(status='waiting_for_gpus', unexpected_pids=unexpected)
            write(directory/'status.json', state)
            time.sleep(10)
        verify_source(config)
        print(json.dumps(dict(event='four_runs_and_validation_complete', dependencies=dependencies)), flush=True)
        for job in config['jobs']:
            argv = command(config, job, allowed)
            with (Path(job['output'])/'logs/controller.log').open('ab') as stream:
                child = subprocess.Popen(argv, stdin=subprocess.DEVNULL, stdout=stream,
                    stderr=subprocess.STDOUT, start_new_session=True)
            children.append(child)
            job.update(pid=child.pid, command=shlex.join(argv), status='submitted',
                       source_digest=config['source_digest'])
            write(Path(job['output'])/'config/web_process.json', job)
            print(json.dumps(dict(event='submitted', **job)), flush=True)
        state.update(status='training', started_at=datetime.now().isoformat(), jobs=config['jobs'])
        write(directory/'status.json', state)
        plotted_at = 0
        while True:
            report(config)
            if time.monotonic() - plotted_at >= 300:
                from v1_groot_memoryvla.experiment.launch_task21_formal import plot
                for job in config['jobs']:
                    root = Path(job['output'])
                    if (root/job['arm']/'metrics/metrics.jsonl').exists():
                        (root/'plots').mkdir(exist_ok=True)
                        plot(root)
                plotted_at = time.monotonic()
            for job, child in zip(config['jobs'], children):
                if child.poll() not in (None, 0):
                    raise RuntimeError(f"{job['run']} exited {child.returncode}; see its controller.log")
            if all(child.poll() == 0 for child in children):
                break
            time.sleep(30)
        state.update(status='completed', completed_at=datetime.now().isoformat())
        write(directory/'status.json', state)
    except Exception as error:
        state.update(status='failed', error=repr(error))
        write(directory/'status.json', state)
        raise
    # A failed arm does not kill the other arm or any external process.


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', type=Path, required=True)
    parser.add_argument('--prepare-only', action='store_true')
    args = parser.parse_args()
    if args.prepare_only:
        config = read(args.config)
        assert not config.get('jobs'), 'Already reserved'
        config['allowed_processes'] = {str(pid): process_identity(pid) for pid in config['allowed_existing_pids']
                                       if process_identity(pid) is not None}
        reserve(config)
        write(args.config, config)
        print(json.dumps(config['jobs'], indent=2))
    else:
        run(args.config)
