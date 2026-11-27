"""Validate GPU/env/instance plans without starting Isaac or contacting Policy."""

import json
import os
from pathlib import Path
import shutil
import subprocess
import time

import pytest


ENTRY = Path(__file__).resolve().parents[1] / 'scripts/run_groot_eval.sh'


def run(*args):
    return subprocess.run(['bash', str(ENTRY), '--dry-run', *args], capture_output=True, text=True)


@pytest.mark.parametrize('mode,gpus,n_envs,instances', [
    ('single1', [1], 1, [[0]]),
    ('single2', [1], 2, [[0, 1]]),
    ('dual1', [0, 1], 1, [[0], [1]]),
    ('dual2', [0, 1], 2, [[0, 1], [2, 3]]),
])
def test_modes_use_valid_underlying_queue_commands(mode, gpus, n_envs, instances):
    result = run('--mode', mode, '--preset', 'fast-debug', '--run-tag', 'test_modes')
    assert result.returncode == 0, result.stderr
    plan, _ = json.JSONDecoder().raw_decode(result.stdout)
    assert list(plan['instances'].values()) == instances
    caches = []
    for gpu, command in zip(gpus, plan['commands']):
        assert command[command.index('--gpu-index') + 1] == str(gpu)
        assert command[command.index('--n-envs') + 1] == str(n_envs)
        assert command[command.index('--episodes') + 1] == str(n_envs)
        output_root = Path(command[command.index('--output-root') + 1])
        assert output_root == Path(plan['output']) / '.runtime' / f'gpu{gpu}'
        assert command[command.index('--public-output-dir') + 1] == plan['output']
        assert '--monitor-resources' not in command
        assert '--record-video' in command
        assert command[command.index('--video-steps-per-render') + 1] == '24'
        assert '--disable-viewer' not in command
        assert command[command.index('--n-envs') + 1] != '4'
        caches.append(command[command.index('--appdata-path') + 1])
        child = subprocess.run(command + ['--dry-run'], capture_output=True, text=True)
        assert child.returncode == 0, child.stderr
        assert '--max_episode_steps 1200' in child.stdout
    assert len(set(caches)) == len(gpus)


def test_official_mode_keeps_native_horizon_and_rejects_debug_limit():
    result = run('--mode', 'dual2', '--preset', 'official-eval')
    assert result.returncode == 0, result.stderr
    plan, _ = json.JSONDecoder().raw_decode(result.stdout)
    for command in plan['commands']:
        child = subprocess.run(command + ['--dry-run'], capture_output=True, text=True)
        assert child.returncode == 0, child.stderr
        assert '--max_episode_steps 999999999' in child.stdout
    assert run('--preset', 'official-eval', '--debug-max-steps', '80').returncode != 0


def test_conflicting_instances_or_cache_are_rejected():
    overlap = run('--mode', 'dual2', '--preset', 'fast-debug', '--instances-gpu0', '0', '1', '--instances-gpu1', '1', '2')
    cache = run('--mode', 'dual2', '--appdata-gpu0', '/tmp/same-cache', '--appdata-gpu1', '/tmp/same-cache')
    assert overlap.returncode != 0 and 'disjoint' in overlap.stderr
    assert cache.returncode != 0 and 'different AppData' in cache.stderr


@pytest.mark.parametrize('mode,instances,episodes', [
    ('single1', [list(range(10))], 10),
    ('single2', [list(range(10))], 10),
    ('dual1', [list(range(5)), list(range(5, 10))], 5),
    ('dual2', [list(range(5)), list(range(5, 10))], 5),
])
def test_official_defaults_cover_ten_distinct_instances(mode, instances, episodes):
    result = run('--mode', mode)
    assert result.returncode == 0, result.stderr
    plan, _ = json.JSONDecoder().raw_decode(result.stdout)
    assert plan['preset'] == 'official-eval'
    assert list(plan['instances'].values()) == instances
    for command in plan['commands']:
        assert command[command.index('--episodes') + 1] == str(episodes)
        assert '--limit-env-episodes' in command
        assert '--debug-max-steps' not in command
        child = subprocess.run(command + ['--dry-run'], capture_output=True, text=True)
        assert child.returncode == 0, child.stderr
        assert '--max_episode_steps 999999999' in child.stdout


def test_no_flags_means_official_single_gpu_single_env():
    result = run()
    plan, _ = json.JSONDecoder().raw_decode(result.stdout)
    assert plan['mode'] == 'single1' and plan['preset'] == 'official-eval'
    assert plan['episodes_per_gpu'] == 10
    assert list(plan['instances']) == ['1']
    assert plan['record_video'] is True
    assert plan['video_steps_per_render'] == 24
    assert plan['video_fps'] == 20 and plan['video_speedup'] == 16


@pytest.mark.parametrize('mode', ['single1', 'single2', 'dual1', 'dual2'])
def test_multiple_rounds_cover_each_instance_equally(mode):
    result = run('--mode', mode, '--repetitions', '5', '--seed', '123')
    assert result.returncode == 0, result.stderr
    plan, _ = json.JSONDecoder().raw_decode(result.stdout)
    assert plan['episodes_per_gpu'] * len(plan['commands']) == 50
    assert plan['repetitions'] == 5
    for gpu, command in zip(plan['instances'], plan['commands']):
        layout = json.loads(command[command.index('--episode-layout') + 1])
        assert sum(slot['quota'] for slot in layout) == plan['episodes_per_gpu']
        assert all(slot['quota'] == len(slot['instances']) * 5 for slot in layout)
        child = subprocess.run(command + ['--dry-run'], capture_output=True, text=True)
        assert child.returncode == 0, child.stderr
    assert run('--repetitions', '0').returncode != 0


def test_explicit_budget_can_exceed_ten():
    result = run('--mode', 'single2', '--episodes-per-gpu', '100')
    assert result.returncode == 0, result.stderr
    plan, _ = json.JSONDecoder().raw_decode(result.stdout)
    assert plan['episodes_per_gpu'] == 100
    assert [slot['quota'] for slot in plan['episode_layout']['1']] == [50, 50]


def test_repeated_resume_uses_full_instance_pool_and_per_environment_offsets(tmp_path):
    result = run('--mode', 'single2', '--gpu-index', '0', '--repetitions', '5', '--seed', '42')
    config, _ = json.JSONDecoder().raw_decode(result.stdout)
    source = tmp_path/'repeat_eval'
    source.mkdir()
    config.update(policy_server='172.28.62.17:5556')
    (source/'results.json').write_text(json.dumps(dict(status='interrupted', evaluation_mode='official-eval',
        task_id=21, model_name='groot', requested_episodes=50, recorded_episodes=13, configuration=config)))
    records = []
    for env_idx, count in enumerate([6, 7]):
        for i in range(count):
            records.append(dict(gpu_index=0, env_idx=env_idx, instance_id=env_idx*5+i%5, seed=42+env_idx+i, valid=True))
    (source/'episodes.jsonl').write_text(''.join(json.dumps(row)+'\n' for row in records))
    resumed = run('--resume-from', str(source))
    assert resumed.returncode == 0, resumed.stderr
    plan, _ = json.JSONDecoder().raw_decode(resumed.stdout)
    command = plan['commands'][0]
    layout = json.loads(command[command.index('--episode-layout')+1])
    assert [slot['offset'] for slot in layout] == [6, 7]
    assert [slot['quota'] for slot in layout] == [19, 18]
    assert plan['instances'] == {'0': list(range(10))}
    assert plan['base_seed'] == 42
    assert plan['episode_layout'] == config['episode_layout']
    child = subprocess.run(command+['--dry-run'], capture_output=True, text=True)
    assert child.returncode == 0, child.stderr


def test_video_can_be_disabled_or_recorded_at_a_different_interval():
    disabled = run('--no-record-video')
    assert disabled.returncode == 0, disabled.stderr
    plan, _ = json.JSONDecoder().raw_decode(disabled.stdout)
    assert plan['record_video'] is False
    command = plan['commands'][0]
    assert '--no-record-video' in command and '--record-video' not in command
    child = subprocess.run(command + ['--dry-run'], capture_output=True, text=True)
    assert child.returncode == 0, child.stderr
    assert '--record-video' not in child.stdout
    changed = run('--video-steps-per-render', '48')
    plan, _ = json.JSONDecoder().raw_decode(changed.stdout)
    assert plan['video_speedup'] == 32
    assert run('--video-steps-per-render', '0').returncode != 0


def test_default_runtime_cache_is_outside_cleanable_evaluation_outputs():
    result = run('--mode', 'single2', '--gpu-index', '1')
    assert result.returncode == 0, result.stderr
    plan, _ = json.JSONDecoder().raw_decode(result.stdout)
    command = plan['commands'][0]
    cache = Path(command[command.index('--appdata-path') + 1])
    assert not cache.is_relative_to(ENTRY.parents[1] / 'evaluation')
    child = subprocess.run(command + ['--dry-run'], capture_output=True, text=True)
    assert child.returncode == 0, child.stderr
    default = subprocess.run(
        ['bash', str(ENTRY.parent / 'run_groot_priority_eval.sh'), '--dry-run', '--tasks', '21'],
        capture_output=True, text=True,
    )
    assert f'OMNIGIBSON_APPDATA_PATH={cache}' in default.stdout


def test_fixed_and_random_seed_plans_record_resolved_base():
    result = run('--seed', '42', '--seed-mode', 'fixed')
    plan, _ = json.JSONDecoder().raw_decode(result.stdout)
    assert plan['base_seed'] == 42 and plan['seed_mode'] == 'fixed'
    random = run('--seed', 'random', '--seed-mode', 'random')
    plan, _ = json.JSONDecoder().raw_decode(random.stdout)
    assert plan['seed_source'] == 'random' and isinstance(plan['base_seed'], int)
    assert plan['commands'][0][plan['commands'][0].index('--seed') + 1] == str(plan['base_seed'])


@pytest.mark.parametrize('mode', ['single1', 'single2', 'dual1', 'dual2'])
def test_public_entry_forwards_gpu_wait_options(mode):
    result = run('--mode', mode, '--wait-for-gpu', '--gpu-wait-interval', '0.1')
    assert result.returncode == 0, result.stderr
    plan, _ = json.JSONDecoder().raw_decode(result.stdout)
    assert plan['wait_for_gpu'] is True
    for command in plan['commands']:
        assert '--wait-for-gpu' in command
        assert command[command.index('--gpu-wait-interval') + 1] == '0.1'
    assert run('--gpu-wait-interval', '0').returncode != 0


def prepare_busy_gpu_entry(tmp_path):
    """Real two-level launchers, stub only GPU/Policy probes; never start Isaac."""
    entry = prepare_stub_entry(tmp_path)
    scripts = tmp_path / 'scripts'
    for name in ('run_groot_priority_eval.sh', 'groot_eval_results.py'):
        shutil.copy2(ENTRY.parent / name, scripts / name)
    shutil.copy2(ENTRY.parents[1] / 'evaluation/scripts/groot_tasks.py', scripts / 'groot_tasks.py')
    probe = tmp_path / 'policy_probe'
    probe.write_text("#!/bin/sh\nprintf 'PING True MODALITY_KEYS []\\n'\n")
    probe.chmod(0o755)
    smi = tmp_path / 'nvidia-smi'
    smi.write_text("#!/bin/sh\nprintf '12000, 37140\\n'\n")
    smi.chmod(0o755)
    environment = {**os.environ, 'PROJECT_ROOT': str(tmp_path),
                   'GROOT_REPO': str(tmp_path), 'B1K_PYTHON_LAUNCHER': str(probe),
                   'B1K_PYTHON': '/usr/bin/python3', 'OMNIGIBSON_ROOT': str(tmp_path),
                   'BDDL_ROOT': str(tmp_path), 'BEHAVIOR_DATA': str(tmp_path),
                   'PATH': str(tmp_path) + os.pathsep + os.environ['PATH']}
    return entry, environment


@pytest.mark.parametrize('mode', ['single2', 'dual2'])
def test_busy_gpu_is_reported_in_terminal_and_queue_exits_nonzero(tmp_path, mode):
    entry, environment = prepare_busy_gpu_entry(tmp_path)
    gpu_args = ['--gpu-index', '0'] if mode == 'single2' else []
    result = subprocess.run(['bash', str(entry), '--mode', mode, *gpu_args],
                            env=environment, capture_output=True, text=True, timeout=10)
    assert result.returncode != 0
    assert '12000 MiB' in result.stdout and '24576 MiB' in result.stdout
    assert ('--resume-from' if mode == 'single2' else '--wait-for-gpu') in result.stdout
    output = tmp_path / 'evaluation/V_1/baseline/task21_eval01'
    summary = json.loads((output / 'results.json').read_text())
    blocked = [gpu for gpu, details in summary['per_gpu'].items()
               if details.get('queue_status') == 'blocked_before_task']
    assert blocked
    assert all(summary['exit_codes'][gpu] != 0 for gpu in blocked)
    assert summary['status'] == 'failed'
    assert summary['recorded_episodes'] == 0
    assert not (output / 'episodes.jsonl').read_text()


def test_busy_gpu_resume_publishes_waiting_and_ignores_previous_stop(tmp_path):
    entry, environment = prepare_busy_gpu_entry(tmp_path)
    first = subprocess.run(['bash', str(entry), '--mode', 'single2', '--gpu-index', '0'],
                           env=environment, capture_output=True, text=True, timeout=10)
    assert first.returncode != 0
    output = tmp_path / 'evaluation/V_1/baseline/task21_eval01'
    process = subprocess.Popen(['bash', str(entry), '--resume-from', str(output),
                                '--gpu-wait-interval', '0.1'], env=environment,
                               stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    try:
        deadline = time.monotonic() + 12
        while time.monotonic() < deadline:
            current = json.loads((output / 'results.json').read_text())
            if current['per_gpu'].get('0', {}).get('queue_status') == 'waiting_for_gpu':
                break
            if process.poll() is not None:
                pytest.fail(str(process.communicate()))
            time.sleep(0.02)
        assert process.poll() is None
        assert current['per_gpu']['0']['queue_status'] == 'waiting_for_gpu'
        assert current['status'] == 'running' and current['recorded_episodes'] == 0
        process.terminate()
        stdout, stderr = process.communicate(timeout=10)
        assert 'GPU_WAIT' in stdout, stdout + stderr
        final = json.loads((output / 'results.json').read_text())
        assert final['status'] == 'interrupted'
        assert final['recorded_episodes'] == 0
        assert final['configuration']['base_seed'] == 0
        assert final['configuration']['instances'] == {'0': list(range(10))}
    finally:
        if process.poll() is None:
            process.terminate()
            process.communicate(timeout=10)


@pytest.mark.parametrize('mode,free,allowed',[
    ('single1',10240,False),('single1',10241,True),('single1',20833,True),
    ('single2',24576,False),('single2',24577,True),('single2',20833,False)])
def test_shared_gpu_admission_uses_free_memory_and_strict_env_specific_limit(tmp_path,mode,free,allowed):
    entry,environment=prepare_busy_gpu_entry(tmp_path)
    smi=tmp_path/'nvidia-smi'
    smi.write_text(f"#!/bin/sh\nprintf '{free}, {49140-free}\\n'\n")
    result=subprocess.run(['bash',str(entry),'--mode',mode,'--gpu-index','0'],
                          env=environment,capture_output=True,text=True,timeout=10)
    # Probe fixtures never create Isaac; reaching TASK_START proves admission.
    log=(tmp_path/'evaluation/V_1/baseline/task21_eval01/gpu0.log').read_text()
    assert ('TASK_START ' in log)==allowed,log+result.stderr
    assert ('QUEUE_STOP GPU0 free' in result.stdout)==(not allowed)


@pytest.mark.parametrize('layout', ['legacy', 'versioned', 'moved_legacy'])
def test_resume_dry_run_skips_completed_instances_and_preserves_seed_sequence(tmp_path, layout):
    source = tmp_path / 'groot_task22_1'
    if layout == 'versioned':
        source = tmp_path / 'V_2/experment_1/groot_task22_1'
    entry = ENTRY
    if layout == 'moved_legacy':
        entry = prepare_stub_entry(tmp_path)
        source = tmp_path / 'evaluation/V_2/experment_1/groot_task22_1'
    source.mkdir(parents=True)
    config = {
        'mode': 'single2', 'preset': 'official-eval', 'model_name': 'groot',
        'task_id': 22, 'n_envs_per_gpu': 2, 'instances': {'0': list(range(10))},
        'base_seed': 0, 'seed_mode': 'sequence', 'episodes_per_gpu': 10,
        'policy_server': '172.28.62.17:5556', 'appdata': {'0': '/tmp/gpu0-cache'},
    }
    result = {
        'evaluation_mode': 'official-eval', 'task_id': 22, 'model_name': 'groot',
        'status': 'failed', 'requested_episodes': 10, 'recorded_episodes': 4,
        'configuration': config,
    }
    if layout == 'versioned':
        result.update(version='V_2', group='experment_1')
        config.update(version='V_2', group='experment_1')
    elif layout == 'moved_legacy':
        result['version'] = config['version'] = 'experiment_1'
    (source / 'results.json').write_text(json.dumps(result))
    records = [
        {'gpu_index': 0, 'env_idx': 0, 'instance_id': 0, 'seed': 0, 'valid': True},
        {'gpu_index': 0, 'env_idx': 1, 'instance_id': 5, 'seed': 1, 'valid': True},
        {'gpu_index': 0, 'env_idx': 1, 'instance_id': 6, 'seed': 2, 'valid': True},
        {'gpu_index': 0, 'env_idx': 0, 'instance_id': 1, 'seed': 1, 'valid': True},
    ]
    (source / 'episodes.jsonl').write_text(''.join(json.dumps(record) + '\n' for record in records))

    proc = subprocess.run(['bash', str(entry), '--resume-from', str(source), '--dry-run'],
                          capture_output=True, text=True)
    assert proc.returncode == 0, proc.stderr
    plan, _ = json.JSONDecoder().raw_decode(proc.stdout)
    assert plan['instances'] == {'0': list(range(10))}
    assert plan['remaining_instances'] == {'0': [2, 3, 4, 7, 8, 9]}
    assert plan['base_seed'] == 0 and plan['resume_child_base_seed'] == 2
    command = plan['commands'][0]
    assert command[command.index('--episodes') + 1] == '6'
    assert command[command.index('--seed') + 1] == '2'
    assert command[command.index('--n-envs') + 1] == '2'
    assert '--wait-for-gpu' in command
    resumed_tag = command[command.index('--run-tag') + 1]
    expected_task_output = (Path(plan['output']) / '.runtime' / 'gpu0_resume1' /
                            f'groot_n1d6_task022_putting_shoes_on_rack_{resumed_tag}_gpu0')
    assert Path(plan['task_outputs']['0']) == expected_task_output.relative_to(source)
    if layout != 'moved_legacy':
        child = subprocess.run(command + ['--dry-run'], capture_output=True, text=True)
        assert child.returncode == 0, child.stderr
        assert f'RESULTS_JSON={expected_task_output / "results.json"}' in child.stdout
    assert plan['output'] == str(source)
    assert plan['version'] == ('V_1' if layout == 'legacy' else 'V_2')
    assert plan['group'] == ('baseline' if layout == 'legacy' else 'experment_1')
    assert json.loads((source / 'results.json').read_text())['recorded_episodes'] == 4


def prepare_stub_entry(tmp_path):
    scripts = tmp_path / 'scripts'
    scripts.mkdir()
    tools = tmp_path / 'evaluation/scripts'
    tools.mkdir(parents=True)
    shutil.copy2(ENTRY.parents[1] / 'evaluation/scripts/run_groot_eval.sh', tools / ENTRY.name)
    shutil.copy2(ENTRY.parents[1] / 'evaluation/scripts/groot_tasks.py', tools / 'groot_tasks.py')
    shutil.copy2(ENTRY.parent / 'groot_eval_output.py', scripts / 'groot_eval_output.py')
    shutil.copy2(ENTRY.parents[1] / 'evaluation/scripts/b1k_metrics.py', scripts / 'b1k_metrics.py')
    # Exercise real directory allocation, publication, aggregation and cleanup.
    # Only the simulator queue is stubbed; no GPU or policy service is contacted.
    (scripts / 'run_groot_priority_eval.sh').write_text(r'''#!/bin/bash
exec /usr/bin/python3 - "$@" <<'PY'
import json, os, pathlib, sys, time
args = sys.argv[1:]
if '--dry-run' in args:
    raise SystemExit(0)
def value(flag): return args[args.index(flag) + 1]
out = pathlib.Path(value('--output-root'))
public = pathlib.Path(value('--public-output-dir'))
sys.path.insert(0, str(pathlib.Path(os.environ['STUB_PROJECT_ROOT']) / 'scripts'))
from groot_eval_output import update_results
gpu = value('--gpu-index')
tag = value('--run-tag') + '_gpu' + gpu
task = out / ('groot_n1d6_task021_collecting_childrens_toys_' + tag)
task.mkdir(parents=True)
status = os.environ.get('STUB_STATUS', 'complete')
count = int(value('--episodes'))
instances = args[args.index('--instance-ids') + 1:args.index('--seed')]
records = []
for i in range(count if status == 'complete' else 1):
    record = {'gpu_index': int(gpu), 'env_idx': i % int(value('--n-envs')),
              'episode_index': i + 1, 'instance_id': int(instances[i]),
              'seed': 0, 'task_id': 21, 'task': 'collecting_childrens_toys',
              'valid': True, 'success': gpu == '0', 'task_progress': 100 if gpu == '0' else 20,
              'sim_steps': 8, 'episode_length': 8, 'duration_s': 1.0,
              'native_termination': {'timeout': True}}
    records.append(record)
    with (task / 'episodes.jsonl').open('a') as handle:
        handle.write(json.dumps(record) + '\n')
    update_results(public, episode=record)
    if os.environ.get('RELEASE_FILE'):
        while not pathlib.Path(os.environ['RELEASE_FILE']).exists():
            time.sleep(0.01)
(task / 'results.json').write_text(json.dumps({'status': status, 'recorded_episodes': len(records)}))
(task / 'profiling.json').write_text(json.dumps({'policy_get_action': {'count': count, 'mean_s': .1}}))
(task / 'run_meta.json').write_text(json.dumps({'seed': 0, 'gpu_index': int(gpu)}))
(task / 'manifest.csv').write_text('episode_index,success\n1,0\n')
if '--record-video' in args:
    (task / 'videos').mkdir()
    (task / 'videos/episode_001.mp4').write_bytes(b'stub-video-' + gpu.encode())
print('QUEUE_FINISHED', flush=True)
PY
''')
    # The stub runs in an isolated project, just like the relocated real entry.
    return tools / ENTRY.name


@pytest.mark.parametrize('mode,status', [('single2', 'complete'), ('dual2', 'complete'), ('dual2', 'failed')])
def test_entry_keeps_only_one_result_bundle_and_checks_queue_status(tmp_path, mode, status):
    entry = prepare_stub_entry(tmp_path)
    historical_ledger = tmp_path / 'evaluation/V_1/baseline/results.md'
    historical_ledger.parent.mkdir(parents=True, exist_ok=True)
    historical_ledger.write_text('Historical results\n')
    environment = {**os.environ, 'STUB_STATUS': status, 'STUB_PROJECT_ROOT': str(tmp_path)}
    result = subprocess.run(['bash', str(entry), '--mode', mode, '--run-tag', 'a_note'],
                            env=environment, capture_output=True, text=True, timeout=20)
    assert result.returncode == (0 if status == 'complete' else 1), result.stdout + result.stderr
    groups = [path for path in (tmp_path / 'evaluation/V_1/baseline').iterdir() if path.is_dir()]
    assert len(groups) == 1  # No sibling task directories or queue logs.
    group = groups[0]
    assert group.name == 'task21_eval01'
    summary = json.loads((group / 'results.json').read_text())
    assert summary['ok'] is (status == 'complete')
    expected_gpus = ['1'] if mode == 'single2' else ['0', '1']
    assert list(summary['per_gpu']) == expected_gpus
    assert summary['configuration']['run_tag'] == 'a_note'
    assert summary['version'] == 'V_1'
    assert summary['group'] == 'baseline'
    assert historical_ledger.read_text() == 'Historical results\n'
    assert {path.name for path in group.iterdir()} == {
        'results.json', 'episodes.jsonl', 'videos', *(f'gpu{gpu}.log' for gpu in expected_gpus),
    }
    assert summary['saved_video_count'] == len(expected_gpus)
    for gpu in expected_gpus:
        relative = f'videos/gpu{gpu}/episode_001.mp4'
        assert relative in summary['videos']
        assert (group / relative).read_bytes() == b'stub-video-' + gpu.encode()
    records = [json.loads(line) for line in (group / 'episodes.jsonl').read_text().splitlines()]
    assert summary['completed_episodes'] == len(records)
    assert len({record['instance_id'] for record in records}) == len(records)
    assert [record['episode_index'] for record in records] == list(range(1, len(records) + 1))
    assert 'manifest' not in str(summary['per_gpu'])
    assert summary['status'] == status
    if status == 'complete':
        assert summary['completed_episodes'] == summary['requested_episodes'] == 10
        assert summary['success_rate'] == (0.5 if mode == 'dual2' else 0)
        assert all(gpu['performance']['policy_get_action']['mean_s'] == .1
                   for gpu in summary['per_gpu'].values())
    moved = tmp_path / 'copied_run'
    shutil.copytree(group, moved)
    assert all((moved / relative).is_file() for relative in summary['logs'].values())
    assert all((moved / relative).is_file() for relative in summary['videos'])


def test_simple_numbering_and_memory_label_do_not_overwrite_existing_runs(tmp_path):
    entry = prepare_stub_entry(tmp_path)
    # Start simultaneously: directory reservation must be atomic.
    environment = {**os.environ, 'STUB_PROJECT_ROOT': str(tmp_path)}
    children = [subprocess.Popen(['bash', str(entry)], env=environment, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
                for _ in range(2)]
    for child in children:
        stdout, stderr = child.communicate(timeout=20)
        assert child.returncode == 0, stdout + stderr
    memory = subprocess.run(['bash', str(entry), '--model-name', 'grootmeory'],
                            env=environment, capture_output=True, text=True, timeout=20)
    assert memory.returncode == 0, memory.stdout + memory.stderr
    outputs = tmp_path / 'evaluation/V_1/baseline'
    assert {path.name for path in outputs.iterdir()} == {'task21_eval01', 'task21_eval02', 'task21_eval03'}
    dry = subprocess.run(['bash', str(entry), '--dry-run'], capture_output=True, text=True)
    plan, _ = json.JSONDecoder().raw_decode(dry.stdout)
    assert Path(plan['output']).name == 'task21_eval04'
    assert not Path(plan['output']).exists()


def test_new_identity_is_saved_and_changed_policy_cannot_resume(tmp_path):
    entry=prepare_stub_entry(tmp_path)
    identity=dict(checkpoint='/weights/step-003000',training_id='V_2_memory_task21_train02',
                  source_digest='frozen-code',policy_seed='42',method='memory',binding='observed_process')
    environment={**os.environ,'STUB_PROJECT_ROOT':str(tmp_path)}
    result=subprocess.run(['bash',str(entry),'--version','V_2','--group','memory','--policy-identity',json.dumps(identity)],
                          env=environment,capture_output=True,text=True,timeout=20)
    assert result.returncode==0,result.stdout+result.stderr
    output=tmp_path/'evaluation/V_2/memory/task21_eval01'
    saved=json.loads((output/'results.json').read_text())
    assert saved['policy_identity']==identity and saved['configuration']['policy_identity']==identity
    assert saved['evaluation_id']=='task21_eval01' and saved['method']=='memory'
    # No completed records are needed to prove a policy change is rejected.
    saved.update(status='interrupted',recorded_episodes=0,requested_episodes=10)
    (output/'results.json').write_text(json.dumps(saved))
    (output/'episodes.jsonl').write_text('')
    changed=dict(identity,checkpoint='/weights/step-004000')
    resumed=subprocess.run(['bash',str(entry),'--resume-from',str(output),'--policy-identity',json.dumps(changed),'--dry-run'],
                           env=environment,capture_output=True,text=True,timeout=10)
    assert resumed.returncode!=0 and 'identity conflicts' in resumed.stderr
    unchanged=subprocess.run(['bash',str(entry),'--resume-from',str(output),'--policy-identity',json.dumps(identity),'--dry-run'],
                             env=environment,capture_output=True,text=True,timeout=10)
    assert unchanged.returncode==0,unchanged.stderr
    plan=json.JSONDecoder().raw_decode(unchanged.stdout)[0]
    assert plan['output']==str(output) and plan['policy_identity']==identity


def test_diagnostic_files_are_opt_in_and_fast_debug_has_no_official_rate(tmp_path):
    entry = prepare_stub_entry(tmp_path)
    result = subprocess.run(['bash', str(entry), '--preset', 'fast-debug', '--diagnostics'],
                            env={**os.environ, 'STUB_PROJECT_ROOT': str(tmp_path)}, capture_output=True, text=True, timeout=20)
    assert result.returncode == 0, result.stdout + result.stderr
    output = tmp_path / 'evaluation/V_1/baseline/task21_eval01'
    summary = json.loads((output / 'results.json').read_text())
    assert summary['evaluation_mode'] == 'fast-debug' and summary['success_rate'] is None
    assert summary['official_episodes'] == 0
    assert (output / 'diagnostics').is_dir() and not (output / '.runtime').exists()
    assert '--monitor-resources' in result.stdout
