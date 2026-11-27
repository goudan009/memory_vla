#!/usr/bin/env bash
# One entry for single GPU 1/2 envs and two GPUs with 1/2 envs each.
set -euo pipefail
GROOT_ENTRY_ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)
export GROOT_ENTRY_ROOT
exec /usr/bin/python3 -u - "$@" <<'PY'
import argparse
import json
import os
from pathlib import Path
import re
import secrets
import shlex
import signal
import shutil
import subprocess
import sys
import time

root = Path(os.environ['GROOT_ENTRY_ROOT'])
evaluation_root = Path(os.environ.get('GROOT_EVAL_ROOT', str(root / 'evaluation'))).resolve()
sys.path.insert(0, str(root / 'scripts'))
sys.path.insert(0, str(root / 'evaluation/scripts'))
from groot_eval_output import allocate_output, update_results
from groot_tasks import TASK_NAMES, print_tasks
import hashlib
parser = argparse.ArgumentParser(description='GR00T single1/single2/dual1/dual2 evaluation presets. Video at 16x, chunk=8, original viewer.')
parser.add_argument('--mode', choices=['single1', 'single2', 'dual1', 'dual2'], default='single1')
parser.add_argument('--preset', choices=['fast-debug', 'official-eval'], default='official-eval')
parser.add_argument('--gpu-index', type=int, choices=[0, 1], help='single modes only; default GPU1')
parser.add_argument('--task', type=int, choices=sorted(TASK_NAMES), default=21, help='B1K-2025 task ID 0-49')
parser.add_argument('--list-tasks', action='store_true', help='list all 50 tasks and local instance availability; do not start evaluation')
parser.add_argument('--episodes-per-gpu', type=int, help='official default: 10 on one GPU, 5 per GPU in dual modes; debug: one window per env')
parser.add_argument('--repetitions', type=int, default=1, help='repeat the per-GPU episode budget; default 1 round')
parser.add_argument('--instances-gpu0', type=int, nargs='+')
parser.add_argument('--instances-gpu1', type=int, nargs='+')
parser.add_argument('--seed', default='0', help='integer base seed or random (record the resolved seed)')
parser.add_argument('--seed-mode', choices=['sequence', 'fixed', 'random'], default='sequence', help='sequence: base+env_idx+episode; fixed: same seed; random: reproducible per-episode seeds derived from base')
parser.add_argument('--debug-max-steps', type=int, help='fast-debug only; default 1200 actions = 150 outer loops')
parser.add_argument('--action-steps', type=int, choices=(8, 16), help='executed action chunk; default8, resume inherits saved value')
parser.add_argument('--policy-host', default='172.28.62.17')
parser.add_argument('--policy-port', type=int, default=5556)
parser.add_argument('--model-name', choices=['groot', 'grootmeory'], default='groot', help='legacy metadata label; new results use taskNN_evalNN; model is selected by policy endpoint')
parser.add_argument('--version', help='version directory, default V_1; custom names accepted; resume inherits its saved version')
parser.add_argument('--group', help='method/result group: baseline, memory, official, ablation_NAME; legacy names still accepted')
parser.add_argument('--policy-identity', help=argparse.SUPPRESS)
parser.add_argument('--policy-checkpoint', help='record the exact policy checkpoint for CLI runs (user-declared, not remotely verified)')
parser.add_argument('--training-id', help='record the training ID that produced --policy-checkpoint')
parser.add_argument('--run-tag', help='optional note stored in results.json; does not affect the folder name')
parser.add_argument('--resume-from', type=Path, help='resume an interrupted single-GPU official evaluation directory; other settings are read from its results.json')
parser.add_argument('--diagnostics', action='store_true', help='also keep detailed per-GPU diagnostic files and resource samples')
parser.add_argument('--record-video', action=argparse.BooleanOptionalAction, default=None, help='save video (default: enabled; --no-record-video disables it)')
parser.add_argument('--video-steps-per-render', type=int, help='action steps per saved frame (default: 24; 16x at 30 Hz and 20 FPS)')
parser.add_argument('--wait-for-gpu', action='store_true', help='wait for free VRAM (>10 GiB for 1 env, >24 GiB for 2 envs); resume always waits')
parser.add_argument('--gpu-wait-interval', type=float, default=30.0, help='seconds between GPU availability checks (default: 30)')
parser.add_argument('--appdata-gpu0', type=Path)
parser.add_argument('--appdata-gpu1', type=Path)
parser.add_argument('--dry-run', action='store_true')
args = parser.parse_args()
try:
    policy_identity=json.loads(args.policy_identity) if args.policy_identity else None
    if policy_identity is not None and not isinstance(policy_identity,dict):
        raise ValueError('expected object')
except ValueError:
    parser.error('--policy-identity must be a JSON object')
if args.policy_checkpoint:
    if not Path(args.policy_checkpoint).is_absolute():
        parser.error('--policy-checkpoint must be absolute')
    policy_identity=dict(checkpoint=args.policy_checkpoint,training_id=args.training_id,binding='user_declared')
elif args.training_id:
    parser.error('--training-id requires --policy-checkpoint')
if args.list_tasks:
    print_tasks(os.environ.get('GROOT_REPO', '/nvme3/projects/GR00T/Isaac-GR00T-n1d6-9b37aa1'))
    raise SystemExit(0)
for option in ('version', 'group'):
    value = getattr(args, option)
    if value is not None and not re.fullmatch(r'\w[\w.-]*', value):
        parser.error(f'--{option} must be a single directory name using letters, digits, underscore, dot or hyphen')
# Keep existing commands usable under the new version/group layout.
if args.group and re.fullmatch(r'experiment_[1-9][0-9]*', args.group):
    args.group = args.group.replace('experiment_', 'experment_', 1)
if args.version == 'baseline' or (args.version and re.fullmatch(r'experiment_[1-9][0-9]*', args.version)):
    legacy_group = args.version.replace('experiment_', 'experment_', 1)
    if args.group is not None and args.group != legacy_group:
        parser.error('legacy --version conflicts with --group; use --version V_1 --group NAME')
    args.version, args.group = 'V_1', legacy_group
seed_source = args.seed
try:
    args.seed = secrets.randbelow(2**31) if args.seed == 'random' else int(args.seed)
except ValueError:
    parser.error('--seed must be an integer or random')
if not 0 <= args.seed < 2**32:
    parser.error('--seed must be in [0, 2**32)')
if args.repetitions < 1:
    parser.error('--repetitions must be positive')
dual = args.mode.startswith('dual')
n_envs = int(args.mode[-1])
if dual and args.gpu_index is not None:
    parser.error('--gpu-index is only for single1/single2; dual modes use GPU0 and GPU1')
if args.debug_max_steps is not None and args.preset != 'fast-debug':
    parser.error('--debug-max-steps requires --preset fast-debug')
if args.debug_max_steps is not None and args.debug_max_steps < 1:
    parser.error('--debug-max-steps must be positive')
if not 1 <= args.policy_port <= 65535:
    parser.error('--policy-port must be between 1 and 65535')
if not args.gpu_wait_interval > 0:
    parser.error('--gpu-wait-interval must be positive')
if args.video_steps_per_render is not None and args.video_steps_per_render < 1:
    parser.error('--video-steps-per-render must be positive')
if args.run_tag is not None and not re.fullmatch(r'[A-Za-z0-9_.-]+', args.run_tag):
    parser.error('run-tag must contain only letters, digits, dot, underscore or hyphen')
resume = None
resume_offsets = {}
resume_original_seed = None
resume_original_assignments = None
episode_layout = {}
child_layout = {}
if args.resume_from:
    output = args.resume_from.expanduser().resolve()
    result_path = output / 'results.json'
    journal = output / 'episodes.jsonl'
    if not result_path.is_file() or not journal.is_file():
        parser.error('--resume-from must contain results.json and episodes.jsonl')
    try:
        resume = json.loads(result_path.read_text())
    except (OSError, json.JSONDecodeError) as error:
        parser.error(f'cannot read resume results: {error}')
    config = resume.get('configuration', {})
    stored_version = resume.get('version') or config.get('version') or (
        output.parent.name if re.fullmatch(r'baseline|experiment_[1-9][0-9]*', output.parent.name) else 'baseline'
    )
    stored_group = resume.get('group') or config.get('group') or 'baseline'
    if stored_version == 'baseline' or re.fullmatch(r'experiment_[1-9][0-9]*', stored_version):
        stored_group, stored_version = stored_version.replace('experiment_', 'experment_', 1), 'V_1'
    # A moved historical run keeps its original result contents. Its new path
    # supplies the version/group; resume still writes into that exact run folder.
    try:
        parts = output.relative_to(evaluation_root).parts
    except ValueError:
        parts = ()
    if len(parts) == 3:
        stored_version, stored_group = parts[:2]
    if args.version is not None and args.version != stored_version:
        parser.error('--version conflicts with the saved resume version')
    if args.group is not None and args.group != stored_group:
        parser.error('--group conflicts with the saved resume group')
    args.version = stored_version
    args.group = stored_group
    if resume.get('evaluation_mode') != 'official-eval' or config.get('preset') != 'official-eval':
        parser.error('only official-eval runs can be resumed')
    if resume.get('status') not in ('failed', 'interrupted'):
        parser.error(f"resume source must be failed/interrupted, got status={resume.get('status')!r}")
    if config.get('mode') not in ('single1', 'single2'):
        parser.error('resume currently supports single1/single2 runs; dual runs are not supported yet')
    args.mode = config['mode']
    args.preset = 'official-eval'
    args.model_name = resume.get('model_name', 'groot')
    args.task = int(resume['task_id'])
    resume_original_seed = int(config['base_seed'])
    args.seed = resume_original_seed
    args.seed_mode = config.get('seed_mode', 'sequence')
    episode_layout = config.get('episode_layout', {})
    args.repetitions = int(config.get('repetitions', 1))
    if args.seed_mode == 'random' and not episode_layout:
        parser.error('resume currently supports seed-mode sequence/fixed only; random runs need explicit per-episode seed mapping')
    args.policy_host, port = config.get('policy_server', '172.28.62.17:5556').rsplit(':', 1)
    args.policy_port = int(port)
    args.episodes_per_gpu = int(resume['requested_episodes'])
    if args.mode == 'single1':
        args.gpu_index = int(next(iter(config['instances'])))
    else:
        args.gpu_index = int(next(iter(config['instances'])))
    stored_instances = config.get('instances', {}).get(str(args.gpu_index))
    if stored_instances is None:
        parser.error('resume configuration is missing the selected GPU instance list')
    n_envs = int(config['n_envs_per_gpu'])
    resume_original_assignments = config['instances']
    all_records = []
    for line in journal.read_text(encoding='utf-8').splitlines():
        try:
            all_records.append(json.loads(line))
        except json.JSONDecodeError:
            parser.error('episodes.jsonl has a malformed line; refusing unsafe resume')
    if any(not record.get('valid', False) for record in all_records):
        parser.error('resume currently requires all recorded Episodes to be valid; inspect invalid records first')
    slots = [stored_instances[i * (len(stored_instances) // n_envs): (i + 1) * (len(stored_instances) // n_envs)]
             if i < n_envs - 1 else stored_instances[i * (len(stored_instances) // n_envs):]
             for i in range(n_envs)]
    saved_layout = episode_layout.get(str(args.gpu_index))
    if saved_layout:
        slots = [slot['instances'] for slot in saved_layout]
    quotas = [slot['quota'] for slot in saved_layout] if saved_layout else [len(slot) for slot in slots]
    done_ids = {i: [] for i in range(n_envs)}
    for record in all_records:
        if int(record.get('gpu_index', args.gpu_index)) != args.gpu_index:
            parser.error('resume source contains records from a different GPU; use an original dual-mode resume workflow')
        env_idx = int(record['env_idx'])
        if env_idx not in done_ids or len(done_ids[env_idx]) >= quotas[env_idx]:
            parser.error('resume journal env_idx does not match its saved environment layout')
        expected = slots[env_idx][len(done_ids[env_idx]) % len(slots[env_idx])]
        if int(record['instance_id']) != expected:
            parser.error(f"resume records are not a completed prefix for env{env_idx}: expected instance {expected}, got {record['instance_id']}")
        done_ids[env_idx].append(expected)
    counts = {len(values) for values in done_ids.values()}
    if len(counts) != 1 and not saved_layout:
        parser.error('resume requires every environment to have completed the same number of Episodes')
    resume_offsets = {str(index): count for index, count in ((i, len(v)) for i, v in done_ids.items())}
    completed_per_env = next(iter(counts))
    # The child receives only the remaining prefix. Shift sequence-mode seeds
    # so its episode number 0 matches the first not-yet-recorded episode.
    if args.seed_mode == 'sequence' and not saved_layout:
        args.seed += completed_per_env
    remaining_slots = {i: slots[i][len(done_ids[i]):] for i in range(n_envs)}
    if any(not values for values in remaining_slots.values()) and not saved_layout:
        parser.error('resume would leave an empty environment slot; this layout cannot be represented safely')
    remaining = [instance for i in range(n_envs) for instance in remaining_slots[i]]
    if not remaining and not saved_layout:
        parser.error('there are no remaining Episodes to resume')
    if resume.get('recorded_episodes', 0) >= resume.get('requested_episodes', 0):
        parser.error('all requested Episodes are already recorded')
    if args.mode == 'single2':
        args.gpu_index = int(next(iter(config['instances'])))
    resume_count = int(config.get('resume_count', 0)) + 1
    assignments = {args.gpu_index: remaining}
    episodes = len(remaining)
    if saved_layout:
        assignments = {args.gpu_index: stored_instances}
        child_layout[args.gpu_index] = [dict(slot, offset=len(done_ids[i]), quota=slot['quota']-len(done_ids[i]))
                                       for i, slot in enumerate(saved_layout)]
        episodes = sum(slot['quota'] for slot in child_layout[args.gpu_index])
    dual = False
    n_envs = int(config['n_envs_per_gpu'])
    gpus = [args.gpu_index]
    if episodes < n_envs:
        parser.error('too few remaining Episodes for the original environment count')
    official = True
else:
    gpus = [0, 1] if dual else [1 if args.gpu_index is None else args.gpu_index]
    official = args.preset == 'official-eval'
    episodes = args.episodes_per_gpu if args.episodes_per_gpu is not None else ((5 if dual else 10) if official else n_envs)
    episodes_per_round = episodes
    episodes *= args.repetitions
    assignments = {}
    resume_count = 0
if not args.resume_from:
    # Assign per GPU below using normal launch settings.
    pass
appdata = {}
for gpu in gpus:
    if args.resume_from:
        ids = assignments[gpu]
    else:
        ids = getattr(args, f'instances_gpu{gpu}')
        if ids is None:
            count = (5 if dual else 10) if official else n_envs
            offset = gpu * count if dual else 0
            ids = list(range(offset, offset + count))
    if len(ids) != len(set(ids)) or any(i not in range(10) for i in ids):
        parser.error('each instance list must have unique IDs in 0-9')
    if len(ids) < n_envs or episodes < n_envs:
        parser.error('at least one assigned instance per environment and n_envs <= episodes-per-gpu is required')
    assignments[gpu] = ids
    if not args.resume_from and (args.repetitions > 1 or episodes > len(ids)):
        per_slot = len(ids) // n_envs
        cycles, remainder = divmod(episodes_per_round, len(ids))
        layout = []
        for index in range(n_envs):
            start = index * per_slot
            slot_ids = ids[start:(index+1)*per_slot] if index < n_envs-1 else ids[start:]
            extra = remainder // n_envs + (remainder % n_envs if index == n_envs-1 else 0)
            quota = (cycles*len(slot_ids) + extra) * args.repetitions
            layout.append(dict(instances=slot_ids, quota=quota, offset=0))
        episode_layout[str(gpu)] = layout
        child_layout[gpu] = layout
    layout = child_layout.get(gpu)
    max_episode = (max(slot['quota'] + slot['offset'] - 1 + index for index, slot in enumerate(layout))
                   if layout else episodes//n_envs + episodes % n_envs + n_envs-2)
    if args.seed_mode == 'sequence' and args.seed + max_episode >= 2**32:
        parser.error('incremented episode seeds exceed the 32-bit range')
    cache = getattr(args, f'appdata_gpu{gpu}')
    # Runtime caches must survive removal of old evaluation results. Do not
    # silently reuse a diagnostic run's cache under evaluation/.
    cache_root = Path(os.environ.get('GROOT_CACHE_ROOT', '/nvme3/tmp/memory_vla'))
    appdata[gpu] = (cache or cache_root / f'groot_runtime_gpu{gpu}/appdata').resolve()
if dual and set(assignments[0]) & set(assignments[1]):
    parser.error('GPU0 and GPU1 instance IDs must be disjoint')
if dual and appdata[0] == appdata[1]:
    parser.error('GPU0 and GPU1 must use different AppData directories')

args.version = args.version or 'V_1'
args.group = args.group or 'baseline'
saved_config = resume.get('configuration', {}) if resume else {}
if args.action_steps is None:
    args.action_steps = int(saved_config.get('action_chunk', 8))
if resume:
    saved_identity=resume.get('policy_identity') or saved_config.get('policy_identity')
    if policy_identity and saved_identity:
        for key in ('checkpoint','training_id','source_digest','policy_seed'):
            if saved_identity.get(key) is not None and policy_identity.get(key)!=saved_identity[key]:
                parser.error('resume policy identity conflicts with saved '+key)
    policy_identity=saved_identity or policy_identity
policy_identity=policy_identity or {'binding':'unlinked'}
if args.record_video is None:
    args.record_video = saved_config.get('record_video', True)
if args.video_steps_per_render is None:
    args.video_steps_per_render = saved_config.get('video_steps_per_render', 24)
version_root = evaluation_root / args.version / args.group
output = (args.resume_from.expanduser().resolve() if args.resume_from else
          allocate_output(version_root, args.model_name, args.task, dry_run=args.dry_run))
protocol_path = root / 'evaluation/evaluation.md'
protocol = ({'id': 'b1k-groot-v1', 'sha256': hashlib.sha256(protocol_path.read_bytes()).hexdigest()}
            if protocol_path.is_file() else {'id': 'unrecorded', 'sha256': None})
if resume:
    # A resume keeps the protocol under which this run began; legacy runs are
    # explicitly unrecorded instead of being retroactively assigned a protocol.
    protocol = resume.get('protocol', {'id': 'legacy-unrecorded', 'sha256': None})
tag = output.name
commands = []
task_names = TASK_NAMES
task_outputs = {}
for gpu in gpus:
    gpu_output = output / '.runtime' / (f'gpu{gpu}_resume{resume_count}' if args.resume_from else f'gpu{gpu}')
    queue_run_tag = f'{tag}_resume{resume_count}' if args.resume_from else tag
    task_output = gpu_output / f'groot_n1d6_task{args.task:03d}_{task_names[args.task]}_{queue_run_tag}_gpu{gpu}'
    task_outputs[str(gpu)] = str(task_output.relative_to(output))
    command = ['bash', str(root / 'scripts/run_groot_priority_eval.sh'),
               '--preset', args.preset, '--tasks', str(args.task),
               '--gpu-index', str(gpu), '--n-envs', str(n_envs),
               '--episodes', str(episodes), '--instance-ids', *map(str, assignments[gpu]),
               '--seed', str(args.seed), '--seed-mode', args.seed_mode,
               '--limit-env-episodes', '--action-steps', str(args.action_steps),
               '--policy-host', args.policy_host, '--policy-port', str(args.policy_port),
               '--appdata-path', str(appdata[gpu]), '--run-tag', queue_run_tag,
               '--output-root', str(gpu_output), '--public-output-dir', str(output)]
    command += ['--record-video' if args.record_video else '--no-record-video',
                '--video-steps-per-render', str(args.video_steps_per_render)]
    if gpu in child_layout:
        command += ['--episode-layout', json.dumps(child_layout[gpu], separators=(',', ':'))]
    if args.resume_from or args.wait_for_gpu:
        # A resume must not fail merely because another job is still using the
        # selected GPU. The child queue waits and can be cancelled safely.
        command.append('--wait-for-gpu')
    command += ['--gpu-wait-interval', str(args.gpu_wait_interval)]
    if args.diagnostics:
        command.append('--monitor-resources')
    if args.debug_max_steps is not None:
        command += ['--debug-max-steps', str(args.debug_max_steps)]
    commands.append(command)
plan = {'mode': args.mode, 'preset': args.preset, 'model_name': args.model_name,
        'device': os.environ.get('GROOT_DEVICE_ID', 'local'),
        'naming_schema': 'groot-runs-v2', 'evaluation_id': output.name, 'policy_identity': policy_identity,
        'method': policy_identity.get('method'),
        'version': args.version, 'group': args.group, 'protocol': protocol,
        'run_tag': args.run_tag, 'task_id': args.task, 'n_envs_per_gpu': n_envs,
        'instances': resume_original_assignments if resume else assignments,
        'base_seed': args.seed, 'seed_source': seed_source,
        'seed_mode': args.seed_mode, 'episodes_per_gpu': episodes, 'commands': commands,
        'repetitions': args.repetitions,
        'episodes_per_round': saved_config.get('episodes_per_round', episodes) if resume else episodes_per_round,
        'episode_layout': episode_layout,
        'output': str(output), 'task_outputs': task_outputs, 'synchronized_start': False,
        'policy_server': f'{args.policy_host}:{args.policy_port}', 'action_chunk': args.action_steps,
        'record_video': args.record_video, 'video_steps_per_render': args.video_steps_per_render,
        'video_fps': 20, 'video_speedup': args.video_steps_per_render * 20 / 30,
        'viewer_enabled': True,
        'task40_stage_trace': args.task == 40 and os.environ.get('GROOT_TASK40_TRACE', '1') == '1',
        'runner_max_episode_steps': (args.debug_max_steps or 1200) if not official else 999999999,
        'diagnostics': args.diagnostics, 'wait_for_gpu': bool(args.resume_from or args.wait_for_gpu),
        'gpu_wait_interval': args.gpu_wait_interval}
if resume:
    plan['remaining_instances'] = assignments
    plan['base_seed'] = resume_original_seed
    plan['resume_child_base_seed'] = args.seed
    plan['resume_count'] = resume_count
print(json.dumps(plan, indent=2), flush=True)
for gpu, command in zip(gpus, commands):
    print(f'GPU{gpu}: {shlex.join(command)}', flush=True)
if args.dry_run:
    raise SystemExit(0)

environment = os.environ.copy()
environment['GROOT_TASK40_TRACE'] = '1' if plan['task40_stage_trace'] else '0'
# These options belong to the separate synchronized diagnostic driver.
environment.pop('GROOT_EVAL_BARRIER_DIR', None)
for flag in ('GROOT_EVAL_DETAIL_PROFILE', 'GROOT_EVAL_SIM_PROFILE', 'GROOT_EVAL_CAMERA_COUNTERS'):
    environment[flag] = '0'
started_at = time.perf_counter()
configuration = {key: value for key, value in plan.items()
                 if key not in ('commands', 'task_outputs', 'output')}
configuration['appdata'] = {str(gpu): str(appdata[gpu]) for gpu in gpus}
configuration['invocation'] = ['evaluation/scripts/run_groot_eval.sh', *sys.argv[1:]]
if resume:
    configuration['resume_from'] = str(output)
    configuration['resume_completed_episodes'] = int(resume.get('recorded_episodes', 0))
    configuration['resume_original_base_seed'] = resume_original_seed
    configuration['resume_child_base_seed'] = args.seed
    configuration['base_seed'] = resume_original_seed
    configuration['resume_count'] = resume_count
update_results(output, updates={
    'evaluation_mode': args.preset, 'task_id': args.task, 'task_name': task_names[args.task],
    'model_name': args.model_name, 'status': 'running', 'ok': False,
    'version': args.version, 'group': args.group, 'protocol': protocol,
    'naming_schema': plan['naming_schema'], 'evaluation_id': output.name,
    'method': plan['method'], 'policy_identity': policy_identity,
    'requested_episodes': (int(resume['requested_episodes']) if resume else episodes * len(gpus)),
    'configuration': configuration,
    'started_at': time.strftime('%Y-%m-%dT%H:%M:%S%z'),
    'logs': {str(gpu): f'gpu{gpu}.log' for gpu in gpus},
})
controller = {'pid': os.getpid(), 'start_ticks': int(Path('/proc/self/stat').read_text().rsplit(')',1)[1].split()[19])}
(output / '.controller.json').write_text(json.dumps(controller))
children, logs = [], []
interrupted = False
queue_log_offsets = {gpu: (output / f'gpu{gpu}.log').stat().st_size
                     if (output / f'gpu{gpu}.log').exists() else 0 for gpu in gpus}
announced_queue_messages = {}

def stop(signum=None, frame=None):
    global interrupted
    if signum is not None:
        interrupted = True
    for child in children:
        if child.poll() is None:
            child.terminate()  # Queue forwards termination to its own simulator group.

signal.signal(signal.SIGINT, stop)
signal.signal(signal.SIGTERM, stop)

def collect_gpu_results():
    per_gpu = {}
    for gpu in gpus:
        task_output = output / task_outputs[str(gpu)]
        details = {}
        for filename, field in [('results.json', 'summary'), ('run_meta.json', 'configuration'),
                                ('profiling.json', 'performance')]:
            path = task_output / filename
            if path.exists():
                details[field] = json.loads(path.read_text())
        # Public paths are all relative to the one run folder. Internal summary
        # links are legacy scratch paths and must not survive cleanup.
        if 'summary' in details:
            for field in ('log', 'manifest', 'episodes'):
                details['summary'].pop(field, None)
        if 'configuration' in details:
            details['configuration'].pop('public_output_dir', None)
        details['environments'] = {
            path.stem: json.loads(path.read_text()) for path in task_output.glob('runtime_env*.json')
        }
        if 'summary' not in details:
            queue_path = output / f'gpu{gpu}.log'
            if queue_path.exists():
                # Resume appends to the same log. Only inspect this attempt;
                # an older QUEUE_STOP must not hide the current GPU_WAIT.
                with queue_path.open('rb') as handle:
                    handle.seek(queue_log_offsets[gpu])
                    queue_text = handle.read().decode('utf-8', errors='replace')
                events = [line for line in queue_text.splitlines()
                          if any(marker in line for marker in ('QUEUE_STOP ', 'GPU_WAIT ', 'TASK_START '))]
                if events and 'QUEUE_STOP ' in events[-1]:
                    details['queue_status'] = 'blocked_before_task'
                    details['queue_message'] = events[-1]
                elif events and 'GPU_WAIT ' in events[-1]:
                    details['queue_status'] = 'waiting_for_gpu'
                    details['queue_message'] = events[-1]
        per_gpu[str(gpu)] = details
    return per_gpu

def report_queue_messages(per_gpu):
    for gpu, details in per_gpu.items():
        message = details.get('queue_message')
        if message and message != announced_queue_messages.get(gpu):
            print(f'GPU{gpu}: {message}', flush=True)
            announced_queue_messages[gpu] = message

try:
    # Validate both queue configurations before launching either GPU.
    for command in commands:
        subprocess.run(command + ['--dry-run'], env=environment, check=True, stdout=subprocess.DEVNULL)
    for gpu, command in zip(gpus, commands):
        if interrupted:
            break
        log = (output / f'gpu{gpu}.log').open('a', buffering=1)
        logs.append(log)
        children.append(subprocess.Popen(command, stdout=log, stderr=subprocess.STDOUT, env=environment))
        print(f'GPU{gpu} queue_pid={children[-1].pid} log={log.name}', flush=True)
    last_publish = 0.0
    while any(child.poll() is None for child in children):
        if any(child.poll() not in (None, 0) for child in children):
            stop()
        if time.perf_counter() - last_publish >= 5.0:
            per_gpu = collect_gpu_results()
            report_queue_messages(per_gpu)
            update_results(output, updates={'per_gpu': per_gpu})
            last_publish = time.perf_counter()
        time.sleep(0.5)
finally:
    if any(child.poll() is None for child in children):
        stop()
    for child in children:
        child.wait()
    for log in logs:
        log.close()
    per_gpu = collect_gpu_results()
    report_queue_messages(per_gpu)
    codes = {str(gpu): child.returncode for gpu, child in zip(gpus, children)}
    statuses = {
        gpu: value.get('summary', {}).get('status', value.get('queue_status', 'missing'))
        for gpu, value in per_gpu.items()
    }
    ok = (not interrupted and len(children) == len(gpus)
          and all(c == 0 for c in codes.values()) and all(s == 'complete' for s in statuses.values()))
    # Ensure both the queue status and the durable public journal are complete.
    final = update_results(output, updates={
        'ok': ok, 'status': 'interrupted' if interrupted else ('complete' if ok else 'failed'),
        'per_gpu': per_gpu, 'exit_codes': codes, 'elapsed_wall_s': time.perf_counter() - started_at,
        'finished_at': time.strftime('%Y-%m-%dT%H:%M:%S%z'),
    })
    if ok and final['completed_episodes'] != final['requested_episodes']:
        ok = False
        final = update_results(output, updates={'ok': False, 'status': 'failed'})
    # Videos are evaluation artifacts; keep them outside the scratch directory
    # before cleanup, with a separate path for each GPU and resume attempt.
    for gpu in gpus:
        source = output / task_outputs[str(gpu)] / 'videos'
        destination = output / 'videos' / (f'gpu{gpu}_resume{resume_count}' if resume else f'gpu{gpu}')
        for video in sorted(source.glob('*.mp4')):
            destination.mkdir(parents=True, exist_ok=True)
            shutil.move(str(video), str(destination / video.name))
    videos = sorted(str(path.relative_to(output)) for path in (output / 'videos').rglob('*.mp4'))
    final = update_results(output, updates={'videos': videos, 'saved_video_count': len(videos)})
    # The journal and all useful summaries are already in the public bundle.
    # Only this invocation's generated scratch files are removed, after workers
    # have exited. Historic runs and persistent Kit caches are never touched.
    runtime = output / '.runtime'
    if args.diagnostics:
        runtime.rename(output / 'diagnostics')
    else:
        shutil.rmtree(runtime)
    (output / '.controller.json').unlink(missing_ok=True)
    print(f'FINISHED ok={ok} statuses={statuses} output={output}', flush=True)
    if any(status == 'blocked_before_task' for status in statuses.values()):
        if dual:
            print('Dual-mode resume is unsupported. Rerun the original command with --wait-for-gpu; '
                  'existing results are retained in this folder.', flush=True)
        else:
            print('No simulation started on the blocked GPU. Resume and wait for it to become idle:\n  '
                  + shlex.join(['./evaluation/scripts/run_groot_eval.sh', '--resume-from', str(output)]), flush=True)
raise SystemExit(0 if ok else 1)
PY
