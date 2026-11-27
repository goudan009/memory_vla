"""Persistent-client evaluation stages for the existing V2_1 workbench.

No simulator or model is launched directly: every mutation uses workbench HTTP.
The caller owns the lock, durable state, timer and checkpoint dependency gates.
"""
from __future__ import annotations

import json
import math
from pathlib import Path
import shlex

ROOT = Path(__file__).resolve().parents[1]
EVALUATION = ROOT / 'evaluation'
EVIDENCE = ROOT / 'Output/test_01/experiment_14/evidence/automation'
PORT = 5594
HOST = '172.28.62.17'
PYTHON = '/nvme3/projects/memory_vla/envs/n1d6-p0/bin/python'
ACTIVE = {'ready', 'loading', 'starting', 'stopping'}


def read(path, default=None):
    try:
        return json.loads(path.read_text())
    except FileNotFoundError:
        return default


def block(state, save, reason):
    state.update(status='blocked', reason=reason)
    save()
    return False


def matching_service(job, checkpoint):
    return (str(job.get('checkpoint_resolved') or job.get('checkpoint')) == checkpoint['checkpoint']
            and job.get('source_dir') == checkpoint['source']
            and job.get('history_mode', 'normal') == 'normal'
            and str(job.get('policy_seed', job.get('seed'))) == '42'
            and str(job.get('gpu_index')) == '0'
            and int(job.get('port', 0)) == PORT)


def eval_request(checkpoint):
    # model_name is a legacy enum (groot/grootmeory), not a free-form model ID.
    # Exact checkpoint identity and run_tag distinguish the three models.
    return dict(device='local', version='V_2_1', group='experiment_14',
                model_name='grootmeory' if checkpoint['name'] == 'memory' else 'groot',
                mode='single1', gpu_index=1, task=40, preset='official-eval',
                seed=0, seed_mode='sequence', policy_host=HOST, policy_port=PORT,
                run_tag=checkpoint['run'] + '__official_eval', episodes_per_gpu=10,
                instances_gpu1=list(range(10)), record_video=True,
                video_steps_per_render=24, wait_for_gpu=True)


def check_plan(plan, checkpoint, tag):
    expected = dict(mode='single1', preset='official-eval', task_id=40,
                    base_seed=0, seed_mode='sequence', episodes_per_gpu=10,
                    action_chunk=8, record_video=True, video_steps_per_render=24,
                    video_fps=20, run_tag=tag, instances={'1': list(range(10))})
    for key, value in expected.items():
        if plan.get(key) != value:
            raise ValueError(f'evaluation protocol mismatch: {key}={plan.get(key)!r}')
    observed = plan.get('policy_identity') or {}
    if (observed.get('checkpoint') != checkpoint['checkpoint']
            or observed.get('source_dir') != checkpoint['source']
            or observed.get('binding') != 'observed_process'):
        raise ValueError('evaluation must bind the observed exact checkpoint/source')
    if plan.get('runner_max_episode_steps') != 999999999:
        raise ValueError('official evaluation must use native termination')


def discover_outputs(tag):
    """Canonical CLI persists run_tag; recover even if the UI lost its jobs."""
    matches = []
    for root in (EVALUATION / 'V2_1/experiment_14', EVALUATION / 'V_2_1/experment_14'):
        for path in root.glob('task40_eval*/results.json'):
            result = read(path, {})
            if result.get('configuration', {}).get('run_tag') == tag:
                matches.append(path.parent)
    return matches


def controller_alive(output):
    marker = read(output / '.controller.json', {})
    try:
        fields = (Path('/proc') / str(marker['pid']) / 'stat').read_text().rsplit(')', 1)[1].split()
        return fields[0] not in ('Z', 'X') and int(fields[19]) == marker['start_ticks']
    except (KeyError, FileNotFoundError, ProcessLookupError):
        return False


def validate_result(output, checkpoint, tag):
    """Require ten physical native outcomes, exact protocol and closed videos."""
    result = read(output / 'results.json', {})
    config = result.get('configuration', {})
    check_plan(config, checkpoint, tag)
    rows = [json.loads(line) for line in (output / 'episodes.jsonl').read_text().splitlines() if line.strip()]
    if len(rows) != 10 or any(row.get('valid') is not True for row in rows):
        raise ValueError(f'expected ten valid episodes, got {len(rows)} records')
    identities = [(row.get('instance_id'), row.get('seed')) for row in rows]
    if sorted(identities) != [(i, i) for i in range(10)]:
        raise ValueError('episode instance/seed mapping differs from the locked protocol')
    for row in rows:
        native = row.get('native_termination') or {}
        conditions = native.get('termination_conditions') or {}
        if (row.get('evaluation_mode') != 'official-eval'
                or row.get('task_id') != 40 or row.get('gpu_index') != 1
                or row.get('env_idx') != 0
                or not (row.get('terminated') or row.get('truncated'))
                or type(native.get('success')) is not bool
                or not any(isinstance(c, dict) and c.get('done') for c in conditions.values())):
            raise ValueError('episode lacks a native official terminal outcome')
        for field in ('q_score', 'task_progress'):
            value = row.get(field)
            if type(value) not in (int, float) or not math.isfinite(value):
                raise ValueError('missing finite native metric: ' + field)
        video = (output / row.get('video_file', '')).resolve()
        if (not video.is_relative_to(output.resolve()) or video.suffix != '.mp4'
                or not video.is_file() or video.stat().st_size == 0):
            raise ValueError('missing closed episode video')
    return dict(valid_episodes=10, successful_episodes=sum(r['native_termination']['success'] for r in rows),
                mean_q_score=sum(r['q_score'] for r in rows) / 10,
                mean_task_progress=sum(r['task_progress'] for r in rows) / 10,
                output=str(output), episodes=rows, configuration=config)


def _ensure_service(api, state, save, checkpoint):
    services = api('GET', '/api/inference/jobs?device=a100')['jobs']
    on_port = [j for j in services if int(j.get('port', 0)) == PORT and j.get('state') in ACTIVE]
    if len(on_port) > 1 or (on_port and not matching_service(on_port[0], checkpoint)):
        return block(state, save, 'reserved inference port belongs to another service')
    if on_port:
        job = on_port[0]
        state['inference_job'] = job
        save()
        return job.get('state') == 'ready'
    if state.get('inference_intent'):
        return block(state, save, 'inference start intent has no observed process; inspect before retry')
    status = api('GET', '/api/services')
    remote = status.get('remote', {})
    gpu = next((g for g in remote.get('gpus', []) if g['index'] == 0), {})
    if not remote.get('ok') or not gpu or gpu['total_mb'] - gpu['used_mb'] < 24 * 1024:
        state.update(status='waiting_for_inference_capacity', reason='A100 GPU0 needs 24GiB free for model load')
        save()
        return False
    request = dict(device='a100', checkpoint=checkpoint['checkpoint'], gpu_index=0,
                   port=PORT, seed=42, source_dir=checkpoint['source'],
                   python_path=PYTHON, history_mode='normal')
    preview = api('POST', '/api/inference/preview', request)
    if preview.get('checkpoint') != checkpoint['checkpoint'] or preview.get('source_dir') != checkpoint['source']:
        return block(state, save, 'inference preview identity mismatch')
    state.update(inference_intent=request, inference_preview=preview, inference_owned=True)
    save()
    state['inference_start'] = api('POST', '/api/inference/start', request)
    state['status'] = 'waiting_for_inference'
    save()
    return False


def _stop_owned(api, state, save):
    if not state.get('inference_owned') or state.get('inference_cleaned'):
        return True
    job = state.get('inference_job') or state.get('inference_start')
    if not job:
        return block(state, save, 'owned inference process identity missing')
    services = api('GET', '/api/inference/jobs?device=a100')['jobs']
    live = next((j for j in services if j.get('pid') == job.get('pid')
                 and j.get('start_ticks') == job.get('start_ticks') and j.get('state') in ACTIVE), None)
    if live:
        if live.get('state') == 'stopping':
            return False
        evaluations = api('GET', '/api/launch/jobs')['jobs']
        if any(j.get('state') == 'running' and j.get('policy_port') == PORT for j in evaluations):
            state.update(status='waiting_for_inference_dependents')
            save()
            return False
        request = {k: live[k] for k in ('pid', 'port', 'checkpoint', 'start_ticks')}
        request['device'] = 'a100'
        state['inference_stop_intent'] = request
        save()
        state['inference_stop'] = api('POST', '/api/inference/stop', request)
        save()
        return False
    state['inference_cleaned'] = True
    save()
    return True


def tick(api, state, save, checkpoints, allow_subset=False):
    """Advance at most one model stage; caller persists state atomically."""
    if state.get('status') == 'blocked' or state.get('paused'):
        return False
    names = [c['name'] for c in checkpoints]
    if (not names or len(set(names)) != len(names)
            or (not allow_subset and names != ['stage1', 'memory', 'continued'])
            or (allow_subset and any(n not in ('stage1', 'memory', 'continued') for n in names))):
        raise ValueError('evaluation dependencies must contain all three locked checkpoints')
    stages = state.setdefault('models', {})
    for checkpoint in checkpoints:
        item = stages.setdefault(checkpoint['name'], {})
        if item.get('complete'):
            continue
        tag = eval_request(checkpoint)['run_tag']
        item['checkpoint'] = checkpoint
        outputs = discover_outputs(tag)
        if len(outputs) > 1:
            return block(state, save, 'duplicate evaluation outputs for ' + tag)
        if outputs:
            item['output'] = str(outputs[0])
        jobs = api('GET', '/api/launch/jobs')['jobs']
        matching = [j for j in jobs if (item.get('job', {}).get('id') == j.get('id')
                    or ('--run-tag' in j.get('command', '') and tag in shlex.split(j['command'])))]
        if len(matching) > 1:
            return block(state, save, 'duplicate evaluation jobs for ' + tag)
        job = matching[0] if matching else None
        if job:
            item['job'] = job
            if job.get('result_ready'):
                item['output'] = job['output']
        result = read(Path(item['output']) / 'results.json', {}) if item.get('output') else {}
        result_status = result.get('status')
        if job and job.get('state') == 'running':
            state['status'] = 'evaluating_' + checkpoint['name']
            item['valid_episodes'] = result.get('completed_episodes', 0)
            save()
            return False
        if result_status in ('paused', 'pausing', 'interrupted'):
            state.update(status='user_paused', reason='evaluation pause preserved')
            save()
            return False
        if (job and job.get('state') in ('failed', 'stopped')) or result_status == 'failed':
            return block(state, save, 'evaluation failed; preserve original outcome and inspect ' + tag)
        if result_status in ('complete', 'completed') or (job and job.get('state') == 'complete'):
            try:
                evidence = validate_result(Path(item['output']), checkpoint, tag)
            except (ValueError, KeyError, OSError) as error:
                return block(state, save, str(error))
            EVIDENCE.mkdir(parents=True, exist_ok=True)
            evidence_path = EVIDENCE / (checkpoint['run'] + '__official_eval.json')
            evidence_path.write_text(json.dumps(evidence, ensure_ascii=False, indent=2) + '\n')
            item.update(valid_episodes=10, evidence=str(evidence_path))
            if not _stop_owned(api, item, save):
                save()
                return False
            item['complete'] = True
            save()
            return False
        if job or item.get('output'):
            # A crashed/lost workbench process must not trigger a second allocation.
            if not job and not controller_alive(Path(item['output'])):
                return block(state, save, 'evaluation controller disappeared; preserve output: ' + tag)
            state.update(status='waiting_for_evaluation_record', reason=tag)
            save()
            return False
        if item.get('evaluation_intent'):
            return block(state, save, 'evaluation start result uncertain; no duplicate launch: ' + tag)
        if not _ensure_service(api, item, save, checkpoint):
            if item.get('status') == 'blocked':
                return block(state, save, item['reason'])
            state['status'] = item.get('status', 'waiting_for_inference')
            save()
            return False
        # Existing workbench refuses another active evaluation on GPU1; wait here.
        if any(j.get('state') == 'running' and '1' in map(str, j.get('gpus', []))
               and str(j.get('output', '')).startswith(str(EVALUATION)) for j in jobs):
            state['status'] = 'waiting_for_simulation_slot'
            save()
            return False
        request = eval_request(checkpoint)
        preview = api('POST', '/api/launch/preview', request)
        check_plan(preview['plan'], checkpoint, tag)
        item.update(evaluation_intent=request, evaluation_preview=preview)
        save()
        item['job'] = api('POST', '/api/launch/start', request)
        save()
        return False
    state.update(status='completed', valid_episodes=30)
    save()
    return True
