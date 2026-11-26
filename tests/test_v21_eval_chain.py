import copy
import importlib.util
import json
from pathlib import Path

import pytest

SPEC = importlib.util.spec_from_file_location('v21_eval_chain', Path(__file__).parents[1] / 'scripts/v21_eval_chain.py')
chain = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(chain)


@pytest.fixture
def checkpoints():
    return [dict(name=name, run=f'v2.1__test01__exp14__run00{i}', arm='memory_v1' if name == 'memory' else 'baseline',
                 checkpoint=f'/training/run00{i}/run/checkpoints/step', source=f'/deployments/run00{i}')
            for i, name in enumerate(('stage1', 'memory', 'continued'), 1)]


def plan(checkpoint):
    request = chain.eval_request(checkpoint)
    return dict(mode='single1', preset='official-eval', task_id=40, base_seed=0,
                seed_mode='sequence', episodes_per_gpu=10, action_chunk=8,
                record_video=True, video_steps_per_render=24, video_fps=20,
                run_tag=request['run_tag'], instances={'1': list(range(10))},
                runner_max_episode_steps=999999999,
                policy_identity=dict(checkpoint=checkpoint['checkpoint'], source_dir=checkpoint['source'], binding='observed_process'))


def service(checkpoint):
    return dict(checkpoint=checkpoint['checkpoint'], source_dir=checkpoint['source'],
                history_mode='normal', gpu_index='0', port=5594, state='ready',
                pid=12, start_ticks=120, policy_seed='42')


class API:
    def __init__(self, checkpoint):
        self.checkpoint = checkpoint
        self.services = []
        self.jobs = []
        self.calls = []
        self.start_error = False

    def __call__(self, method, path, body=None):
        self.calls.append((method, path, copy.deepcopy(body)))
        if path == '/api/launch/jobs':
            return dict(jobs=self.jobs)
        if path == '/api/inference/jobs?device=a100':
            return dict(jobs=self.services)
        if path == '/api/services':
            return dict(remote=dict(ok=True, gpus=[dict(index=0, total_mb=81920, used_mb=0)]))
        if path == '/api/inference/preview':
            return dict(body, state='checked')
        if path == '/api/inference/start':
            self.services.append(service(self.checkpoint))
            return self.services[-1]
        if path == '/api/launch/preview':
            return dict(plan=plan(self.checkpoint))
        if path == '/api/launch/start':
            if self.start_error:
                raise TimeoutError('ambiguous response')
            job = dict(id='eval001', state='running', output='/evaluation/task40_eval01')
            self.jobs.append(job)
            return job
        if path == '/api/inference/stop':
            self.services[0]['state'] = 'stopped'
            return dict(state='stopping')
        raise AssertionError((method, path, body))


def test_preview_and_intent_are_durable_before_both_starts(checkpoints, monkeypatch):
    monkeypatch.setattr(chain, 'discover_outputs', lambda _: [])
    state, snapshots = {}, []
    api = API(checkpoints[0])
    save = lambda: snapshots.append(copy.deepcopy(state))
    assert not chain.tick(api, state, save, checkpoints)
    assert snapshots[-2]['models']['stage1']['inference_intent']
    assert not chain.tick(api, state, save, checkpoints)
    assert snapshots[-2]['models']['stage1']['evaluation_intent']
    assert not chain.tick(api, state, save, checkpoints)
    assert sum(path == '/api/inference/start' for _, path, _ in api.calls) == 1
    assert sum(path == '/api/launch/start' for _, path, _ in api.calls) == 1
    assert state['status'] == 'evaluating_stage1'


def test_uncertain_start_does_not_repeat(checkpoints, monkeypatch):
    monkeypatch.setattr(chain, 'discover_outputs', lambda _: [])
    state, api = {}, API(checkpoints[0])
    api.services = [service(checkpoints[0])]
    api.start_error = True
    with pytest.raises(TimeoutError):
        chain.tick(api, state, lambda: None, checkpoints)
    assert state['models']['stage1']['evaluation_intent']
    assert not chain.tick(api, state, lambda: None, checkpoints)
    assert state['status'] == 'blocked'
    assert sum(path == '/api/launch/start' for _, path, _ in api.calls) == 1


def test_foreign_service_is_never_stopped(checkpoints, monkeypatch):
    monkeypatch.setattr(chain, 'discover_outputs', lambda _: [])
    state, api = {}, API(checkpoints[0])
    api.services = [dict(service(checkpoints[0]), checkpoint='/foreign')]
    assert not chain.tick(api, state, lambda: None, checkpoints)
    assert state['status'] == 'blocked'
    assert not any(method == 'POST' for method, _, _ in api.calls)


def test_missing_real_dependencies_rejected(checkpoints):
    with pytest.raises(ValueError, match='all three'):
        chain.tick(None, {}, lambda: None, checkpoints[:1])


def write_result(output, checkpoint):
    output.mkdir(parents=True)
    (output / 'results.json').write_text(json.dumps(dict(status='complete', configuration=plan(checkpoint))))
    rows = []
    for i in range(10):
        video = output / f'videos/{i}.mp4'
        video.parent.mkdir(exist_ok=True)
        video.write_bytes(b'closed-video')
        rows.append(dict(instance_id=i, seed=i, evaluation_mode='official-eval', task_id=40,
                         gpu_index=1, env_idx=0, valid=True, terminated=False, truncated=True,
                         native_termination=dict(success=False, termination_conditions=dict(timeout=dict(done=True))),
                         q_score=0., task_progress=50., video_file=f'videos/{i}.mp4'))
    (output / 'episodes.jsonl').write_text(''.join(json.dumps(r) + '\n' for r in rows))
    return rows


def test_validate_actual_metrics_videos_and_native_termination(tmp_path, checkpoints):
    output = tmp_path / 'task40_eval01'
    write_result(output, checkpoints[0])
    result = chain.validate_result(output, checkpoints[0], chain.eval_request(checkpoints[0])['run_tag'])
    assert result['valid_episodes'] == 10
    assert result['successful_episodes'] == 0
    (output / 'videos/9.mp4').unlink()
    with pytest.raises(ValueError, match='video'):
        chain.validate_result(output, checkpoints[0], chain.eval_request(checkpoints[0])['run_tag'])


@pytest.mark.parametrize('mutation', ['invalid', 'duplicate', 'unnative', 'metric', 'protocol'])
def test_bad_result_blocks(tmp_path, checkpoints, mutation):
    output = tmp_path / 'task40_eval01'
    rows = write_result(output, checkpoints[0])
    if mutation == 'invalid': rows[0]['valid'] = False
    if mutation == 'duplicate': rows[0]['instance_id'] = 1
    if mutation == 'unnative': rows[0]['native_termination'] = None
    if mutation == 'metric': rows[0]['q_score'] = None
    if mutation == 'protocol':
        value = json.loads((output / 'results.json').read_text())
        value['configuration']['action_chunk'] = 16
        (output / 'results.json').write_text(json.dumps(value))
    (output / 'episodes.jsonl').write_text(''.join(json.dumps(r) + '\n' for r in rows))
    with pytest.raises(ValueError):
        chain.validate_result(output, checkpoints[0], chain.eval_request(checkpoints[0])['run_tag'])


def test_only_owned_inference_cleanup_after_real_complete(tmp_path, checkpoints, monkeypatch):
    output = tmp_path / 'task40_eval01'
    write_result(output, checkpoints[0])
    monkeypatch.setattr(chain, 'discover_outputs', lambda tag: [output] if tag.startswith(checkpoints[0]['run']) else [])
    monkeypatch.setattr(chain, 'EVIDENCE', tmp_path / 'evidence')
    api = API(checkpoints[0])
    api.services = [service(checkpoints[0])]
    state = dict(models=dict(stage1=dict(inference_owned=True, inference_job=api.services[0])))
    assert not chain.tick(api, state, lambda: None, checkpoints)
    assert not state['models']['stage1'].get('complete')
    assert api.services[0]['state'] == 'stopped'
    assert not chain.tick(api, state, lambda: None, checkpoints)
    assert state['models']['stage1']['complete']


def test_user_pause_is_preserved(tmp_path, checkpoints, monkeypatch):
    output = tmp_path / 'task40_eval01'
    output.mkdir()
    (output / 'results.json').write_text(json.dumps(dict(status='paused')))
    monkeypatch.setattr(chain, 'discover_outputs', lambda _: [output])
    state, api = {}, API(checkpoints[0])
    assert not chain.tick(api, state, lambda: None, checkpoints)
    assert state['status'] == 'user_paused'
    assert not any(method == 'POST' for method, _, _ in api.calls)
