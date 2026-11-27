"""Exercise launch gates and boundary resume without dispatching real jobs."""
import importlib.util
from pathlib import Path
from unittest.mock import Mock

import pytest


@pytest.fixture
def watch(tmp_path, monkeypatch):
    spec = importlib.util.spec_from_file_location('training_watch', Path(__file__).parents[1]/'scripts/v2_1_training_watch.py')
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    module.DIRECTORY = tmp_path/'evidence'
    module.EXPERIMENT = tmp_path
    module.write(tmp_path/'continued-preflight-request.json', dict(preflight_only=True, run_id=module.PREFLIGHT))
    monkeypatch.setattr(module, 'source_digest', lambda: 'certified')
    return module


def test_changed_source_never_launches(watch):
    watch.api = Mock(return_value={'jobs': []})
    watch.detail = Mock(return_value={'state': {'status': 'completed'}, 'preflight': {'passed': True, 'source_digest': 'different'}})
    with pytest.raises(RuntimeError, match='Source changed'):
        watch.tick({'status': 'running'})
    assert watch.api.call_count == 1


def test_archive_before_purge_before_preview_before_start(watch):
    events = []
    watch.detail = Mock(return_value={'state': {'status': 'completed'}, 'preflight': {'passed': True, 'source_digest': 'certified'}})
    watch.archive = lambda pre: events.append('archive') or '/permanent/preflight.json'

    def api(method, route, body=None):
        events.append(route)
        if method == 'GET':
            return {'jobs': [{'run': watch.PREFLIGHT, 'output': '/temporary'}]}
        if route == 'start':
            assert body['run_id'] == watch.RUN
            assert not body['preflight_only']
            assert body['accept_preflight'] == '/permanent/preflight.json'
        return {'ok': True}

    watch.api = api
    state = {'status': 'running'}
    watch.tick(state)
    assert events == ['jobs?device=a100', 'archive', 'delete', 'preview', 'start']
    assert state['phase'] == 'continued_initializing'


def test_uncertain_mutation_not_repeated(watch):
    watch.api = Mock()
    with pytest.raises(RuntimeError, match='Uncertain'):
        watch.mutation({'actions': {'start': {'status': 'sending'}}}, 'start', 'start', {})
    watch.api.assert_not_called()


def test_memory_preflight_gate_and_correct_arm(watch):
    watch.RUN = 'v2_1__test02__exp01__run002'
    watch.PREFLIGHT = watch.RUN+'__preflight001'
    watch.ARM = 'memory_v1'
    watch.STAGE = 'memory'
    watch.detail = lambda run: watch.api('GET', 'detail?run='+run+'&arm='+watch.ARM)
    watch.archive = lambda pre: '/permanent/memory-preflight.json'
    calls = []

    def api(method, route, body=None):
        calls.append((route, body))
        if route.startswith('jobs'):return {'jobs': [{'run': watch.PREFLIGHT, 'output': '/temporary/memory'}]}
        if route.startswith('detail'):
            assert 'arm=memory_v1' in route
            return {'state': {'status': 'completed'}, 'preflight': {'passed': True, 'source_digest': 'certified'}}
        if route == 'delete':assert body['arm'] == 'memory_v1'
        if route == 'start':assert body['run_id'] == watch.RUN
        return {'ok': True}
    watch.api = api
    state = {'status': 'running'}
    watch.tick(state)
    assert state['phase'] == 'memory_initializing'
    assert [route for route, body in calls][-3:] == ['delete', 'preview', 'start']


def test_failed_memory_calibration_never_starts_formal(watch):
    watch.STAGE = 'memory'
    watch.api = Mock(return_value={'jobs': []})
    watch.detail = Mock(return_value={'state': {'status': 'failed'}, 'preflight': None})
    with pytest.raises(RuntimeError, match='memory preflight failed'):
        watch.tick({'status': 'running'})
    assert watch.api.call_count == 1


@pytest.mark.parametrize('user_stop', [True, False])
def test_user_pause_and_resume_release_gate(watch, user_stop):
    watch.api = Mock(return_value={'jobs': [{'run': watch.RUN}]})
    watch.detail = Mock(return_value={'state': {'status': 'stopped', 'step': 6}, 'config': {}, 'metrics': []})
    watch.remote_inventory = Mock(return_value={'stop': {'stop_after_step': 6}, 'live_pids': [123]})
    state = {'status': 'running'}
    if not user_stop:
        state['resume_proof'] = {'target': 6}
    watch.tick(state)
    assert state['status'] == ('user_paused' if user_stop else 'running')
    assert watch.api.call_count == 1


def test_exact_checkpoint_resume_and_next_update(watch):
    def api(method, route, body=None):
        if method == 'GET':return {'jobs': [{'run': watch.RUN}]}
        assert route == 'resume'
        return {'resume_step': 6}
    watch.api = Mock(side_effect=api)
    current = {'state': {'status': 'stopped', 'step': 6}, 'config': {}, 'metrics': []}
    watch.detail = Mock(return_value=current)
    watch.remote_inventory = Mock(return_value={'stop': {'stop_after_step': 6}, 'live_pids': []})
    state = {'status': 'running', 'resume_proof': {'target': 6}}
    watch.tick(state)
    assert state['resume_proof']['resumed']
    assert watch.api.call_count == 3
    current['state'] = {'status': 'training', 'step': 7}
    watch.tick(state)
    assert state['resume_proof']['next_update_verified']
