"""Only real accepted task/weight identities can advertise runtime readiness."""
import hashlib
import json
from pathlib import Path
import sys

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))
from v1_2_groot_baseline import b1k2026_capabilities as cap


@pytest.fixture
def accepted(tmp_path, monkeypatch):
    config = {key: tmp_path / key for key in ('python','upstream','behavior_python','behavior_root','checkpoint')}
    config['checkpoint_sha256'] = 'verified-weight-digest'
    monkeypatch.setattr(cap, 'inspect_upstream', lambda root: dict(revision='groot-pinned'))
    monkeypatch.setattr(cap.subprocess, 'check_output', lambda argv, **kw: 'behavior-pinned' if 'rev-parse' in argv else '')
    catalog = config['behavior_root'] / 'docs/challenge/task_data.json'
    catalog.parent.mkdir(parents=True)
    catalog.write_text(json.dumps({'tasks': [dict(id=t, name=t, instruction=t) for t in
        ['turning_on_radio', 'make_microwave_popcorn', 'another_official_task']]}))
    output = tmp_path / 'evaluation/V_1_2/official/public_test_instance_18'
    (output / 'json').mkdir(parents=True)
    (output / 'videos').mkdir()
    result = b'{"success":false,"q_score":{"final":0.0}}'
    (output / 'json/radio.json').write_bytes(result)
    (output / 'videos/radio.mp4').write_bytes(b'fixture-video')
    job = dict(job_id='accepted-radio', kind=cap.B1K2026, task_id='turning_on_radio',
        checkpoint_sha256=config['checkpoint_sha256'], policy_identity=dict(
            checkpoint=dict(path=str(config['checkpoint'])), source=dict(root=str(config['upstream'])),
            argv=[str(config['python'])]),
        evaluator=dict(argv=[str(config['behavior_python'])], cwd=str(config['behavior_root'])))
    (output / 'job.json').write_text(json.dumps(job))
    (output / 'runner.json').write_text(json.dumps(dict(state='complete', evaluator_returncode=0)))
    evidence = tmp_path / 'docs/v1_2/evidence/native-acceptance-2026-11-28.json'
    evidence.parent.mkdir(parents=True)
    evidence.write_text(json.dumps(dict(native_runtime_accepted=True, groot_revision='groot-pinned',
        behavior_revision='behavior-pinned', checkpoint_sha256=config['checkpoint_sha256'], runs=[dict(
            job_id='accepted-radio', output=str(output.relative_to(tmp_path)),
            native_json_sha256=hashlib.sha256(result).hexdigest())])))
    return tmp_path, config, output


def test_runtime_completion_does_not_claim_task_success_or_training(accepted):
    root, config, _ = accepted
    result = cap.options(root, root / 'evaluation', config, True)
    assert result['launch_ready'] is True
    assert result['task_performance_accepted'] is False
    assert result['verified_job_ids'] == ['accepted-radio']
    assert result['checkpoint_whitelist'][0]['tasks'] == ['turning_on_radio']
    assert [t['id'] for t in result['task_catalog'] if t['launch_ready']] == ['turning_on_radio']
    assert not result['capabilities']['training']['available']
    assert not result['capabilities']['self_trained_evaluation']['available']


@pytest.mark.parametrize('mismatch', ['checkpoint_sha256', 'checkpoint', 'upstream', 'python', 'behavior_python'])
def test_changed_runtime_is_not_implicitly_accepted(accepted, mismatch):
    root, config, _ = accepted
    config[mismatch] = 'different-digest' if mismatch.endswith('sha256') else root / 'different'
    result = cap.options(root, root / 'evaluation', config, True)
    assert not result['launch_ready']
    assert result['checkpoint_whitelist'] == []


def test_modified_native_result_cannot_supply_acceptance(accepted):
    root, config, output = accepted
    (output / 'json/radio.json').write_text('{"success":true}')
    assert not cap.options(root, root / 'evaluation', config, True)['launch_ready']


def test_paths_without_native_acceptance_do_not_claim_availability(accepted):
    root, config, output = accepted
    (output / 'runner.json').write_text(json.dumps(dict(state='failed', evaluator_returncode=1)))
    result = cap.options(root, root / 'evaluation', config, True)
    assert result['environment_present'] is True
    assert not result['launch_ready']
    assert all(not t['launch_ready'] for t in result['task_catalog'])


def test_hidden_task_without_public_instruction_stays_in_catalog(accepted):
    root, config, _ = accepted
    catalog = config['behavior_root'] / 'docs/challenge/task_data.json'
    catalog.write_text(json.dumps({'tasks': [dict(id='hidden_task', name='Hidden task')]}))
    task = cap.options(root, root / 'evaluation', config, True)['task_catalog'][0]
    assert task['instruction'] is None
    assert not task['launch_ready']


@pytest.fixture
def engineering(accepted):
    root, config, _ = accepted
    run = 'v0_0__test04__exp01__run001'
    training = root / 'training/b1k2026' / run
    checkpoint = training / 'native/checkpoint-4'
    checkpoint.mkdir(parents=True)
    output = root / 'evaluation/b1k2026/V_0_0/self_trained' / (run + '__step000004__task40__instance00__attempt01')
    (output / 'json').mkdir(parents=True)
    (output / 'videos').mkdir()
    records = {
        training / 'job.json': dict(run_id=run, tasks=[40], formal_training=False),
        training / 'status.json': dict(state='complete', step=4),
        training / 'metrics.jsonl': dict(step=4, parameter_delta={'weight': 1e-5}),
        training / 'loss.jsonl': dict(step=4, loss=1.204),
        training / 'resume_events.jsonl': dict(step=2, optimizer_entries=537),
        checkpoint / 'b1k2026_training.json': dict(run_id=run, optimizer_step=4,
            checkpoint_sha256='step4-digest', source_revision='groot-pinned'),
        output / 'job.json': dict(job_id='self-trained-job', checkpoint_sha256='step4-digest',
            task_id='make_microwave_popcorn', policy_identity=dict(argv=[str(config['python'])])),
        output / 'runner.json': dict(state='complete', evaluator_returncode=0),
        output / 'json/task40.json': dict(success=False, q_score={'final': 0.0})}
    for path, value in records.items():
        path.write_text(json.dumps(value))
    (output / 'videos/task40.mp4').write_bytes(b'fixture-video')
    evidence = dict(accepted=True, benchmark='b1k2026', version=cap.B1K2026_VERSION,
        tasks=[40], optimizer_steps=4, run_id=run, upstream=str(config['upstream']),
        python=str(config['python']), checkpoint=str(checkpoint), checkpoint_sha256='step4-digest',
        evaluation_output=str(output.relative_to(root)), evaluation_job_id='self-trained-job',
        native_json='json/task40.json', files_sha256={str(p.relative_to(root)):
            hashlib.sha256(p.read_bytes()).hexdigest() for p in records})
    (root / 'docs/v1_2/evidence/training-acceptance-v031.json').write_text(json.dumps(evidence))
    return root, config, training, checkpoint, output


def test_real_engineering_evidence_enables_only_scoped_capabilities(engineering):
    root, config, _, _, _ = engineering
    result = cap.options(root, root / 'evaluation', config, True)
    for name in ('training', 'self_trained_inference', 'self_trained_evaluation', 'strict_training_resume'):
        assert result['capabilities'][name]['available']
        assert result['capabilities'][name]['engineering_only']
        assert result['capabilities'][name]['tasks'] == ['make_microwave_popcorn']
    assert not result['formal_training_available']
    assert not result['task_performance_accepted']
    assert not result['capabilities']['multi_environment']['available']
    assert not result['capabilities']['evaluation_resume']['available']
    assert result['checkpoint_whitelist'][1]['model_role'] == 'self_trained'


@pytest.mark.parametrize('artifact', ['resume_events.jsonl', 'metrics.jsonl', 'status.json'])
def test_changed_training_record_revokes_engineering_acceptance(engineering, artifact):
    root, config, training, _, _ = engineering
    (training / artifact).write_text('{}')
    assert cap.verified_engineering_run(root, config['upstream'], config['python']) is None


def test_incomplete_eval_or_changed_interpreter_cannot_enable_engineering(engineering):
    root, config, _, _, output = engineering
    assert cap.verified_engineering_run(root, config['upstream'], root / 'other-python') is None
    (output / 'runner.json').write_text(json.dumps(dict(state='failed', evaluator_returncode=1)))
    assert cap.verified_engineering_run(root, config['upstream'], config['python']) is None
