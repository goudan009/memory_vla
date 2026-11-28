import json
from pathlib import Path
import sys

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "evaluation" / "scripts"))
from groot_web_launch import LaunchManager
from v1_2_groot_baseline import contracts as c


def test_task_language_reaches_official_wrapper_and_restores_module():
    from types import SimpleNamespace
    from v12_b1k2026_server import task_language

    class OfficialWrapper:
        def __init__(self, policy, text_prompt="pick up the object and place it on the table"):
            self.policy, self.text_prompt = policy, text_prompt

    module = SimpleNamespace(B1KPolicyWrapper=OfficialWrapper)
    prompt = "Turn on the radio receiver that's on the table in the living room."
    assert module.B1KPolicyWrapper("checkpoint").text_prompt != prompt
    with task_language(module, prompt):
        policy = module.B1KPolicyWrapper(policy="checkpoint")
        assert policy.text_prompt == prompt
        assert policy.policy == "checkpoint"
    assert module.B1KPolicyWrapper is OfficialWrapper


def _fixture_checkpoint(root):
    (root / "config.json").write_text(json.dumps({
        "model_type": "Gr00tN1d7", "action_horizon": 40,
        "max_action_dim": 132, "max_state_dim": 132,
    }))
    modes = {
        k: {"modality_keys": list(v), "delta_indices": list(range(16)) if k == "action" else [0]}
        for k, v in {"state": c.STATE_GROUPS_2026, "action": c.ACTION_GROUPS,
                     "video": ["head", "left_wrist", "right_wrist"],
                     "language": ["annotation.human.task_description"]}.items()
    }
    modes["state"]["modality_keys"] = list(c.STATE_GROUPS_2026)
    modes["action"]["action_configs"] = [
        {"rep": "RELATIVE" if k in ("torso", "left_arm", "right_arm") else "ABSOLUTE",
         "type": "NON_EEF", "format": "DEFAULT",
         "state_key": k if k in ("torso", "left_arm", "right_arm") else None,
         "is_gripper": "gripper" in k}
        for k in c.ACTION_GROUPS
    ]
    (root / "processor_config.json").write_text(json.dumps({
        "processor_class": "Gr00tN1d7Processor", "processor_kwargs": {
            "modality_configs": {"new_embodiment": modes}, "max_action_horizon": 40,
            "max_action_dim": 132, "max_state_dim": 132, "use_relative_action": True,
        }}))
    stats = {}
    for kind, layout in (("state", c.STATE_GROUPS_2026), ("action", c.ACTION_GROUPS),
                         ("relative_action", {k: c.ACTION_GROUPS[k] for k in ("torso", "left_arm", "right_arm")})):
        stats[kind] = {k: {"min": [-1] * (b - a), "max": [1] * (b - a)} for k, (a, b) in layout.items()}
    (root / "statistics.json").write_text(json.dumps({"new_embodiment": stats}))
    (root / "model.safetensors").write_bytes(b"fixture")


def test_v12_preview_binds_native_route(tmp_path, monkeypatch):
    upstream = tmp_path / "upstream"
    (upstream / "examples/b1k").mkdir(parents=True)
    ckpt = tmp_path / "checkpoint"
    ckpt.mkdir()
    _fixture_checkpoint(ckpt)
    python = tmp_path / "python"
    python.touch()
    behavior_python = tmp_path / "behavior-python"
    behavior_python.touch()
    monkeypatch.setattr(c, "inspect_upstream", lambda root: {"root": str(root), "revision": c.FORK_REVISION})
    monkeypatch.setenv("V12_GROOT_ROOT", str(upstream))
    monkeypatch.setenv("V12_GROOT_PYTHON", str(python))
    monkeypatch.setenv("V12_BEHAVIOR_ROOT", str(tmp_path / "behavior"))
    monkeypatch.setenv("V12_BEHAVIOR_PYTHON", str(behavior_python))
    monkeypatch.setenv("V12_CHECKPOINT", str(ckpt))
    monkeypatch.setenv("V12_CHECKPOINT_SHA256", c.checkpoint_digest(ckpt))
    (tmp_path / "behavior").mkdir()
    manager = LaunchManager(tmp_path, log_root=tmp_path / "logs")
    request = {"version": "V_1_2", "protocol": c.B1K2026, "instance": 3,
               "rollouts": 1, "gpu_index": 1, "notes": "V1.2 / N1.7 / B1K2026: official radio, RTC off"}
    prepared = manager.prepare(request)
    plan = prepared["plan"]
    assert plan["kind"] == c.B1K2026 and plan["instances"] == {"1": [3]}
    assert plan['benchmark'] == 'b1k2026'
    assert plan['version'] == 'V_0_0' and plan['display_version'] == 'V0_0'
    assert plan['method_family'] == 'V0'
    assert Path(plan['output']).is_relative_to(manager.evaluation/'b1k2026/V_0_0/official')
    assert plan["notes"].startswith("V1.2 / N1.7")
    assert "run_groot_eval.sh" not in prepared["command"]
    assert "serve_b1k.py" in json.dumps(plan["policy_identity"])
    with pytest.raises(ValueError, match="B1K-2026"):
        manager.prepare({**request, "protocol": c.B1K2025})

    metadata = tmp_path / "tasks.jsonl"
    prompt = "Turn on the radio receiver that's on the table in the living room."
    metadata.write_text(json.dumps(dict(task_index=0, task_name="turning_on_radio", task=prompt)) + "\n")
    monkeypatch.setenv("V12_TASK_METADATA", str(metadata))
    (Path(plan['output']) / 'job.json').write_text('original attempt retained')
    corrected = manager.prepare({**request, 'attempt': 2, 'language_mode': 'task_description'})['plan']
    assert corrected['output'].endswith('b1k2026__v0_0__official__instance03__attempt02')
    assert corrected['language']['text_prompt'] == prompt
    assert corrected['policy_identity']['language'] == corrected['language']
    assert (Path(plan['output']) / 'job.json').read_text() == 'original attempt retained'
    assert plan['language']['mode'] == 'official_default'
    with pytest.raises(ValueError, match='language_mode'):
        manager.prepare({**request, 'language_mode': 'arbitrary'})
    metadata.write_text(json.dumps(dict(task_index=0, task_name='turning_on_radio', task='wrong task')) + '\n')
    with pytest.raises(ValueError, match='Task0'):
        manager.prepare({**request, 'attempt': 2, 'language_mode': 'task_description'})
    with pytest.raises(ValueError, match='exclusively'):
        manager.prepare({**request, 'version': 'V0_0', 'benchmark': 'b1k'})


def test_completed_native_run_survives_primary_workbench_reload(tmp_path):
    from view_results import restore_evaluation_jobs

    manager = LaunchManager(tmp_path, log_root=tmp_path / "logs")
    manager.log_root.mkdir()
    (manager.log_root / "native-radio.log").write_text("native evaluator finished\n")
    output = manager.evaluation / "V_1_2/official/public_test_instance_18"
    output.mkdir(parents=True)
    plan = dict(job_id="native-radio", output=str(output), kind=c.B1K2026,
                version="V_1_2", task_id="turning_on_radio", gpus=[0],
                started_at="2026-11-28 13:29:55", notes="V1.2 / N1.7 / B1K2026: RTC off")
    (output / "job.json").write_text(json.dumps(plan))
    (output / "runner.json").write_text(json.dumps(dict(
        state="complete", evaluator_returncode=0, evaluator_argv=["python", "evaluator.py"])))
    (output / "json").mkdir()
    (output / "json/radio.json").write_text(json.dumps(dict(
        instance_id=319, steps=120, success=False, q_score=dict(final=0.5))))
    restore_evaluation_jobs(manager, [])
    restore_evaluation_jobs(manager, [])
    jobs = manager.snapshot()
    assert len(jobs) == 1
    assert jobs[0]["state"] == "complete"
    assert jobs[0]["benchmark"] == 'b1k2026'
    assert jobs[0]["notes"] == plan["notes"]
    assert jobs[0]["native_metrics"] == [dict(instance_id=319, steps=120, success=False, q_score=0.5)]
    assert jobs[0]["native_episode_keys"] == ['json/radio.json']
    assert any(item["path"] == "json/radio.json" for item in jobs[0]["artifacts"])


@pytest.mark.parametrize('relative', [
    'V_1_2/official/public_test_instance_18_attempt_02',
    'b1k2026/V_0_0/official/b1k2026__v0_0__official__instance18__attempt02',
])
def test_same_instance_retry_artifact_is_readable_over_http(tmp_path, relative):
    import threading
    import urllib.request
    import view_results as viewer

    manager = LaunchManager(tmp_path)
    output = manager.evaluation / relative
    (output / 'json').mkdir(parents=True)
    result = dict(instance_id=319, steps=3225, success=False, q_score=dict(final=0.0))
    (output / 'json/radio.json').write_text(json.dumps(result))

    class Handler(viewer.Handler):
        launch_manager = manager

    with viewer.ThreadingHTTPServer(('127.0.0.1', 0), Handler) as server:
        thread = threading.Thread(target=server.serve_forever, daemon=True)
        thread.start()
        try:
            url = f'http://127.0.0.1:{server.server_port}/api/launch/artifact?run={output.name}&path=json%2Fradio.json'
            with urllib.request.urlopen(url, timeout=5) as response:
                assert json.load(response) == result
        finally:
            server.shutdown()
            thread.join(timeout=5)


def _training_checkpoint(checkpoint):
    _fixture_checkpoint(checkpoint)
    for name in c.TRAINING_STATE_FILES:
        (checkpoint/name).write_text(json.dumps({'global_step':4} if name=='trainer_state.json' else {}))
    identity=dict(version='V_0_0',benchmark='b1k2026',run_id='v0_0__test99__exp99__run999',
        complete=True,parameter_update_verified=True,optimizer_step=4,tasks=[40],
        source_revision=c.FORK_REVISION,recipe_sha256='a'*64,data_manifest_sha256='b'*64,
        language_mode='task_description',task_prompts={'40':'Make microwave popcorn.'},
        checkpoint_sha256=c.checkpoint_digest(checkpoint),
        training_state_sha256={name:c.file_sha256(checkpoint/name) for name in c.TRAINING_STATE_FILES})
    (checkpoint/'b1k2026_training.json').write_text(json.dumps(identity))
    return identity


def test_self_trained_task40_route_does_not_unlock_official_radio(tmp_path,monkeypatch):
    upstream=tmp_path/'upstream';upstream.mkdir()
    behavior=tmp_path/'behavior';behavior.mkdir()
    python=tmp_path/'python';python.touch()
    checkpoint=tmp_path/'training/b1k2026/v0_0__test99__exp99__run999/native/checkpoint-4'
    checkpoint.mkdir(parents=True)
    identity=_training_checkpoint(checkpoint)
    monkeypatch.setattr(c,'inspect_upstream',lambda root:dict(root=str(root),revision=c.FORK_REVISION))
    manager=LaunchManager(tmp_path,log_root=tmp_path/'logs')
    config=dict(python=python,upstream=upstream,behavior_python=python,behavior_root=behavior,
                checkpoint=checkpoint,checkpoint_sha256=identity['checkpoint_sha256'])
    monkeypatch.setattr(manager,'_v12_config',lambda:config)
    request=dict(version='V0_0',benchmark='b1k2026',protocol=c.B1K2026,task='make_microwave_popcorn',
        model_role='self_trained',engineering_only=True,checkpoint=str(checkpoint),
        checkpoint_sha256=identity['checkpoint_sha256'],instance=0,rollouts=1,gpu_index=1,
        policy_host='127.0.0.1',policy_port=8040,notes='V0_0 / N1.7: self-trained engineering checkpoint')
    plan=manager.prepare(request)['plan']
    assert plan['official'] is False and plan['task_id']=='make_microwave_popcorn'
    assert plan['language']['text_prompt']=='Make microwave popcorn.'
    assert plan['policy_identity']['training_identity']==identity
    assert plan['evaluator']['group']=='self_trained'
    assert '/self_trained/' in plan['output']
    assert '/b1k2026/V_0_0/' in plan['output']
    assert plan['version'] == 'V_0_0'
    with pytest.raises(ValueError,match='官方 radio'):
        manager.prepare({**request,'model_role':'official'})
    with pytest.raises(ValueError,match='not in this training'):
        manager.prepare({**request,'task':'turning_on_radio'})
    with pytest.raises(ValueError,match='language'):
        manager.prepare({**request,'language_mode':'official_default'})
    (checkpoint/'optimizer.pt').write_bytes(b'changed')
    with pytest.raises(ValueError,match='training state hash'):
        manager.prepare(request)


@pytest.mark.parametrize('version,prefix,directory', [
    ('V_1_2', 'v1_2', 'V_1_2'),
    ('V_0_0', 'v0_0', 'b1k2026/V_0_0'),
])
def test_self_trained_artifacts_and_restoration_use_native_outputs(tmp_path, version, prefix, directory):
    import threading
    import urllib.request
    import view_results as viewer
    manager=LaunchManager(tmp_path,log_root=tmp_path/'logs')
    manager.log_root.mkdir()
    (manager.log_root/'trained.log').touch()
    name=f'{prefix}__test99__exp99__run999__step000004__task40__instance00__attempt01'
    output=manager.evaluation/directory/'self_trained'/name
    (output/'json').mkdir(parents=True)
    (output/'videos').mkdir()
    native=dict(instance_id=5,steps=12,success=False,q_score=dict(final=.2))
    (output/'json/popcorn.json').write_text(json.dumps(native))
    (output/'videos/popcorn.mp4').write_bytes(b'fixture-video')
    (output/'job.json').write_text(json.dumps(dict(job_id='trained',output=str(output),gpus=[1],
        kind=c.B1K2026,version=version,official=False,started_at='fixture',task_id='make_microwave_popcorn',
        display_version='V0_0' if version=='V_0_0' else None,
        method_family='V0' if version=='V_0_0' else None)))
    (output/'runner.json').write_text(json.dumps(dict(state='complete',evaluator_argv=['python','evaluator.py'])))
    viewer.restore_evaluation_jobs(manager,[])
    jobs=manager.snapshot()
    assert jobs[0]['native_episode_keys']==['json/popcorn.json']
    assert jobs[0]['official'] is False
    assert jobs[0]['version'] == version
    if version == 'V_0_0':
        assert jobs[0]['display_version'] == 'V0_0' and jobs[0]['method_family'] == 'V0'

    class Handler(viewer.Handler):
        launch_manager=manager

    with viewer.ThreadingHTTPServer(('127.0.0.1',0),Handler) as server:
        thread=threading.Thread(target=server.serve_forever,daemon=True);thread.start()
        try:
            for artifact in jobs[0]['artifacts']:
                with urllib.request.urlopen(f'http://127.0.0.1:{server.server_port}'+artifact['url'],timeout=5) as response:
                    assert response.read()==(output/artifact['path']).read_bytes()
        finally:
            server.shutdown();thread.join(timeout=5)
