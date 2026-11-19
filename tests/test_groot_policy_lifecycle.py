import importlib.util
from pathlib import Path
import time
from types import SimpleNamespace

import pytest


def test_only_ended_environments_reset_and_requests_keep_session_identity():
    path = Path(__file__).resolve().parents[1] / "scripts/groot_policy_lifecycle.py"
    spec = importlib.util.spec_from_file_location("lifecycle", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    class Policy:
        def __init__(self):
            self.resets = []
            self.requests = []

        def reset(self, options=None):
            self.resets.append(options)

        def get_action(self, obs, options=None):
            self.requests.append((obs, options))
            return obs, {}

    base = Policy()
    client = module.EpisodePolicy(base, 2, session_id="gpu0")
    client.reset()
    obs = object()
    assert client.get_action(obs)[0] is obs
    client.reset_ended([False, True])
    client.reset_ended([False, False])
    client.get_action(obs)
    assert len(base.resets) == 2
    assert base.resets[-1]["env_ids"] == [1]
    assert base.resets[-1]["episode_ids"] == ["gpu0/env1/episode1"]
    assert base.requests[-1][1]["episode_ids"] == ["gpu0/env0/episode0", "gpu0/env1/episode1"]
    other = module.EpisodePolicy(Policy(), 1, session_id="gpu1")
    assert other._options([0])["episode_ids"] != client._options([0])["episode_ids"]


def test_disconnected_rpc_times_out_and_socket_closes_without_lingering():
    zmq = pytest.importorskip("zmq")
    path = Path(__file__).resolve().parents[1] / "scripts/groot_policy_lifecycle.py"
    spec = importlib.util.spec_from_file_location("lifecycle_timeout", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    context = zmq.Context()
    socket = context.socket(zmq.REQ)
    # No server: simulate a policy service disappearing during scene loading.
    socket.connect("inproc://missing-evaluation-policy")
    def reset(options=None):
        socket.send_json(options)
        return socket.recv_json()
    policy = SimpleNamespace(socket=socket, timeout_ms=50, host="fixture", port=0, reset=reset)
    try:
        client = module.EpisodePolicy(policy, 1)
        started = time.monotonic()
        with pytest.raises(zmq.Again):
            client.reset()
        assert time.monotonic() - started < 1
        assert socket.getsockopt(zmq.RCVTIMEO) == 50
        assert socket.getsockopt(zmq.SNDTIMEO) == 50
        assert socket.getsockopt(zmq.LINGER) == 0
    finally:
        socket.close(linger=0)
        context.term()


def test_task40_memory_diagnostics_reuses_policy_info_and_episode_identity(tmp_path,monkeypatch):
    import json
    path=Path(__file__).resolve().parents[1]/'scripts/groot_policy_lifecycle.py'
    spec=importlib.util.spec_from_file_location('lifecycle_memory',path)
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    monkeypatch.setenv('GROOT_TASK40_TRACE','1')
    monkeypatch.setenv('GROOT_EVAL_OUTPUT_DIR',str(tmp_path))
    (tmp_path/'run_meta.json').write_text(json.dumps({'task_id':40,'gpu_index':1}))
    (tmp_path/'runtime_env0.json').write_text(json.dumps({'instance_id':3,'seed':18003,'episode_number':0}))
    requests=[]
    def action(obs,options=None):
        requests.append(options)
        return obs, {'memory_diagnostics':{'observations':[{'timestep':0,'read_source':['a',0,0]}]}}
    client=module.EpisodePolicy(SimpleNamespace(get_action=action),1,session_id='test')
    obs=object();assert client.get_action(obs)[0] is obs
    assert requests[-1]['memory_diagnostics'] is True
    assert requests[-1]['memory_action_controls'] is True
    saved=tmp_path/'stages/memory_gpu1_env0.jsonl'
    row=json.loads(saved.read_text())
    assert row['instance_id']==3 and row['seed']==18003 and row['env_idx']==0
    assert row['memory']['timestep']==0 and row['session_id']=='test'
    client.get_action(obs)
    assert 'memory_diagnostics' not in requests[-1]
    assert 'memory_action_controls' not in requests[-1]
    assert len(saved.read_text().splitlines())==1
