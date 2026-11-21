import importlib.util
from hashlib import sha256
from pathlib import Path
import random
from types import SimpleNamespace

import msgpack
import numpy as np
import pytest
import torch


spec = importlib.util.spec_from_file_location(
    "task40_memory_instrumentation",
    Path(__file__).resolve().parents[1] / "evaluation/scripts/task40_memory_instrumentation.py",
)
trace = importlib.util.module_from_spec(spec)
spec.loader.exec_module(trace)

SESSION = {"session_id": "archive", "env_ids": [0], "episode_ids": ["75410"]}
KEY = ("archive", "0", "75410")


@pytest.fixture
def worker(monkeypatch):
    # Exercise the four-source RNG payload on a CPU test host.
    saved_rng = trace.capture_rng()
    cuda_state = []
    monkeypatch.setattr(torch.cuda, "device_count", lambda: 1)
    monkeypatch.setattr(torch.cuda, "get_rng_state_all", lambda: cuda_state.copy())
    monkeypatch.setattr(torch.cuda, "set_rng_state_all", lambda states: cuda_state.__setitem__(slice(None), states))
    cuda_state.append(torch.tensor([17], dtype=torch.uint8))
    adapter = SimpleNamespace(**{
        name: SimpleNamespace(capacity=2, bank={}, last_written={}, merge_count={}) for name in trace.BANKS
    })
    state = SimpleNamespace(memory=adapter, slots={})
    policy = SimpleNamespace(memory_state=state, model=SimpleNamespace(memory=adapter, device="cpu"))
    rpc = {}
    server = SimpleNamespace(register_endpoint=rpc.__setitem__)
    extension = trace.register_task40_instrumentation(server, policy, None)
    try:
        yield rpc, extension
    finally:
        trace._restore_rng(saved_rng)


def request():
    memory = {"slots": {KEY[:2]: (KEY, 3)}}
    for i, name in enumerate(trace.BANKS):
        feature = torch.tensor([[1 + i, -0.25]], dtype=torch.bfloat16)
        memory[name] = {
            "capacity": 2,
            "bank": {KEY: [(0, feature), (2, feature + 1)]},
            "last_written": {KEY: 2},
            "merge_count": {KEY: 1},
        }
    return {"session": SESSION, "memory": trace.pack_memory(memory),
            "rng": trace.pack_rng(trace.capture_rng()), "decision": 3}


def test_restore_rpc_roundtrips_exact_memory_and_rng(worker):
    rpc, extension = worker
    payload = request()
    random.random()
    np.random.rand()
    torch.rand(1)
    torch.cuda.set_rng_state_all([torch.tensor([99], dtype=torch.uint8)])

    result = rpc["task40_restore_memory"](**payload)
    assert msgpack.unpackb(msgpack.packb(result)) == result
    assert result["decision"] == 3
    assert result["clock"]["key"] == list(KEY)
    for name in trace.BANKS:
        assert result["clock"][name] == {"capacity": 2, "times": [0, 2], "last_written": 2, "merges": 1}
    restored = extension.snapshot(SESSION)
    archived_memory = trace.unpack_memory(payload["memory"])
    installed_memory = trace.unpack_memory(restored["memory"])
    assert trace._clock(installed_memory) == trace._clock(archived_memory)
    for name in trace.BANKS:
        original = archived_memory[name]["bank"][KEY]
        installed = installed_memory[name]["bank"][KEY]
        for (_, expected), (_, actual) in zip(original, installed, strict=True):
            assert actual.dtype == expected.dtype == torch.bfloat16
            assert torch.equal(actual.view(torch.int16), expected.view(torch.int16))
    assert restored["rng"] == payload["rng"]
    for name in ("memory", "rng"):
        assert result[f"{name}_sha256"] == sha256(restored[name]).hexdigest()


@pytest.mark.parametrize("invalid", ["session", "extra_archive_session", "decision", "capacity", "live_session"])
def test_rejects_wrong_archive_or_shared_worker(worker, invalid):
    rpc, extension = worker
    payload = request()
    snapshot = trace.unpack_memory(payload["memory"])
    if invalid == "session":
        payload["session"] = SESSION | {"episode_ids": ["elsewhere"]}
    elif invalid == "extra_archive_session":
        other = ("other", "0", "external")
        snapshot["slots"][other[:2]] = (other, 3)
    elif invalid == "decision":
        payload["decision"] = 4
    elif invalid == "capacity":
        snapshot["cognitive"]["capacity"] = 3
    else:
        other = ("other", "0", "external")
        extension.state.slots[other[:2]] = (other, 1)
    payload["memory"] = trace.pack_memory(snapshot)
    before = extension.snapshot(SESSION)
    before_slots = extension.state.slots.copy()
    with pytest.raises(ValueError):
        rpc["task40_restore_memory"](**payload)
    assert extension.state.slots == before_slots
    assert extension.snapshot(SESSION) == before


def test_failed_rng_install_rolls_back_existing_state(worker):
    rpc, extension = worker
    payload = request()
    restore = rpc["task40_restore_memory"]
    restore(**payload)
    random.random()
    np.random.rand()
    torch.rand(1)
    before = extension.snapshot(SESSION)
    snapshot = trace.unpack_memory(payload["memory"])
    snapshot["perceptual"]["bank"][KEY][0][1].fill_(9)
    python, numpy, _cpu, cuda = trace._unpack_rng(payload["rng"])
    malformed_rng = trace.pack_rng((python, numpy, torch.zeros(1, dtype=torch.uint8), cuda))

    with pytest.raises(RuntimeError, match="RNG state"):
        restore(**(payload | {"memory": trace.pack_memory(snapshot), "rng": malformed_rng}))
    assert extension.snapshot(SESSION) == before
