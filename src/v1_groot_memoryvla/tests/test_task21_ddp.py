"""Two Gloo ranks must match one process at the same global batch and history."""

from datetime import timedelta
import json
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from torch import nn
from torch.nn.parallel import DistributedDataParallel

from v1_groot_memoryvla.experiment.task21_ddp import EpisodePlan, TrainableStep, backward_windows, restore_rank, resume_manifest
from v1_groot_memoryvla.experiment.task21_formal import window_context, memory_runtime
from v1_groot_memoryvla.experiment.first_training import slice_features
from v1_groot_memoryvla.model.memory.adapter import MemoryAdapter
from v1_groot_memoryvla.tests.test_memory_v1 import features
from v1_groot_memoryvla.experiment.task21_async import ExistingProcess, shared_stop_target, take_over_async


def test_validation_queue_is_published_when_parent_directory_does_not_exist(tmp_path):
    from v1_groot_memoryvla.experiment.task21_formal import atomic_json
    path = tmp_path / "memory_v1/validation_queue/step-001000.json"
    atomic_json(path, {"step": 1000, "checkpoint": "/immutable/step-001000"})
    assert json.loads(path.read_text())["step"] == 1000
    assert not path.with_suffix(".partial.json").exists()


class Head(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = nn.Linear(32, 4)

    def forward(self, context, actions):
        prediction = self.linear(context["backbone_features"].mean(1))
        return {"loss": (prediction-actions["target"]).square().mean()}


def core():
    torch.manual_seed(52)
    return TrainableStep(SimpleNamespace(memory=MemoryAdapter(vlm_dim=32, perceptual_dim=32), action_head=Head())).train()


def sequence(slot, window):
    if slot == 0 and window == 2:
        return None
    torch.manual_seed(300+slot*10+window)
    tensors, ids = features(32)
    return {"features": tensors, "input_ids": ids, "actions": {"target": torch.randn(32, 4)},
            "episode_key": f"episode-{slot}", "episode_start": window == 0,
            "episode_end": window == (1 if slot == 0 else 2),
            "timesteps": list(range(window*32, (window+1)*32))}


def _rank(rank, temporary):
    torch.set_num_threads(1)
    dist.init_process_group("gloo", init_method="file://"+str(Path(temporary) / "rendezvous"),
                            rank=rank, world_size=2, timeout=timedelta(seconds=90))
    try:
        model = core()
        ddp = DistributedDataParallel(model, broadcast_buffers=False, find_unused_parameters=True)
        optimizer = torch.optim.SGD(model.parameters(), lr=.01)
        history = []
        for window in range(3):
            optimizer.zero_grad(set_to_none=True)
            backward_windows(ddp, [sequence(rank, window)], [900+rank*100],
                             global_count=2 if window < 2 else 1, device="cpu")
            history.append({"last_written": dict(model.memory.cognitive.last_written),
                            "merges": dict(model.memory.cognitive.merge_count)})
            optimizer.step()
            if window == 0:
                from v1_groot_memoryvla.experiment import task21_ddp
                def save_test_model(model, processor, directory):
                    directory.mkdir(parents=True, exist_ok=True)
                    torch.save(model.state_dict(), directory / "weights.pt")
                task21_ddp.save_model = save_test_model
                save_root = Path(temporary) / "saved"
                (save_root / "checkpoints").mkdir(parents=True, exist_ok=True)
                task21_ddp.checkpoint(model, None, optimizer, save_root,
                    SimpleNamespace(global_windows_per_step=2, seed=42), 1, 0, 1,
                    control_group=dist.group.WORLD)
        torch.save({"weights": model.state_dict(), "history": history}, Path(temporary) / f"rank{rank}.pt")
    finally:
        dist.destroy_process_group()


def test_two_ranks_equal_single_process_with_history_and_an_empty_tail(tmp_path):
    mp.start_processes(_rank, args=(str(tmp_path),), nprocs=2, join=True, start_method="spawn")
    torch.set_num_threads(1)
    model = core()
    optimizer = torch.optim.SGD(model.parameters(), lr=.01)
    for window in range(3):
        optimizer.zero_grad(set_to_none=True)
        count = 2 if window < 2 else 1
        for slot in range(2):
            seq = sequence(slot, window)
            if seq is None:
                continue
            for start in range(0, 32, 4):
                loss = model(slice_features(seq["features"], start, start+4, "cpu"),
                    seq["input_ids"][start:start+4], window_context(seq, start, start+4),
                    slice_features(seq["actions"], start, start+4, "cpu"), 900+slot*100+start)
                (loss*4/32/count).backward()
        optimizer.step()
    ranks = [torch.load(tmp_path / f"rank{rank}.pt", weights_only=True) for rank in range(2)]
    saved = tmp_path / "saved/checkpoints"
    assert json.loads((saved / "latest.json").read_text())["step"] == 1
    for rank in range(2):
        state = torch.load(saved / f"latest_rank-{rank:03d}.pt", weights_only=True)
        assert state["memory_runtime"][1]["merge_count"] == {f"episode-{rank}": 16}
    for name, expected in model.state_dict().items():
        torch.testing.assert_close(ranks[0]["weights"][name], ranks[1]["weights"][name], rtol=0, atol=0)
        torch.testing.assert_close(ranks[0]["weights"][name], expected, rtol=2e-5, atol=2e-7)
    assert ranks[0]["history"][0]["merges"] == {"episode-0": 16}
    assert ranks[1]["history"][1]["merges"] == {"episode-1": 48}
    assert ranks[0]["history"][1]["last_written"] == {}
    assert ranks[1]["history"][2]["last_written"] == {}
    difference = max(float((ranks[0]["weights"][n]-v).abs().max()) for n, v in model.state_dict().items())
    print(json.dumps({"cpu_ddp_equivalence": "passed", "world_size": 2,
        "max_parameter_difference_from_single_process": difference,
        "rank_parameters_bitwise_equal": True, "cross_window_history": True,
        "empty_tail_rank": True, "merges_verified": [16, 48]}))


def test_episode_plan_has_fixed_ownership_no_drop_and_chronological_windows():
    by_episode = {10: [0, 1, 2], 11: [3, 4], 12: [5], 13: [6, 7, 8, 9, 10]}
    groups = [(0, e, i*256) for e, indices in by_episode.items() for i, _ in enumerate(indices)]
    dataset = SimpleNamespace(by_episode=by_episode, groups=groups,
        order=lambda seed: [indices[w] for w in range(5) for indices in by_episode.values() if w < len(indices)])
    plan = EpisodePlan(dataset, slots=2, seed=42)
    seen = {e: [] for e in by_episode}
    all_indices = []
    for position in range(plan.steps):
        batch = plan.batch(position)
        for slot, index in enumerate(batch):
            if index is not None:
                eid = groups[index][1]
                assert plan.owner[eid] == slot
                seen[eid].append(index)
                all_indices.append(index)
    assert seen == by_episode
    assert sorted(all_indices) == list(range(len(groups)))
    # One GPU handles slots[0,1]; two GPUs handle one slot each. Same global batches.
    assert [plan.batch(i) for i in range(plan.steps)] == [EpisodePlan(dataset, 2, 42).batch(i) for i in range(plan.steps)]


class MaskedHead(Head):
    def forward(self, context, actions):
        prediction = self.linear(context["backbone_features"].mean(1))
        mask = actions["action_mask"]
        return {"loss": ((prediction-actions["target"]).square()*mask).sum()/mask.sum()}


def tail_core():
    torch.manual_seed(52)
    return TrainableStep(SimpleNamespace(memory=MemoryAdapter(32, 32, capacity=64,
        conditioning="direct"), action_head=MaskedHead())).train()


def tail_sequence(rank):
    torch.manual_seed(300+rank)
    length = 5 if rank == 0 else 1
    tensors, ids = features(length)
    mask = torch.ones(length, 4)
    mask[-1, 1:] = 0
    return {"features": tensors, "input_ids": ids,
            "actions": {"target": torch.randn(length, 4), "action_mask": mask},
            "episode_key": f"tail-{rank}", "episode_start": True, "episode_end": True,
            "timesteps": list(range(length))}


def _tail_rank(rank, temporary):
    torch.set_num_threads(1)
    dist.init_process_group("gloo", init_method="file://"+str(Path(temporary)/"tail-rendezvous"),
                           rank=rank, world_size=2, timeout=timedelta(seconds=90))
    try:
        model = tail_core()
        ddp = DistributedDataParallel(model, broadcast_buffers=False, find_unused_parameters=True)
        backward_windows(ddp, [tail_sequence(rank)], [900+rank*100], 2, device="cpu")
        assert not model.memory.cognitive.bank
        torch.save({n: p.grad for n, p in model.named_parameters()}, Path(temporary)/f"tail-{rank}.pt")
    finally:
        dist.destroy_process_group()


def test_two_ranks_short_windows_match_global_masked_loss(tmp_path):
    mp.start_processes(_tail_rank, args=(str(tmp_path),), nprocs=2, join=True, start_method="spawn")
    torch.set_num_threads(1)
    model = tail_core()
    sequences = [tail_sequence(rank) for rank in range(2)]
    units = sum(float(s["actions"]["action_mask"].sum()) for s in sequences)
    for rank, seq in enumerate(sequences):
        for start in range(0, len(seq["timesteps"]), 4):
            end = min(start+4, len(seq["timesteps"]))
            actions = slice_features(seq["actions"], start, end, "cpu")
            loss = model(slice_features(seq["features"], start, end, "cpu"),
                seq["input_ids"][start:end], window_context(seq, start, end), actions, 900+rank*100+start)
            (loss*float(actions.action_mask.sum())/units).backward()
    for rank in range(2):
        grads = torch.load(tmp_path/f"tail-{rank}.pt", weights_only=True)
        for name, parameter in model.named_parameters():
            if parameter.grad is None:
                assert grads[name] is None or not grads[name].any()
            else:
                torch.testing.assert_close(grads[name], parameter.grad, atol=3e-7, rtol=3e-5)


def test_async_early_stop_leaves_common_future_checkpoint_for_both_arms():
    assert shared_stop_target([4150, 5280], 1000) == 7000
    assert shared_stop_target([6000, 5999], 1000) == 8000


def test_resume_preserves_optimizer_and_episode_memory_for_next_window(tmp_path):
    torch.set_num_threads(1)
    uninterrupted = core()
    optimizer = torch.optim.AdamW(uninterrupted.parameters(), lr=1e-5)

    def advance(model, opt, window):
        seq = sequence(1, window)
        opt.zero_grad(set_to_none=True)
        for start in range(0, 32, 4):
            loss = model(slice_features(seq["features"], start, start+4, "cpu"),
                seq["input_ids"][start:start+4], window_context(seq, start, start+4),
                slice_features(seq["actions"], start, start+4, "cpu"), 1000+start)
            (loss/8).backward()
        opt.step()
        return loss.detach().clone()

    advance(uninterrupted, optimizer, 0)
    directory = tmp_path / "checkpoints"
    directory.mkdir()
    torch.save(optimizer.state_dict(), directory / "latest_optimizer.pt")
    manifest = {"step": 1, "epoch": 0, "position": 1}
    torch.save(dict(manifest, memory_runtime=memory_runtime(uninterrupted)), directory / "latest_rank-001.pt")
    resumed = core()
    resumed.load_state_dict(uninterrupted.state_dict())
    restored_optimizer = torch.optim.AdamW(resumed.parameters(), lr=1e-5)
    restore_rank(resumed, restored_optimizer, tmp_path, manifest, 1, "cpu")
    assert resumed.memory.cognitive.merge_count == {"episode-1": 16}
    for bank in (resumed.memory.perceptual, resumed.memory.cognitive):
        assert len(bank.bank["episode-1"]) == 16
        assert all(not tensor.requires_grad for _, tensor in bank.bank["episode-1"])
    expected = advance(uninterrupted, optimizer, 1)
    actual = advance(resumed, restored_optimizer, 1)
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    for name, tensor in uninterrupted.state_dict().items():
        torch.testing.assert_close(resumed.state_dict()[name], tensor, rtol=0, atol=0)
    assert resumed.memory.cognitive.merge_count == {"episode-1": 48}
    with pytest.raises(AssertionError):
        restore_rank(resumed, restored_optimizer, tmp_path, dict(manifest, position=2), 1, "cpu")


def test_resume_after_one_frame_episode_preserves_partial_adam_steps(tmp_path):
    torch.set_num_threads(2)
    model = tail_core()
    optimizer = torch.optim.AdamW(model.parameters(),lr=1e-5)
    def advance(core,opt,rank):
        seq=tail_sequence(rank)
        opt.zero_grad(set_to_none=True)
        loss=core(seq['features'],seq['input_ids'],window_context(seq,0,len(seq['timesteps'])),seq['actions'],900)
        loss.backward()
        opt.step()
        return loss.detach()
    advance(model,optimizer,0)
    advance(model,optimizer,1)  # no retrieval in this one-observation episode
    assert {int(v['step']) for v in optimizer.state.values()} == {1,2}
    directory=tmp_path/'checkpoints'
    directory.mkdir()
    manifest=dict(step=2,epoch=0,position=2)
    torch.save(optimizer.state_dict(),directory/'latest_optimizer.pt')
    torch.save(dict(manifest,memory_runtime=memory_runtime(model)),directory/'latest_rank-000.pt')
    resumed=tail_core()
    resumed.load_state_dict(model.state_dict())
    restored_optimizer=torch.optim.AdamW(resumed.parameters(),lr=1e-5)
    restore_rank(resumed,restored_optimizer,tmp_path,manifest,0,'cpu')
    torch.testing.assert_close(advance(model,optimizer,0),advance(resumed,restored_optimizer,0),rtol=0,atol=0)
    for name,tensor in model.state_dict().items():
        torch.testing.assert_close(tensor,resumed.state_dict()[name],rtol=0,atol=0)


def test_resume_refuses_a_changed_global_batch_or_seed(tmp_path):
    (tmp_path / "config").mkdir()
    (tmp_path / "checkpoints/step-001000").mkdir(parents=True)
    args = SimpleNamespace(resume_step=1000, seed=42, global_windows_per_step=2,
        dataset=Path("/dataset"), checkpoint=Path("/official"), arm="memory_v1",
        microbatch=4, learning_rate=1e-5, steps=10000)
    manifest = {"step": 1000, "epoch": 0, "position": 1000, "seed": 42,
        "world_size": 2, "global_windows_per_step": 2, "path": str(tmp_path / "checkpoints/step-001000")}
    original = {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()}
    original["world_size"] = 2
    (tmp_path / "config/ddp.json").write_text(json.dumps(original))
    (tmp_path / "checkpoints/latest.json").write_text(json.dumps(manifest))
    assert resume_manifest(args, tmp_path, 2) == manifest
    assert resume_manifest(SimpleNamespace(**dict(vars(args), steps=15000)), tmp_path, 2) == manifest
    for key, value in (("seed", 43), ("global_windows_per_step", 4),
                       ("fine_tuning_mode", "lora"), ("freeze_action_dit", True),
                       ("lora_rank", 8), ("weight_decay", .05), ("max_grad_norm", .5)):
        with pytest.raises(AssertionError):
            resume_manifest(SimpleNamespace(**dict(vars(args), **{key: value})), tmp_path, 2)
    for steps in (1000, 5000):
        with pytest.raises(AssertionError, match="steps"):
            resume_manifest(SimpleNamespace(**dict(vars(args), steps=steps)), tmp_path, 2)

    from v1_groot_memoryvla.experiment.memory_training_options import TRAINING_PROTOCOL
    protocol = dict(training_protocol=TRAINING_PROTOCOL, memory_length=64,
        memory_conditioning="residual", memory_cognitive_qk_norm="layernorm", image_augmentation=True)
    (tmp_path / "config/ddp.json").write_text(json.dumps(dict(original, **protocol)))
    compatible = SimpleNamespace(**dict(vars(args), **protocol))
    assert resume_manifest(compatible, tmp_path, 2) == manifest
    with pytest.raises(AssertionError):
        resume_manifest(SimpleNamespace(**dict(vars(compatible), memory_residual_readout="value")), tmp_path, 2)


def test_takeover_restarts_only_baseline_and_preserves_live_memory(tmp_path, monkeypatch):
    from v1_groot_memoryvla.experiment import task21_async, launch_task21_ddp
    (tmp_path / "config").mkdir()
    (tmp_path / "status.json").write_text(json.dumps({"mode": "ddp_async", "status": "training",
        "pid": 91000, "processes": {"train_memory_v1": 91001, "train_baseline": 91002,
        "validate_memory_v1": 91003, "validate_baseline": 91004}}))
    (tmp_path / "config/ddp_launch.json").write_text(json.dumps({"commands": {
        "memory_v1": {"CUDA_VISIBLE_DEVICES": "2,3"}, "baseline": {"CUDA_VISIBLE_DEVICES": "1"}}}))
    old_read = Path.read_bytes
    def command_line(path):
        if str(path).startswith("/proc/910"):
            return f"launch_task21_ddp torch.distributed.run task21_async {tmp_path}".encode()
        return old_read(path)
    monkeypatch.setattr(Path, "read_bytes", command_line)
    monkeypatch.setattr(launch_task21_ddp, "busy_gpus", lambda allowed: set())
    monkeypatch.setattr(ExistingProcess, "poll", lambda self: 1)
    killed, trashed = [], []
    monkeypatch.setattr(task21_async.os, "kill", lambda pid, sig: killed.append(pid))
    monkeypatch.setattr(task21_async.subprocess, "run", lambda argv, **kwargs: trashed.append(argv[-1]))
    monkeypatch.setattr(task21_async, "launch_async", lambda args, groups, adopted: adopted)
    args = SimpleNamespace(output=tmp_path, allow_service_pids=[1, 2])
    adopted = take_over_async(args, {"memory_v1": [2, 3], "baseline": [0, 1]}, {"commands": {}})
    assert set(killed) == {91000, 91002, 91004}
    assert trashed == [str(tmp_path / "baseline")]
    assert {k: v.pid for k, v in adopted.items()} == {"train_memory_v1": 91001, "validate_memory_v1": 91003}
    assert json.loads((tmp_path / "config/transition_2plus2.json").read_text())["memory_training_continues"]
