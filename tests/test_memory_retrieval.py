"""Causal, equal-budget memory retrieval behavior."""

import torch

from vlamemory.memory import BaseMemoryBank


def test_session_reset_clears_both_memory_streams_without_touching_other_envs():
    from vlamemory.memory import ActionConditionedMemoryBank
    from vlamemory.memory.sessions import EpisodeMemorySessions

    bank = ActionConditionedMemoryBank(
        model_dim=8, num_groups=2, num_attention_heads=2,
        retrieval_layers=1, use_timestep_pe=False,
    )
    sessions = EpisodeMemorySessions(bank)
    for session, env_id in [("gpu0", 0), ("gpu1", 0), ("gpu0", 1)]:
        sessions.reset({"session_id": session, "env_ids": [env_id], "episode_ids": ["0"]})
        key = sessions.episode_id(session, env_id)
        for stream in ("cognitive", "perceptual"):
            bank.write(key, torch.ones((1, 8)), timestep=0, memory_type=stream)
    old = sessions.episode_id("gpu0", 0)
    other_keys = [sessions.episode_id("gpu1", 0), sessions.episode_id("gpu0", 1)]
    sessions.reset({"session_id": "gpu0", "env_ids": [0], "episode_ids": ["1"]})
    new = sessions.episode_id("gpu0", 0)
    assert new != old
    for stream in (bank.cognitive_bank, bank.perceptual_bank):
        assert not stream.read(old, 1)
        assert not stream.read(new, 1)
        assert all(stream.read(key, 1) for key in other_keys)


def test_tome_merge_never_exposes_future_features_to_past_queries():
    bank = BaseMemoryBank(mem_length=2, consolidate_type="tome")
    bank.write("ep", torch.tensor([[1.0, 0.0]]), timestep=0)
    bank.write("ep", torch.tensor([[0.0, 1.0]]), timestep=2)
    bank.write("ep", torch.tensor([[0.0, 1.0]]), timestep=3)

    assert [t for t, _ in bank.read("ep", current_timestep=3)] == [0]
    assert [t for t, _ in bank.read("ep", current_timestep=4)] == [0, 3]


def test_retrieval_never_reads_current_or_future_entries():
    from vlamemory.memory import ActionConditionedMemoryBank, UniformMemoryBank

    state = torch.ones((1, 2, 8), dtype=torch.float32)
    feature = torch.ones((1, 8), dtype=torch.float32)
    models = (
        UniformMemoryBank(model_dim=8, num_output_tokens=3, num_attention_heads=2,
                          retrieval_layers=1, use_timestep_pe=False),
        ActionConditionedMemoryBank(model_dim=8, num_groups=2, num_attention_heads=2,
                                    retrieval_layers=1, use_timestep_pe=False),
    )
    for model in models:
        model.write("ep", feature, timestep=4)
        result = model.retrieve(state, "ep", current_timestep=4)
        outputs = result if isinstance(result, tuple) else (result,)
        for context in outputs:
            assert bool(torch.all(context == 0))
        model.write("ep", feature * 2, timestep=5)
        result = model.retrieve(state, "ep", current_timestep=4)
        outputs = result if isinstance(result, tuple) else (result,)
        for context in outputs:
            assert bool(torch.all(context == 0))


def test_perceptual_only_history_does_not_use_cognitive_retriever():
    from vlamemory.memory import ActionConditionedMemoryBank

    model = ActionConditionedMemoryBank(
        model_dim=8, num_groups=2, num_attention_heads=2,
        retrieval_layers=1, use_timestep_pe=False,
    )
    state = torch.ones((1, 2, 8), dtype=torch.float32)
    feature = torch.arange(8, dtype=torch.float32)[None, :]
    model.write("ep", feature, timestep=0, memory_type="perceptual")
    perceptual_global, _ = model.retrieve(state, "ep", current_timestep=1)
    model.reset("ep")
    model.write("ep", feature, timestep=0, memory_type="cognitive")
    cognitive_global, _ = model.retrieve(state, "ep", current_timestep=1)

    assert not bool(torch.allclose(perceptual_global, cognitive_global))


def test_uniform_is_unconditioned_view_of_same_dual_stream_retriever():
    from vlamemory.memory import ActionConditionedMemoryBank, UniformMemoryBank

    shared = dict(model_dim=8, mem_length=3, retrieval_layers=1,
                  num_attention_heads=2, use_timestep_pe=False,
                  consolidate_type="fifo", fusion_type="add")
    uniform = UniformMemoryBank(num_output_tokens=3, **shared)
    unconditioned = ActionConditionedMemoryBank(
        num_groups=2, condition_on_group=False, **shared
    )
    unconditioned.load_state_dict(uniform.state_dict())
    feature = torch.arange(8, dtype=torch.float32)[None, :]
    for model in (uniform, unconditioned):
        model.write("ep", feature, timestep=0, memory_type="cognitive")
        model.write("ep", feature.flip(0) + 1, timestep=1, memory_type="perceptual")
    state = torch.ones((1, 2, 8), dtype=torch.float32)
    uniform_tokens = uniform.retrieve(state, "ep", current_timestep=2)
    global_context, groups = unconditioned.retrieve(state, "ep", current_timestep=2)

    assert uniform_tokens.shape == (1, 3, 8)
    assert bool(torch.allclose(groups[:, 0], groups[:, 1]))
    expected = torch.concatenate([global_context, groups.reshape(1, 2, 8)], dim=1)
    assert bool(torch.allclose(uniform_tokens, expected))
