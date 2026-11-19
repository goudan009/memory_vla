"""Regression checks for direct MemoryVLA conditioning and complete episodes."""

from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import pandas as pd
import pytest
import torch

from v1_groot_memoryvla.model.memory.adapter import MemoryAdapter
from v1_groot_memoryvla.tests.test_memory_v1 import features
from v1_groot_memoryvla.data.dataset.history_dataset import HistoryDataset, HistoryCollator
from v1_groot_memoryvla.data.types import ModalityConfig
from v1_groot_memoryvla.experiment.task21_ddp import EpisodePlan


@pytest.mark.parametrize('seed', [42, 42 + 100000 * (43000 - 1)])
def test_window_rng_is_reproducible_for_long_training_budgets(seed):
    import random
    from v1_groot_memoryvla.experiment.first_training import seeded
    seeded(seed)
    expected = (random.random(), np.random.rand(), torch.rand(3))
    seeded(seed)
    assert random.random() == expected[0]
    assert np.random.rand() == expected[1]
    torch.testing.assert_close(torch.rand(3), expected[2], atol=0, rtol=0)
    if seed == 42:
        assert expected[1] == np.random.RandomState(42).rand()


class TinyBackbone(torch.nn.Module):
    """Offline substitute for Eagle; all memory and action-head code stays real."""
    def __init__(self, **kwargs):
        super().__init__()
        self.embedding = torch.nn.Embedding(16, 32)

    def prepare_input(self, inputs):
        from transformers.feature_extraction_utils import BatchFeature
        return BatchFeature(data=inputs)

    def forward(self, inputs):
        from transformers.feature_extraction_utils import BatchFeature
        return BatchFeature(data={"backbone_features": self.embedding(inputs["input_ids"] % 16),
            "visual_features": inputs["pixels"], "image_mask": inputs["image_mask"],
            "backbone_attention_mask": inputs["attention_mask"]})


@pytest.mark.parametrize("conditioning", ["direct", "residual", "residual_value"])
@pytest.mark.parametrize("qk_norm", ["none", "layernorm"])
@pytest.mark.parametrize("use_lora", [False, True])
def test_real_groot_head_loss_inference_and_hf_reload(conditioning, qk_norm, use_lora, tmp_path, monkeypatch):
    import json
    from v1_groot_memoryvla.configs.model.gr00t_n1d6 import Gr00tN1d6Config
    from v1_groot_memoryvla.model.gr00t_n1d6 import gr00t_n1d6 as module
    from v1_groot_memoryvla.model.gr00t_n1d6 import processing_gr00t_n1d6 as processing
    monkeypatch.setattr(module, "get_backbone_cls", lambda config: TinyBackbone)
    monkeypatch.setattr(processing, "Gr00tN1d6DataCollator", lambda **kwargs: None)
    torch.set_num_threads(2)
    torch.manual_seed(17)
    readout = "value" if conditioning == "residual_value" else "delta"
    conditioning = "residual" if readout == "value" else conditioning
    config = Gr00tN1d6Config(memory_enabled=True, memory_conditioning=conditioning, memory_length=64,
        memory_cognitive_qk_norm=qk_norm, memory_residual_readout=readout,
        backbone_embedding_dim=32, hidden_size=32, input_embedding_dim=32, max_state_dim=8,
        max_action_dim=4, action_horizon=4, max_num_embodiments=1, load_bf16=False,
        use_flash_attention=False, diffusion_model_cfg=dict(num_layers=4, num_attention_heads=4,
            attention_head_dim=8, norm_type="ada_norm", dropout=0., final_dropout=False,
            output_dim=32, interleave_self_attention=True, positional_embeddings=None))
    model = module.Gr00tN1d6(config).train()
    if use_lora:
        from v1_groot_memoryvla.model.modules.lora import install_lora
        config.finetuning_lora = dict(rank=2, alpha=4., dropout=0., modules=[
            name for name, layer in model.named_modules()
            if name.startswith("action_head.model.") and isinstance(layer, torch.nn.Linear)])
        assert config.finetuning_lora["modules"]
        install_lora(model, config.finetuning_lora)
        for name in config.finetuning_lora["modules"]:
            torch.nn.init.normal_(model.get_submodule(name).lora_B, std=.01)
    data, ids = features(2)
    inputs = dict(input_ids=ids, pixels=data.visual_features, image_mask=data.image_mask,
        attention_mask=data.backbone_attention_mask, state=torch.randn(2, 1, 8),
        action=torch.randn(2, 4, 4), action_mask=torch.ones(2, 4, 4),
        embodiment_id=torch.zeros(2, dtype=torch.long),
        memory_context=dict(mode="group", group_lengths=[2], episode_keys=["a"]*2, timesteps=[0, 1]))
    loss = model(inputs)["loss"]
    assert torch.isfinite(loss)
    loss.backward()
    assert model.memory.perceptual_encoder.reduce.weight.grad.norm() > 0
    assert model.memory.cognitive.retrieval_blocks[0].v_proj.weight.grad.norm() > 0
    model.eval()
    sample = {k: v[:1] for k, v in inputs.items() if torch.is_tensor(v)}
    sample["memory_context"] = dict(mode="stream", episode_keys=["probe"], timesteps=[0])
    torch.manual_seed(51)
    before = model.get_action(sample).action_pred
    assert before.shape == (1, 4, 4) and torch.isfinite(before).all()
    assert model.memory.cognitive.last_written["probe"] == 0
    assert len(model.memory.cognitive.bank["probe"]) == 1  # four denoising calls, one write
    model.save_pretrained(tmp_path)
    if conditioning == "residual" or qk_norm == "none":
        # Simulate checkpoints predating either new config field.
        saved = json.loads((tmp_path / "config.json").read_text())
        if conditioning == "residual":
            saved.pop("memory_conditioning")
        if qk_norm == "none":
            saved.pop("memory_cognitive_qk_norm")
        (tmp_path / "config.json").write_text(json.dumps(saved))
    restored, info = module.Gr00tN1d6.from_pretrained(tmp_path, output_loading_info=True)
    assert not info["missing_keys"] and not info["unexpected_keys"] and not info["mismatched_keys"]
    assert restored.config.memory_conditioning == conditioning
    assert restored.config.memory_cognitive_qk_norm == qk_norm
    assert restored.memory.residual_readout == readout
    assert restored.memory.cognitive.retrieval_blocks[0].qk_norm == qk_norm
    assert restored.memory.cognitive.capacity == 64 and not restored.memory.cognitive.bank
    if use_lora:
        from v1_groot_memoryvla.model.modules.lora import LoRALinear
        assert restored.config.finetuning_lora == config.finetuning_lora
        assert all(isinstance(restored.get_submodule(name), LoRALinear)
                   for name in config.finetuning_lora["modules"])
    torch.manual_seed(51)
    after = restored.get_action(sample).action_pred
    torch.testing.assert_close(after, before, atol=0, rtol=0)


def test_direct_conditions_have_no_context_bypass_and_train_perception_on_first_frame():
    torch.set_num_threads(2)
    torch.manual_seed(41)
    adapter = MemoryAdapter(vlm_dim=32, perceptual_dim=32, capacity=64,
                            conditioning="direct").eval()
    original, ids = features(1)
    original.visual_features.requires_grad_()
    context = {"mode": "empty", "episode_keys": ["a"], "timesteps": [0]}
    result = adapter(original, ids, context)
    assert result.backbone_features.shape == (1, 13, 32)
    assert result.image_mask.tolist() == [[True] * 12 + [False]]
    assert result.backbone_attention_mask.all()
    torch.testing.assert_close(result.backbone_features[:, -1], original.backbone_features[:, 20])
    result.backbone_features.square().mean().backward()
    assert adapter.perceptual_encoder.reduce.weight.grad.norm() > 0
    assert adapter.perceptual_projection.weight.grad.norm() > 0
    assert original.visual_features.grad.norm() > 0
    assert not adapter.cognitive.bank
    changed = {k: v.clone() for k, v in original.items()}
    changed["backbone_features"][:, :20] += 100
    other = adapter(changed, ids, context)
    torch.testing.assert_close(other.backbone_features, result.backbone_features)


class FakeProcessor:
    def __init__(self):
        self.modality_configs = {"behavior_r1_pro": {
            "video": ModalityConfig(delta_indices=[0], modality_keys=["front"]),
            "state": ModalityConfig(delta_indices=[0], modality_keys=["q"]),
            "language": ModalityConfig(delta_indices=[0], modality_keys=["task"]),
            "action": ModalityConfig(delta_indices=list(range(32)), modality_keys=["q"]),
        }}

    def configure_b1k_language(self, paths):
        pass

    def __call__(self, messages):
        return {"action": torch.tensor(messages[0]["content"].actions["q"]),
                "action_mask": torch.ones(32, 1)}


class FakeLoader:
    fps = 30
    episodes_metadata = [{"episode_index": 0, "length": 266},
                         {"episode_index": 1, "length": 1}]

    def __init__(self, *args, **kwargs):
        pass

    def get_dataset_statistics(self):
        return {}

    def _load_parquet_data(self, episode):
        length = self.episodes_metadata[episode]["length"]
        return pd.DataFrame({"state.q": [np.array([i]) for i in range(length)],
                             "action.q": [np.array([i]) for i in range(length)],
                             "language.task": ["task"] * length})

    def _load_video_data(self, episode, frames):
        assert max(frames) < self.episodes_metadata[episode]["length"]
        return {"front": np.zeros((len(frames), 2, 2, 3), dtype=np.uint8)}


def test_tail_observations_survive_and_padded_actions_do_not_supervise():
    with patch("v1_groot_memoryvla.data.dataset.history_dataset.LeRobotEpisodeLoader", FakeLoader):
        dataset = HistoryDataset(["/unused"], FakeProcessor(), group_size=32)
    assert dataset.groups == [(0, 0, 0), (0, 0, 256), (0, 1, 0)]
    tail, short = dataset[1], dataset[2]
    assert tail["frame_indices"] == [256, 264]
    assert tail["timesteps"] == [32, 33]
    assert tail["steps"][0]["action_mask"].sum() == 10
    assert tail["steps"][1]["action_mask"].sum() == 2
    assert short["frame_indices"] == [0]
    assert short["steps"][0]["action_mask"].sum() == 1
    collator = HistoryCollator(lambda steps: {"inputs": {"count": len(steps)}}, group_size=32)
    context = collator([tail, short])["inputs"]["memory_context"]
    assert context["group_lengths"] == [2, 1]
    assert len(context["episode_keys"]) == 3


def test_each_slot_finishes_an_episode_before_switching():
    by_episode = {0: [0, 1, 2], 1: [3, 4, 5], 2: [6, 7]}
    groups = [(0, e, w * 256) for e, indices in by_episode.items() for w in range(len(indices))]
    dataset = SimpleNamespace(by_episode=by_episode, groups=groups,
        order=lambda seed: [0, 3, 6, 1, 4, 7, 2, 5])
    plan = EpisodePlan(dataset, 1, 42)
    seen = [plan.batch(i)[0] for i in range(plan.steps)]
    for indices in by_episode.values():
        positions = [seen.index(i) for i in indices]
        assert positions == list(range(positions[0], positions[0] + len(indices)))


@pytest.mark.parametrize("capacity", [16, 64, 256])
@pytest.mark.parametrize("qk_norm", ["none", "layernorm"])
def test_direct_checkpoint_roundtrip_and_chunked_stream(capacity, qk_norm, tmp_path):
    torch.set_num_threads(2)
    from transformers.feature_extraction_utils import BatchFeature
    torch.manual_seed(13)
    adapter = MemoryAdapter(32, 32, capacity=capacity, conditioning="direct", cognitive_qk_norm=qk_norm).eval()
    data, ids = features(19)
    group = adapter(data, ids, {"mode": "group", "group_lengths": [19],
        "episode_keys": ["a"] * 19, "timesteps": list(range(19))})
    torch.save(adapter.state_dict(), tmp_path / "weights.pt")
    restored = MemoryAdapter(32, 32, capacity=capacity, conditioning="direct", cognitive_qk_norm=qk_norm).eval()
    restored.load_state_dict(torch.load(tmp_path / "weights.pt", weights_only=True))
    chunks = []
    for start in range(0, 19, 4):
        end = min(start + 4, 19)
        result = restored(BatchFeature({k: v[start:end] for k, v in data.items()}), ids[start:end],
            {"mode": "train_sequence", "sequence_start": start == 0, "sequence_end": end == 19,
             "episode_keys": ["a"] * (end-start), "timesteps": list(range(start, end))})
        chunks.append(result.backbone_features)
    torch.testing.assert_close(torch.cat(chunks), group.backbone_features, atol=0, rtol=0)
    assert not restored.cognitive.bank


@pytest.mark.parametrize("capacity,start", [(16, 0), (16, 20), (64, 70), (256, 260)])
def test_validation_empty_wrong_keep_interface_and_live_history(capacity, start, monkeypatch):
    from contextlib import nullcontext
    from itertools import repeat
    from transformers.feature_extraction_utils import BatchFeature
    from v1_groot_memoryvla.experiment import task21_formal as formal
    from v1_groot_memoryvla.experiment.first_training import bank_trace
    monkeypatch.setattr(formal, "slice_features", lambda f, s, e: BatchFeature({k:v[s:e] for k,v in f.items()}))
    monkeypatch.setattr(torch.Tensor, "cuda", lambda self, *args, **kwargs: self)
    monkeypatch.setattr(torch, "autocast", lambda *args, **kwargs: nullcontext())
    torch.set_num_threads(2)
    adapter = MemoryAdapter(32, 32, capacity=capacity, conditioning="direct").eval()
    model = SimpleNamespace(memory=adapter)
    data, ids = features(3)
    target = dict(features=data, input_ids=ids, episode_key="live", timesteps=list(range(start,start+3)))
    with torch.no_grad():
        for t in range(start):
            adapter(formal.slice_features(data,0,1),ids[:1],
                dict(mode="stream",episode_keys=["live"],timesteps=[t]))
        with bank_trace(adapter) as trace:
            normal = adapter(data,ids,dict(mode="stream",episode_keys=["live"]*3,timesteps=target["timesteps"]))
        history = list(adapter.cognitive.bank["live"])
        donor_data, donor_ids = features(2)
        donor = dict(features=donor_data,input_ids=donor_ids,timesteps=[0,1])
        wrong = formal.wrong_features(model,target,repeat(donor),trace)
        empty = formal.empty_features(model,target)
    for output in (wrong,empty):
        assert output["backbone_features"].shape == normal.backbone_features.shape == (3,13,32)
        assert torch.equal(output["image_mask"],normal.image_mask)
    assert set(adapter.cognitive.bank)=={"live"}
    assert all(t==s and f is g for (t,f),(s,g) in zip(history,adapter.cognitive.bank["live"]))
    if start==0:
        torch.testing.assert_close(empty["backbone_features"][:1],normal.backbone_features[:1])
        torch.testing.assert_close(wrong["backbone_features"][:1],normal.backbone_features[:1])
    else:
        assert not torch.equal(empty["backbone_features"],normal.backbone_features)
        assert not torch.equal(wrong["backbone_features"],normal.backbone_features)


def test_validation_restores_training_augmentation_even_on_error():
    from v1_groot_memoryvla.experiment.task21_formal import evaluation_processors
    class Processor:
        def __init__(self, training):
            self.training = training
        def train(self):
            self.training = True
        def eval(self):
            self.training = False
    train, validation = Processor(True), Processor(False)
    with pytest.raises(RuntimeError):
        with evaluation_processors(train,validation):
            assert not train.training and not validation.training
            raise RuntimeError("probe")
    assert train.training and not validation.training


def test_async_initial_validation_uses_same_new_memory_config_as_training(tmp_path, monkeypatch):
    import json
    from v1_groot_memoryvla.experiment import task21_formal as formal
    class Processor:
        training = True
        def train(self): self.training = True
        def eval(self): self.training = False
    class Backbone:
        def cuda(self): return self
        def eval(self): return self
    checkpoint = tmp_path / "base"
    checkpoint.mkdir()
    (checkpoint/"config.json").write_text('{}')
    (tmp_path/"config").mkdir()
    (tmp_path/"config/data_split.json").write_text(json.dumps(
        dict(task=0,train=list(range(180)),validation=list(range(180,200)))))
    monkeypatch.setattr(formal.Gr00tN1d6Processor,"from_pretrained",lambda path: Processor())
    monkeypatch.setattr(formal,"EpisodeWindows",lambda path,p,ids,**kwargs: SimpleNamespace(processor=p))
    calls=[]
    monkeypatch.setattr(formal,"load_model",lambda path,memory,seed,**kwargs:
        calls.append(kwargs) or SimpleNamespace(backbone=Backbone()))
    args=SimpleNamespace(checkpoint=checkpoint,output=tmp_path,dataset=tmp_path,arm="memory_v1",seed=42,
                         memory_conditioning="direct",memory_length=256,image_augmentation=True)
    processor, train, validation, _ = formal.initialize(args)
    formal.initialize(args,checkpoint)
    assert calls[0]==calls[1]==dict(memory_conditioning="direct",memory_length=256,
                                   memory_cognitive_qk_norm="layernorm",finetuning=args)
    assert train.processor is processor and train.processor.training
    assert validation.processor is not processor and not validation.processor.training
    (checkpoint/"config.json").write_text(json.dumps(dict(memory_enabled=True,memory_conditioning="direct",memory_length=64)))
    formal.initialize(args,checkpoint)
    assert calls[-1]==dict(finetuning=args)  # trained memory config is authoritative on reload


@pytest.mark.parametrize("existing_memory", [False, True])
def test_residual_identity_initialization_only_applies_to_new_memory(tmp_path, monkeypatch, existing_memory):
    import json
    from v1_groot_memoryvla.experiment import task21_formal as formal
    class Processor:
        def train(self): pass
        def eval(self): pass
    class Backbone:
        def cuda(self): return self
        def eval(self): return self
    checkpoint = tmp_path / "base"
    checkpoint.mkdir()
    (checkpoint / "config.json").write_text(json.dumps(dict(
        memory_enabled=existing_memory, memory_conditioning="residual",
        memory_cognitive_qk_norm="layernorm", memory_residual_identity_init=False)))
    (tmp_path / "config").mkdir()
    (tmp_path / "config/data_split.json").write_text(json.dumps(dict(
        task=0, train=list(range(180)), validation=list(range(180, 200)))))
    monkeypatch.setattr(formal.Gr00tN1d6Processor, "from_pretrained", lambda path: Processor())
    monkeypatch.setattr(formal, "EpisodeWindows", lambda *a, **kw: None)
    calls = []
    monkeypatch.setattr(formal, "load_model", lambda *a, **kw:
        calls.append(kw) or SimpleNamespace(backbone=Backbone()))
    args = SimpleNamespace(checkpoint=checkpoint, output=tmp_path, dataset=tmp_path,
        arm="memory_v1", seed=42, memory_conditioning="residual", memory_length=64)
    formal.initialize(args)
    # Inserting a zero gate into a trained legacy residual adapter would disable
    # its learned cognitive branch. Only newly added Memory needs a zero gate.
    assert calls[-1].get("memory_residual_identity_init", False) is not existing_memory


def test_image_augmentation_is_repeatable_per_window_and_changes_between_seeds(monkeypatch):
    from contextlib import nullcontext
    from v1_groot_memoryvla.experiment.first_training import cache_sequence
    from v1_groot_memoryvla.model.gr00t_n1d6.image_augmentations import build_image_transformations_albumentations
    transform, _ = build_image_transformations_albumentations(None,None,None,
        dict(brightness=.3,contrast=.3,saturation=.3,hue=.1),32,.75)
    def collate(steps):
        return {"inputs": {k: torch.stack([s[k] for s in steps]) for k in steps[0]}}
    processor = SimpleNamespace(train_image_transform=transform,collator=collate)
    class Dataset:
        def __init__(self): self.processor = processor
        def __getitem__(self,index):
            pixels = np.arange(32*32*3,dtype=np.uint8).reshape(32,32,3)
            values = [torch.tensor(transform(image=pixels)["image"].copy()) for _ in range(2)]
            return dict(timesteps=[0,1],steps=[dict(input_ids=torch.zeros(3,dtype=torch.long),
                pixels=value,state=torch.zeros(1,2),action=torch.zeros(2,2),action_mask=torch.ones(2,2),
                embodiment_id=torch.tensor(0)) for value in values])
    monkeypatch.setattr(torch,"autocast",lambda *args,**kwargs:nullcontext())
    model = SimpleNamespace(prepare_input=lambda b:(b,b),backbone=lambda b:dict(backbone_features=b['pixels']))
    dataset=Dataset()
    a=cache_sequence(model,processor,dataset,0,1,augmentation_seed=42)
    b=cache_sequence(model,processor,dataset,0,2,augmentation_seed=43)
    c=cache_sequence(model,processor,dataset,0,2,augmentation_seed=42)
    assert torch.equal(a['features']['backbone_features'],c['features']['backbone_features'])
    assert not torch.equal(a['features']['backbone_features'],b['features']['backbone_features'])
