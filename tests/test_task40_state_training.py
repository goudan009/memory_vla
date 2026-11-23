import unittest
from types import SimpleNamespace

import torch
from torch import nn
from transformers.feature_extraction_utils import BatchFeature

from v1_groot_memoryvla.configs.model.gr00t_n1d6 import Gr00tN1d6Config
from v1_groot_memoryvla.experiment.task40_state_training import (
    Task40StateHead, forward_state_only, masked_state_loss, processor_inputs,
)
from v1_groot_memoryvla.model.gr00t_n1d6.gr00t_n1d6 import Gr00tN1d6ActionHead
from v1_groot_memoryvla.model.memory.adapter import MemoryAdapter


class TinyProcessor:
    training = False
    modality_configs = {"behavior_r1_pro": {
        "video": SimpleNamespace(modality_keys=["camera"]),
        "state": SimpleNamespace(modality_keys=["robot"]),
    }}

    def __call__(self, messages):
        self.step = messages[0]["content"]
        return torch.from_numpy(self.step.states["robot"])

    def collator(self, states):
        return {"inputs": {"state": torch.stack(states), "embodiment_id": torch.zeros(1, dtype=torch.long)}}


class TinyBackbone(nn.Module):
    def __init__(self):
        super().__init__()
        self.features = nn.Parameter(torch.randn(1, 3, 8))

    def forward(self, inputs):
        assert not self.training and not torch.is_grad_enabled()
        return BatchFeature(data=dict(
            backbone_features=self.features + 0, visual_features=self.features[:, None, :1] + 0,
            image_mask=torch.tensor([[True, False, False]]),
            backbone_attention_mask=torch.ones(1, 3, dtype=torch.bool),
        ))


class TinyPolicy(nn.Module):
    def __init__(self, route):
        super().__init__()
        self.backbone = TinyBackbone()
        config = Gr00tN1d6Config(
            diffusion_model_cfg=dict(memory_fusion_mode=route, num_attention_heads=2,
                                     attention_head_dim=4, num_layers=2, output_dim=8,
                                     dropout=0.2, positional_embeddings=None),
            backbone_embedding_dim=8, hidden_size=8, input_embedding_dim=8, max_state_dim=3,
            max_action_dim=2, max_num_embodiments=1, max_seq_len=8, action_horizon=2,
            num_inference_timesteps=2, use_alternate_vl_dit=False, tune_diffusion_model=False,
            tune_projector=False, tune_vlln=False, memory_enabled=True,
        )
        self.action_head = Gr00tN1d6ActionHead(config)
        self.memory = MemoryAdapter(vlm_dim=8, capacity=1, eos_token_id=3,
                                    residual_identity_init=True, residual_readout="value", fusion_mode=route)
        self.memory.requires_grad_(False)
        if route == "route_a":
            self.memory.cognitive_fusion.requires_grad_(True)
        self.state_head = Task40StateHead(8)

    def prepare_input(self, inputs):
        assert "action" not in inputs
        return {"input_ids": torch.tensor([[1, 2, 3]])}, BatchFeature(data=inputs)

    def forward(self, processor, sample, **kwargs):
        return forward_state_only(self, processor, self.state_head, sample, **kwargs)


def sample_input():
    key = ("archived", "0", "episode")
    snapshot = {name: dict(capacity=1, bank={key: [(0, torch.randn(1, dim))]},
                           last_written={key: 0}, merge_count={key: 0})
                for name, dim in (("perceptual", 256), ("cognitive", 8))}
    return dict(
        observation={"video.camera": torch.zeros(1, 1, 2, 2, 3, dtype=torch.uint8),
                     "state.robot": torch.rand(1, 1, 3)}, text="Make microwave popcorn.",
        memory=snapshot, memory_context=dict(mode="stream", episode_keys=[key], timesteps=[1]),
        targets=torch.tensor([0., 1., 0.]), target_mask=torch.tensor([False, True, True]),
    )


class Task40StateTrainingTest(unittest.TestCase):
    def test_processor_returns_ddp_inputs_to_host(self):
        class DeviceTensor(torch.Tensor):
            def numpy(self):
                raise TypeError("Convert device tensor to CPU before NumPy")

            def cpu(self):
                return self.as_subclass(torch.Tensor)

        sample = sample_input()
        sample["observation"] = {key: tensor.as_subclass(DeviceTensor)
                                 for key, tensor in sample["observation"].items()}
        processor = TinyProcessor()
        processor_inputs(processor, sample)
        self.assertEqual(processor.step.images["camera"].shape, (1, 2, 2, 3))
        self.assertEqual(processor.step.states["robot"].shape, (1, 3))

    def test_uses_final_real_dit_tokens_and_backpropagates_to_each_fusion(self):
        for route in ("route_a", "route_b"):
            with self.subTest(route=route):
                torch.manual_seed(42)
                model, processor, sample = TinyPolicy(route), TinyProcessor(), sample_input()
                model.memory.cognitive.eval()  # Mixed modes must survive the auxiliary forward.
                live = []
                for bank in (model.memory.perceptual, model.memory.cognitive):
                    bank.bank["demo"] = [(2, torch.tensor([1.]))]
                    bank.last_written["demo"] = 2
                    bank.merge_count["demo"] = 3
                    live.append((bank, bank.bank, bank.last_written, bank.merge_count))

                dit_outputs, head_inputs = [], []
                model.action_head.model.register_forward_hook(lambda _, args, result: dit_outputs.append(result))
                model.state_head.register_forward_pre_hook(lambda _, args: head_inputs.append(args[0]))
                rng_before = torch.get_rng_state()
                result = model(processor, sample, noise_seed=7)
                self.assertTrue(torch.equal(rng_before, torch.get_rng_state()))
                self.assertEqual(len(dit_outputs), model.action_head.num_inference_timesteps)
                torch.testing.assert_close(head_inputs[0], dit_outputs[-1][:, -model.action_head.action_horizon:])
                self.assertEqual(processor.step.actions, {})
                self.assertEqual(processor.step.text, sample["text"])
                self.assertEqual(processor.step.images["camera"].shape, (1, 2, 2, 3))
                self.assertEqual(processor.step.states["robot"].shape, (1, 3))
                torch.testing.assert_close(result["logits"], model(processor, sample, noise_seed=7)["logits"])
                self.assertTrue(model.backbone.training and model.action_head.training and model.memory.training)
                self.assertFalse(model.memory.cognitive.training)
                for bank, history, clock, merges in live:
                    self.assertIs(bank.bank, history)
                    self.assertIs(bank.last_written, clock)
                    self.assertIs(bank.merge_count, merges)
                    self.assertEqual(list(bank.bank), ["demo"])
                for snapshot in sample["memory"].values():
                    self.assertEqual([t for t, _ in next(iter(snapshot["bank"].values()))], [0])

                result["loss"].backward()
                projection = (model.memory.cognitive_fusion.value if route == "route_a"
                              else model.action_head.cognitive_fusion.output)
                self.assertGreater(projection.weight.grad.norm().item(), 0)
                self.assertGreater(model.state_head.projection.weight.grad.norm().item(), 0)
                self.assertIsNone(model.backbone.features.grad)

    def test_restores_banks_when_stream_forward_raises_after_write(self):
        model, sample = TinyPolicy("route_b"), sample_input()
        history = model.memory.cognitive.bank

        def fail_after_write(adapter, args, outputs):
            key, = sample["memory_context"]["episode_keys"]
            self.assertEqual(adapter.cognitive.last_written[key], 1)
            raise RuntimeError("after archive write")

        model.memory.register_forward_hook(fail_after_write)
        with self.assertRaisesRegex(RuntimeError, "after archive write"):
            model(TinyProcessor(), sample)
        self.assertIs(model.memory.cognitive.bank, history)
        self.assertEqual(history, {})
        self.assertTrue(model.memory.training)

    def test_bce_excludes_unknown_labels_including_all_unknown(self):
        logits = torch.tensor([[100., 0., 0.]], requires_grad=True)
        targets = torch.tensor([[float("nan"), 1., 0.]])
        mask = torch.tensor([[False, True, True]])
        loss = masked_state_loss(logits, targets, mask)
        torch.testing.assert_close(loss, torch.tensor(2.).log())
        loss.backward()
        torch.testing.assert_close(logits.grad, torch.tensor([[0., -0.25, 0.25]]))
        logits.grad = None
        loss = masked_state_loss(logits, targets, torch.zeros_like(mask))
        self.assertEqual(loss.item(), 0.)
        loss.backward()
        torch.testing.assert_close(logits.grad, torch.zeros_like(logits))


if __name__ == "__main__":
    unittest.main()
