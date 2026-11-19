"""Real B1K + official-weight acceptance of the complete v1 integration.

Run after train_official_batch. Saves an actual standalone HF v1 checkpoint;
the official checkpoint is read-only. No simulated/fabricated model features.
"""

import argparse
import gc
import json
from pathlib import Path
import time

import torch
from transformers.feature_extraction_utils import BatchFeature

from v3_bodymemory.audit.train_official_batch import load_real_step
from v3_bodymemory.data.dataset.history_dataset import HistoryCollator, HistoryDataset
from v3_bodymemory.model.gr00t_n1d6.gr00t_n1d6 import Gr00tN1d6
from v3_bodymemory.model.gr00t_n1d6.processing_gr00t_n1d6 import Gr00tN1d6Processor
from v3_bodymemory.policy.memory_state import MemoryState


def subset(features, index):
    return BatchFeature(data={k: v[index:index + 1] for k, v in features.items()})


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=Path("outputs/v3_bodymemory/validation/acceptance"))
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    report = {"weights": "official + newly initialized memory", "checks": {}}
    checks = report["checks"]
    started = time.monotonic()
    torch.manual_seed(42)
    processor = Gr00tN1d6Processor.from_pretrained(args.checkpoint)
    processor.eval()  # Deterministic temporal equivalence, no random image augmentation.
    dataset = HistoryDataset([args.dataset], processor)
    group = dataset[0]
    batch = HistoryCollator(processor.collator)([group])
    context = batch["inputs"]["memory_context"]
    assert context["frame_indices"] == list(range(0, 128, 8))
    assert len(set(context["episode_keys"])) == 1
    report["training_groups"] = len(dataset)
    report["history_frames"] = context["frame_indices"]
    print("REAL_GROUP", report["history_frames"], flush=True)
    model, info = Gr00tN1d6.from_pretrained(args.checkpoint, memory_enabled=True,
        tune_llm=False, tune_visual=False, tune_top_llm_layers=0,
        backbone_trainable_params_fp32=False, torch_dtype=torch.float32,
        output_loading_info=True)
    assert info["missing_keys"] and all(k.startswith("memory.") for k in info["missing_keys"])
    assert not info["unexpected_keys"] and not info.get("mismatched_keys")
    assert all(torch.isfinite(p).all() for p in model.memory.parameters())
    model.backbone.requires_grad_(False).bfloat16()
    model.cuda().eval()
    dataset_index, episode_id, first_frame = dataset.groups[0]
    step = load_real_step(dataset.loaders[dataset_index], episode_id, first_frame)
    single = processor.collator([processor([{"type": "episode_step", "content": step}])])
    stream = {"mode": "stream", "episode_keys": ["identity"], "timesteps": [0]}
    single["inputs"]["memory_context"] = stream
    with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
        # Compare full action generation using exactly the same weights/noise.
        memory = model.memory
        model.memory = None
        torch.manual_seed(123)
        baseline = model.get_action(**single)["action_pred"].clone()
        model.memory = memory
        torch.manual_seed(123)
        with_memory = model.get_action(**single)["action_pred"].clone()
        assert torch.equal(baseline, with_memory)
        checks["empty_history_full_actions_exact"] = True
    model.memory.reset()
    with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
        backbone_input, _ = model.prepare_input(batch["inputs"])
        actual = model.backbone(backbone_input)
        report["shapes"] = {k: list(v.shape) for k, v in actual.items()}
        ids = backbone_input["input_ids"]
        model.memory.train()
        group_features = model.memory(actual, ids, context).backbone_features
        model.memory.eval()
        sequential = []
        for i in range(16):
            sequential.append(model.memory(subset(actual, i), ids[i:i + 1], {
                "mode": "stream", "episode_keys": [context["episode_keys"][i]],
                "timesteps": [i]}).backbone_features)
        sequential = torch.cat(sequential)
        # The same batched Eagle features isolate temporal semantics from
        # CUDA kernels' batch-size-dependent VLM rounding.
        torch.testing.assert_close(group_features, sequential, rtol=0, atol=0)
        checks["training_inference_memory_sequence_exact"] = True
        assert model.memory.perceptual.merge_count == {}
        seventeenth = model.memory(subset(actual, 0), ids[:1], {
            "mode": "stream", "episode_keys": [context["episode_keys"][0]], "timesteps": [16]})
        for bank in (model.memory.perceptual, model.memory.cognitive):
            key = context["episode_keys"][0]
            assert len(bank.bank[key]) == 16 and bank.merge_count[key] == 1
        checks["seventeenth_write_merges_both_banks"] = True
        model.memory.reset()
        state = MemoryState(model.memory)
        options = {"session_id": "test", "env_ids": [0, 1], "episode_ids": ["e0", "e0"]}
        ctx = state.context(2, options)
        model.memory(BatchFeature(data={k: v[:2] for k, v in actual.items()}), ids[:2], ctx)
        state.advance(ctx)
        state.reset({"session_id": "test", "env_ids": [0], "episode_ids": ["e1"]})
        for bank in (model.memory.perceptual, model.memory.cognitive):
            assert ("test", "0", "e0") not in bank.bank
            assert ("test", "1", "e0") in bank.bank
        checks["independent_environment_reset"] = True
    del actual, backbone_input, group_features, sequential, seventeenth, baseline, with_memory
    model.memory.reset()
    model.train()
    parameters = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(parameters, lr=1e-5, foreach=False)
    before = model.memory.perceptual_residual.weight.detach().clone()
    with torch.autocast("cuda", dtype=torch.bfloat16):
        output = model(**batch)
        loss = output["loss"]
    assert torch.isfinite(loss)
    loss.backward()
    gradients = {}
    for name, parameter in model.memory.named_parameters():
        assert parameter.grad is not None and torch.isfinite(parameter.grad).all(), name
        gradients[name] = float(parameter.grad.float().norm())
    for name in ("perceptual_encoder.reduce.weight", "perceptual_residual.weight",
                 "perceptual.retrieval_blocks.0.q_proj.weight", "cognitive.retrieval_blocks.0.q_proj.weight",
                 "perceptual.gate_fusion_blocks.proj.weight", "cognitive.gate_fusion_blocks.proj.weight",
                 "perceptual.timestep_encoder.mlp.0.weight", "cognitive.timestep_encoder.mlp.0.weight"):
        assert gradients[name] > 0, name
    norm = torch.nn.utils.clip_grad_norm_(parameters, 1.0)
    assert torch.isfinite(norm)
    optimizer.step()
    assert not torch.equal(before, model.memory.perceptual_residual.weight)
    checks["real_group_action_loss_memory_gradients_and_update"] = True
    report["group_loss"] = float(loss)
    report["memory_gradient_norms"] = gradients
    report["gradient_norm_before_clip"] = float(norm)
    del output, loss, optimizer, parameters, before
    model.zero_grad(set_to_none=True)
    gc.collect()
    torch.cuda.empty_cache()
    model.eval()
    model.memory.reset()
    with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
        torch.manual_seed(456)
        expected_first = model.get_action(**single)["action_pred"].cpu().clone()
        single["inputs"]["memory_context"]["timesteps"] = [1]
        torch.manual_seed(789)
        expected_second = model.get_action(**single)["action_pred"].cpu().clone()
    saved = args.output / "checkpoint"
    model.save_pretrained(saved, max_shard_size="5GB")
    processor.save_pretrained(saved)
    del model, memory, state
    gc.collect()
    torch.cuda.empty_cache()
    reloaded, load_info = Gr00tN1d6.from_pretrained(saved, torch_dtype=torch.float32,
        output_loading_info=True, backbone_trainable_params_fp32=False)
    assert not load_info["missing_keys"] and not load_info["unexpected_keys"]
    assert reloaded.config.memory_enabled and reloaded.config.memory_perceptual_dim == 256
    assert not reloaded.memory.perceptual.bank and not reloaded.memory.cognitive.bank
    reloaded.backbone.bfloat16()
    reloaded.cuda().eval()
    with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
        single["inputs"]["memory_context"]["timesteps"] = [0]
        torch.manual_seed(456)
        first = reloaded.get_action(**single)["action_pred"].cpu()
        single["inputs"]["memory_context"]["timesteps"] = [1]
        torch.manual_seed(789)
        second = reloaded.get_action(**single)["action_pred"].cpu()
    assert torch.equal(expected_first, first) and torch.equal(expected_second, second)
    checks["full_checkpoint_reload_actions_exact_with_and_without_history"] = True
    report["saved_checkpoint"] = str(saved.resolve())
    report["peak_cuda_bytes"] = torch.cuda.max_memory_allocated()
    report["elapsed_seconds"] = time.monotonic() - started
    (args.output / "report.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2), flush=True)


if __name__ == "__main__":
    main()
