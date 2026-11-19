#!/usr/bin/env python3
"""Instrument the local OpenVLA-based MemoryVLA without changing its source.

Uses the full-size, randomly initialized DINOv2/SigLIP/LLaMA-7B/DiT-L.
Only pretrained weight/config/tokenizer loading is replaced during construction;
all actual forward, memory, loss, generation and sampling methods are original.
The tokenizer must be the original LLaMA/OpenVLA tokenizer.model.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import importlib.util
import json
import platform
import sys
import time
from pathlib import Path
from unittest.mock import patch

import numpy as np
import torch
from PIL import Image


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--tokenizer", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=Path(__file__).with_name("forward_trace.json"))
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[2]
    source = root / "src/references_code/memoryvla"
    sys.path.insert(0, str(source))

    import timm
    from transformers import LlamaConfig, LlamaTokenizerFast
    from prismatic.models.backbones.llm.llama2 import LLaMa2LLMBackbone
    from prismatic.models.backbones.vision.dinosiglip_vit import DinoSigLIPViTBackbone
    from prismatic.models.vlms.prismatic import PrismaticVLM

    # Bypass vla/__init__.py's unrelated TensorFlow training-dataset imports.
    spec = importlib.util.spec_from_file_location("memoryvla_reference_probe", source / "vla/memory_vla.py")
    memory_module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = memory_module
    spec.loader.exec_module(memory_module)

    torch.set_num_threads(8)
    torch.manual_seed(20261113)
    np.random.seed(20261113)
    device = torch.device(args.device)
    dtype = torch.bfloat16
    tokenizer = LlamaTokenizerFast(vocab_file=str(args.tokenizer), model_max_length=2048, padding_side="right")
    config = LlamaConfig(
        vocab_size=32000, hidden_size=4096, intermediate_size=11008,
        num_hidden_layers=32, num_attention_heads=32, num_key_value_heads=32,
        max_position_embeddings=4096, rms_norm_eps=1e-5,
        bos_token_id=1, eos_token_id=2, tie_word_embeddings=False,
    )
    config._attn_implementation = "sdpa"
    trace = {
        "scope": "Full-size original modules, random weights, synthetic RGB/actions; no trained policy evaluation",
        "checkpoint_loaded": False,
        "seed": 20261113,
        "environment": {"python": platform.python_version(), "device": str(device), "dtype": str(dtype),
                        "gpu": torch.cuda.get_device_name(device),
                        "packages": {name: importlib.metadata.version(name) for name in
                                     ["torch", "torchvision", "transformers", "timm", "tokenizers", "numpy"]}},
        "construction_overrides": [
            "timm.create_model: pretrained=False; all original model identifiers and depths retained",
            "AutoConfig.from_pretrained: explicit LLaMA-2 7B config; 32 layers, 4096 hidden, 11008 MLP",
            "AutoTokenizer.from_pretrained: local original tokenizer.model",
            "vla/memory_vla.py loaded directly to skip dataset imports",
            "BF16 parameters, SDPA backend, training autocast as in base_strategy.py; no backward or optimizer step",
            "Synthetic action q01/q99 statistics, not a real robot calibration",
        ],
        "tokenizer_sha256": hashlib.sha256(args.tokenizer.read_bytes()).hexdigest(),
        "source_sha256": {str(p.relative_to(source)): hashlib.sha256(p.read_bytes()).hexdigest()
                          for p in sorted(source.rglob("*.py"))},
        "events": [],
    }
    stage = "construction"
    full_stats_seen = set()

    def describe(value, stats=False):
        if isinstance(value, torch.Tensor):
            item = {"shape": list(value.shape), "dtype": str(value.dtype), "device": str(value.device)}
            if stats and value.numel():
                flat = value.detach().float().reshape(-1)
                item.update(finite=bool(torch.isfinite(flat).all()), mean=float(flat.mean()),
                            std=float(flat.std(unbiased=False)), min=float(flat.min()), max=float(flat.max()),
                            first_values=flat[:8].cpu().tolist())
            return item
        if isinstance(value, np.ndarray):
            return {"shape": list(value.shape), "dtype": str(value.dtype), "values": value.tolist()}
        if isinstance(value, dict):
            return {k: describe(v, stats) for k, v in value.items()}
        if isinstance(value, (tuple, list)):
            return [describe(v, stats) for v in value]
        if isinstance(value, (int, float, str, bool)) or value is None:
            return value
        return str(type(value))

    def event(name, inputs, outputs=None, **extra):
        stats = (stage, name) not in full_stats_seen
        full_stats_seen.add((stage, name))
        trace["events"].append({"stage": stage, "name": name,
                                "inputs": describe(inputs, stats), "outputs": describe(outputs, stats), **extra})

    def hook(name):
        def capture(module, inputs, kwargs, output):
            # Avoid copying every hidden layer to CPU; record their shapes separately.
            if hasattr(output, "hidden_states"):
                event(name, {"args": inputs, "kwargs": kwargs}, {
                    "logits": output.logits,
                    "last_hidden_state": output.hidden_states[-1] if output.hidden_states else None,
                    "loss": output.loss,
                }, hidden_state_count=len(output.hidden_states or []))
            else:
                event(name, {"args": inputs, "kwargs": kwargs}, output)
        return capture

    def bank_state(bank):
        return {str(eid): [{"timestep": int(t), "shape": list(f.shape), "requires_grad": f.requires_grad}
                           for t, f in items] for eid, items in bank.bank.items()}

    def trace_bank(bank, name):
        original = bank.process_batch

        def process(*a, **kw):
            before = bank_state(bank)
            out = original(*a, **kw)
            event(name, {"args": a, "kwargs": kw}, out, before=before, after=bank_state(bank))
            return out
        bank.process_batch = process

    print("Building full-size random-weight model", flush=True)
    start = time.perf_counter()
    real_create_model = timm.create_model

    def random_vision(*a, **kw):
        kw["pretrained"] = False
        return real_create_model(*a, **kw)

    old_dtype = torch.get_default_dtype()
    torch.set_default_dtype(dtype)
    with torch.device(device), patch("timm.create_model", side_effect=random_vision), \
         patch("transformers.AutoConfig.from_pretrained", return_value=config), \
         patch("transformers.AutoTokenizer.from_pretrained", return_value=tokenizer):
        vision = DinoSigLIPViTBackbone("dinosiglip-vit-so-224px", "resize-naive")
        llm = LLaMa2LLMBackbone("llama2-7b-pure", inference_mode=True, use_flash_attention_2=False)
        vlm = PrismaticVLM("prism-dinosiglip-224px+7b", vision, llm,
                           arch_specifier="no-align+fused-gelu-mlp")
        model = memory_module.MemoryVLA(
            vlm, action_model_type="DiT-L", token_size=4096, future_action_window_size=15,
            norm_stats={"synthetic": {"action": {
                "q01": [-0.1, -0.1, -0.1, -0.2, -0.2, -0.2, 0.0],
                "q99": [0.1, 0.1, 0.1, 0.2, 0.2, 0.2, 1.0],
                "mask": [True, True, True, True, True, True, False],
            }}},
        )
    torch.set_default_dtype(old_dtype)
    model.eval()
    trace["construction_seconds"] = time.perf_counter() - start
    trace["model"] = {
        "parameters": sum(p.numel() for p in model.parameters()),
        "parameters_by_child": {n: sum(p.numel() for p in m.parameters()) for n, m in model.named_children()},
        "dino_layers": len(vision.dino_featurizer.blocks), "dino_width": vision.dino_featurizer.embed_dim,
        "siglip_layers": len(vision.siglip_featurizer.blocks), "siglip_width": vision.siglip_featurizer.embed_dim,
        "llama_config": llm.llm.config.to_dict(),
        "dit_layers": len(model.action_model.net.blocks), "dit_width": 1024, "dit_heads": 16,
        "mem_length": model.mem_length, "retrieval_layers": model.retrieval_layers,
        "update_fused": model.update_fused, "group_size": model.group_size,
    }
    modules = {
        "vision.dino": vision.dino_featurizer, "vision.siglip": vision.siglip_featurizer,
        "vision.concat": vision, "vlm.projector": vlm.projector, "vlm.llama": llm,
        "per_compr": model.per_compr, "cog.gate": model.cog_mem_bank.gate_fusion_blocks,
        "per.gate": model.per_mem_bank.gate_fusion_blocks,
        "cog.gate_logits": model.cog_mem_bank.gate_fusion_blocks.proj,
        "per.gate_logits": model.per_mem_bank.gate_fusion_blocks.proj,
        "dit.net": model.action_model.net,
        "dit.x_embedder": model.action_model.net.x_embedder,
        "dit.t_embedder": model.action_model.net.t_embedder,
        "dit.z_embedder": model.action_model.net.z_embedder,
        "dit.per_embedder": model.action_model.net.per_token_embedder,
        "dit.block0": model.action_model.net.blocks[0],
        "dit.block0.per_attn": model.action_model.net.blocks[0].per_attn,
        "dit.final": model.action_model.net.final_layer,
    }
    for prefix, bank in [("cog", model.cog_mem_bank), ("per", model.per_mem_bank)]:
        trace_bank(bank, prefix + ".process_batch")
        for i, block in enumerate(bank.retrieval_blocks):
            modules[f"{prefix}.retrieval.{i}"] = block
        modules[prefix + ".time_embedding"] = bank.timestep_encoder
    for name, module in modules.items():
        module.register_forward_hook(hook(name), with_kwargs=True)

    # Generated probe input, not a dataset sample or an illustration asset.
    yy, xx = np.mgrid[:224, :224]
    rgb = np.stack([xx % 256, yy % 256, (xx + yy) % 256], axis=-1).astype(np.uint8)
    image = Image.fromarray(rgb, "RGB")
    instruction = "put the red block into the bowl"
    pixels = {k: v.unsqueeze(0).to(device=device, dtype=dtype) for k, v in vision.image_transform(image).items()}
    prompt = vlm.get_prompt_builder()
    prompt.add_turn("human", f"What action should the robot take to {instruction}?")
    inference_ids = torch.cat([tokenizer(prompt.get_prompt(), return_tensors="pt").input_ids,
                               torch.tensor([[29871, 2]], dtype=torch.long)], dim=1).to(device)
    training_prompt = vlm.get_prompt_builder()
    training_prompt.add_turn("human", f"What action should the robot take to {instruction}?")
    training_prompt.add_turn("gpt", "")
    training_ids = tokenizer(training_prompt.get_prompt(), return_tensors="pt").input_ids.to(device)
    trace["example_input"] = {"instruction": instruction, "image_kind": "synthetic coordinate color grid",
                              "rgb_shape": list(rgb.shape), "rgb_sha256": hashlib.sha256(rgb.tobytes()).hexdigest(),
                              "inference_prompt": prompt.get_prompt(), "inference_ids": inference_ids.cpu().tolist(),
                              "inference_tokens": tokenizer.convert_ids_to_tokens(inference_ids[0].tolist()),
                              "training_prompt": training_prompt.get_prompt(), "training_ids": training_ids.cpu().tolist(),
                              "pixels": describe(pixels, True)}

    print("Training forward: B=2, same episode at observations 0 and 8, R=4", flush=True)
    stage = "training_forward"
    model.train()
    ids = training_ids.repeat(2, 1)
    masks = ids.ne(tokenizer.pad_token_id)
    labels = ids.clone()
    labels[:, :-1] = -100
    actions = torch.linspace(-0.8, 0.8, 2 * 16 * 7, device=device).reshape(2, 16, 7).to(dtype)
    # No backward: this task measures forward tensors and state only.
    with torch.no_grad(), torch.autocast("cuda", dtype=dtype):
        loss, output = model(input_ids=ids, attention_mask=masks,
                             pixel_values={k: v.repeat(2, 1, 1, 1) for k, v in pixels.items()},
                             labels=labels, actions=actions, action_masks=torch.ones(2, 16, device=device, dtype=torch.bool),
                             episode_ids=np.array([7, 7]), timesteps=np.array([0, 8]),
                             output_hidden_states=True, return_dict=True, repeated_diffusion_steps=4)
    trace["training_result"] = {"loss": float(loss), "loss_finite": bool(torch.isfinite(loss)),
                                "returned_logits": describe(output.logits),
                                "hidden_states_count": len(output.hidden_states),
                                "last_hidden_state": describe(output.hidden_states[-1]),
                                "vlm_loss": float(output.loss), "input_ids": describe(ids),
                                "actions": describe(actions), "returned_tuple": ["scalar diffusion loss", "CausalLMOutputWithPast"]}
    assert bool(torch.isfinite(loss))
    del loss, output
    model.eval()

    trace["inference_results"] = []
    for name, first in [("inference_first", "True"), ("inference_history", "False"), ("inference_reset", "True")]:
        print(name, flush=True)
        stage = name
        begin = time.perf_counter()
        result, normalized = model.predict_action(image, instruction, unnorm_key="synthetic", cfg_scale=1.5,
                                                   use_ddim=True, num_ddim_steps=10, episode_first_frame=first)
        torch.cuda.synchronize(device)
        record = {"stage": stage, "seconds": time.perf_counter() - begin,
                  "actions": describe(result), "normalized_actions": describe(normalized),
                  "finite": bool(np.isfinite(result).all()), "cur_timestep": model.cur_timestep,
                  "cog_bank": bank_state(model.cog_mem_bank), "per_bank": bank_state(model.per_mem_bank)}
        assert result.shape == normalized.shape == (16, 7) and record["finite"]
        trace["inference_results"].append(record)

    print("Memory-only sweep: 18 writes, full 16-entry capacity", flush=True)
    trace["capacity_sweep"] = []
    for bank in [model.cog_mem_bank, model.per_mem_bank]:
        bank.reset()
    with torch.no_grad():
        for t in range(18):
            stage = f"memory_capacity_{t}"
            cog = torch.randn(1, 1, 4096, device=device, dtype=dtype)
            per = torch.randn(1, 256, 256, device=device, dtype=dtype)
            co = model.cog_mem_bank.process_batch(cog, [0], [t])
            po = model.per_mem_bank.process_batch(per, [0], [t])
            trace["capacity_sweep"].append({"write": t, "cog_length": len(model.cog_mem_bank.bank[0]),
                                            "per_length": len(model.per_mem_bank.bank[0]),
                                            "cog_timestamps": [int(s) for s, _ in model.cog_mem_bank.bank[0]],
                                            "per_timestamps": [int(s) for s, _ in model.per_mem_bank.bank[0]],
                                            "cog_output": describe(co), "per_output": describe(po)})
        assert len(model.cog_mem_bank.bank[0]) == len(model.per_mem_bank.bank[0]) == 16

        # Small hand-checkable example uses the original consolidation method.
        stage = "controlled_merge"
        bank = memory_module.CogMemBank("group", 16, 8, mem_length=3, retrieval_layers=0,
                                       use_timestep_pe=False, fusion_type="add")
        features = [torch.tensor([[1., 0., 0., 0., 0., 0., 0., 0.]]),
                    torch.tensor([[2., 0., 0., 0., 0., 0., 0., 0.]]),
                    torch.tensor([[0., 1., 0., 0., 0., 0., 0., 0.]]),
                    torch.tensor([[0., 0., 1., 0., 0., 0., 0., 0.]])]
        for t, f in enumerate(features):
            bank._memory_consolidate(0, f, t)
        trace["controlled_merge"] = {"capacity": 3, "inputs": [f.tolist() for f in features],
                                     "output": [{"timestep": int(t), "feature": f.tolist()} for t, f in bank.bank[0]]}
        assert [t for t, _ in bank.bank[0]] == [0, 2, 3]
        assert bank.bank[0][0][1][0, 0].item() == 1.5

    trace["peak_gpu_allocated_gib"] = torch.cuda.max_memory_allocated(device) / 1024 ** 3
    trace["total_seconds"] = time.perf_counter() - start
    trace["assertions_passed"] = True
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(trace, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps({"output": str(args.output), "loss": trace["training_result"]["loss"],
                      "parameters": trace["model"]["parameters"], "peak_gpu_gib": trace["peak_gpu_allocated_gib"],
                      "seconds": trace["total_seconds"], "events": len(trace["events"])}, indent=2), flush=True)


if __name__ == "__main__":
    main()
