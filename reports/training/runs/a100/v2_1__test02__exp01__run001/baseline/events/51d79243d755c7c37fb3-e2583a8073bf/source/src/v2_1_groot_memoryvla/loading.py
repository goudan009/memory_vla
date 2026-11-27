"""Shared V2_1 checkpoint precision contract for training and serving."""
import json
from pathlib import Path

import torch


def load_base(path, stage, device="cuda"):
    from v1_groot_memoryvla.model.gr00t_n1d6.gr00t_n1d6 import Gr00tN1d6

    if stage not in ("stage1", "continued", "memory"):
        raise ValueError("invalid V2_1 stage")
    saved = json.loads((Path(path) / "config.json").read_text())
    if saved.get("memory_enabled") or saved.get("finetuning_lora"):
        raise ValueError("expected original no-memory full checkpoint")
    base, info = Gr00tN1d6.from_pretrained(
        path, memory_enabled=False, tune_llm=False, tune_visual=False,
        tune_top_llm_layers=0, backbone_trainable_params_fp32=False,
        torch_dtype=torch.float32, output_loading_info=True)
    if any(info.get(k) for k in ("missing_keys", "unexpected_keys", "mismatched_keys", "error_msgs")):
        raise ValueError("base checkpoint load mismatch: " + str(info))
    base.backbone.requires_grad_(False).bfloat16().eval()
    tune = stage != "memory"
    base.action_head.set_trainable_parameters(tune, tune, tune)
    # Never round the action expert through BF16 before restoring FP32.
    base.action_head.float()
    base.to(device=device)
    if stage == "memory":
        base.requires_grad_(False).eval()
    else:
        base.action_head.train()
    return base
