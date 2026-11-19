"""One actual converted B1K batch + official N1.6 loss/backward/AdamW step.

No mock features or random-weight substitute. Does not save or overwrite the
official checkpoint. Use the independent version environment without DeepSpeed.
"""

import argparse
import json
from pathlib import Path
import time

import numpy as np
import torch

from v3_bodymemory.data.dataset.lerobot_episode_loader import LeRobotEpisodeLoader
from v3_bodymemory.data.dataset.sharded_single_step_dataset import extract_step_data
from v3_bodymemory.data.types import EmbodimentTag
from v3_bodymemory.model.gr00t_n1d6.gr00t_n1d6 import Gr00tN1d6
from v3_bodymemory.model.gr00t_n1d6.processing_gr00t_n1d6 import Gr00tN1d6Processor


def load_real_step(loader, episode_id, frame_index, embodiment=EmbodimentTag.BEHAVIOR_R1_PRO):
    low_dim = loader._load_parquet_data(episode_id)
    modalities = {key: value for key, value in loader.modality_configs.items() if key != "video"}
    step = extract_step_data(low_dim, frame_index, modalities, embodiment)
    video = loader._load_video_data(episode_id, np.array([frame_index]))
    step.images = {key: list(images) for key, images in video.items()}
    return step


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--frame", type=int, default=0)
    parser.add_argument("--output", type=Path, default=Path("outputs/v3_bodymemory/validation/official_batch"))
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    started = time.monotonic()
    torch.manual_seed(42)
    processor = Gr00tN1d6Processor.from_pretrained(args.checkpoint,
        transformers_loading_kwargs={"trust_remote_code": True, "local_files_only": True})
    processor.train()
    modalities = processor.modality_configs["behavior_r1_pro"]
    loader = LeRobotEpisodeLoader(args.dataset, modalities, video_backend="ffmpeg")
    eid = loader.episodes_metadata[0]["episode_index"]
    step = load_real_step(loader, eid, args.frame)
    processed = processor([{"type": "episode_step", "content": step}])
    batch = processor.collator([processed])
    print("REAL_BATCH", {k: list(v.shape) if torch.is_tensor(v) else [list(t.shape) for t in v]
                         for k, v in batch["inputs"].items()}, flush=True)
    model, loading_info = Gr00tN1d6.from_pretrained(args.checkpoint,
        torch_dtype=torch.float32, output_loading_info=True,
        tune_llm=False, tune_visual=False, tune_top_llm_layers=0,
        backbone_trainable_params_fp32=False,
        transformers_loading_kwargs={"trust_remote_code": True, "local_files_only": True})
    if loading_info["missing_keys"] or loading_info["unexpected_keys"] or loading_info.get("mismatched_keys"):
        raise RuntimeError(f"Official checkpoint mismatch: {loading_info}")
    model.backbone.requires_grad_(False)
    model.backbone.bfloat16()
    model.cuda().train()
    parameters = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(parameters, lr=1e-5)
    # Snapshot a small actually used parameter to verify the optimizer changes it.
    monitored = model.action_head.vlln.weight
    before = monitored.detach().clone()
    with torch.autocast("cuda", dtype=torch.bfloat16):
        output = model(**batch)
        loss = output["loss"]
    if not torch.isfinite(loss):
        raise RuntimeError("Non-finite actual GR00T loss")
    loss.backward()
    grad_norm = torch.nn.utils.clip_grad_norm_(parameters, 1.0)
    if not torch.isfinite(grad_norm) or grad_norm <= 0:
        raise RuntimeError("Invalid actual GR00T action-head gradients")
    optimizer.step()
    # BF16 can round very small updates away; optimizer step state and gradients
    # are reported separately from exact parameter changes.
    report = {"checkpoint": str(args.checkpoint.resolve()), "weights": "official",
        "dataset": str(args.dataset.resolve()), "episode_id": eid, "frame_index": args.frame,
        "loss": float(loss), "gradient_norm_before_clip": float(grad_norm),
        "trainable_parameters": sum(p.numel() for p in parameters),
        "optimizer_step": int(optimizer.state[monitored]["step"]),
        "vlln_parameter_changed": not torch.equal(before, monitored),
        "action_shape": list(batch["inputs"]["action"].shape),
        "valid_action_elements": int(batch["inputs"]["action_mask"].sum()),
        "backbone_features_shape": list(output["backbone_features"].shape),
        "peak_cuda_bytes": torch.cuda.max_memory_allocated(), "elapsed_seconds": time.monotonic() - started}
    (args.output / "report.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2), flush=True)


if __name__ == "__main__":
    main()
