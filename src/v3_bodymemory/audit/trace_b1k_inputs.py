"""Read-only B1K input audit; writes diagnostic artifacts, never model weights.

Run in the existing N1.6 model environment from the repository root:
  PYTHONDONTWRITEBYTECODE=1 python src/v3_bodymemory/audit/trace_b1k_inputs.py
The model package initializer imports the training pipeline. This probe creates
namespace packages for model/ and model/gr00t_n1d6/ to import the unchanged
processor without importing Trainer/DeepSpeed. No tensors or processor methods
are mocked. --vision executes random-weight Eagle for shape evidence only.
"""

import argparse
import csv
import hashlib
import importlib
import json
from pathlib import Path
import re
import sys
import types
import urllib.request


ROOT = Path(__file__).resolve().parents[3]
VERSION = ROOT / "src/v3_bodymemory"
VENDOR = VERSION / "model/modules/nvidia/Eagle-Block2A-2B-v2"


def json_get(url):
    with urllib.request.urlopen(url, timeout=45) as response:
        return json.load(response)


def import_processor():
    sys.path.insert(0, str(ROOT / "src"))
    for suffix in ("model", "model.gr00t_n1d6"):
        name = "v3_bodymemory." + suffix
        module = types.ModuleType(name)
        module.__path__ = [str(VERSION.joinpath(*suffix.split(".")))]
        sys.modules[name] = module
    return importlib.import_module(
        "v3_bodymemory.model.gr00t_n1d6.processing_gr00t_n1d6"
    )


def shape(value):
    if hasattr(value, "shape"):
        return list(value.shape)
    if isinstance(value, (list, tuple)):
        return [shape(v) for v in value]
    return type(value).__name__


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=VERSION / "audit/artifacts")
    parser.add_argument("--dataset", type=Path, default=ROOT / "assets/datasets/b1k2025")
    parser.add_argument("--vision", action="store_true")
    parser.add_argument("--revision", default="300db814db8ab5dd010026d5631f280048d06b91",
                        help="Pinned official checkpoint configuration revision")
    parser.add_argument("--without-deepspeed", action="store_true",
                        help="Probe-only: disable Transformers DeepSpeed availability check")
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    import numpy as np
    import pandas as pd
    import av
    import torch

    repo_id = "nvidia/GR00T-N1.6-BEHAVIOR1k"
    revision = args.revision
    configs = {}
    for name in ("config.json", "processor_config.json"):
        configs[name] = json_get(f"https://huggingface.co/{repo_id}/resolve/{revision}/{name}")
        (args.output / name).write_text(json.dumps(configs[name], indent=2) + "\n")
    pcfg = configs["processor_config.json"]["processor_kwargs"]
    print("CHECKPOINT", repo_id, revision, flush=True)
    print("PROCESSOR_KEYS", list(pcfg), flush=True)
    info = json.loads((args.dataset / "meta/info.json").read_text())
    cameras = ["head", "left_wrist", "right_wrist"]
    source = next(
        p for p in sorted((args.dataset / "data/task-0021").glob("*.parquet"))
        if all((args.dataset / f"videos/task-0021/observation.images.rgb.{c}/{p.stem}.mp4").is_file() for c in cameras)
    )
    df = pd.read_parquet(source)
    tasks = {v["task_index"]: v["task"] for v in map(json.loads, (args.dataset / "meta/tasks.jsonl").read_text().splitlines())}
    language = tasks[int(df.iloc[0]["task_index"])]
    images = {}
    videos = {}
    for camera in cameras:
        path = args.dataset / f"videos/task-0021/observation.images.rgb.{camera}/{source.stem}.mp4"
        with av.open(str(path)) as container:
            stream = container.streams.video[0]
            frame = next(container.decode(video=0))
            image = frame.to_ndarray(format="rgb24")
            key = f"observation.images.rgb.{camera}_256_256"
            images[key] = [image]
            videos[camera] = {
                "path": str(path), "rgb_shape": shape(image),
                "fps": float(stream.average_rate), "frames": stream.frames,
                "first_pts": frame.pts, "time_base": str(stream.time_base),
                "frame_sha256": hashlib.sha256(image.tobytes()).hexdigest(),
            }
    module = import_processor()
    parameters = set(__import__("inspect").signature(module.Gr00tN1d6Processor).parameters)
    kwargs = {k: v for k, v in pcfg.items() if k in parameters}
    kwargs["transformers_loading_kwargs"] = {"trust_remote_code": True, "local_files_only": True}
    processor = module.Gr00tN1d6Processor(**kwargs)

    # Raw challenge state/action layout differs from converted GR00T data. Use
    # real raw vectors to report shape, and exercise the exact visual-language
    # branch independently of unavailable converted normalization metadata.
    raw_state = np.asarray(df.iloc[0]["observation.state"])
    raw_action = np.asarray(df.iloc[0]["action"])
    # Exact language formalization in Gr00tN1d6Processor.__call__.
    processed_language = re.sub(r"[^\w\s]", "", language.lower()) if processor.formalize_language else language
    visual = {}
    image_keys = pcfg["modality_configs"]["behavior_r1_pro"]["video"]["modality_keys"]
    for mode in ("eval", "train"):
        getattr(processor, mode)()
        transform = processor.eval_image_transform if mode == "eval" else processor.train_image_transform
        visual[mode] = processor._get_vlm_inputs(
            image_keys=image_keys, images=images, masks=None,
            language=processed_language,
            image_transform=transform,
        )
    batch = processor.collator([visual["eval"]])["inputs"]
    train_batch = processor.collator([visual["train"]])["inputs"]
    # Same real images and a shorter instruction create actual left padding.
    short = processor._get_vlm_inputs(
        image_keys=image_keys, images=images, masks=None,
        language="collect toys", image_transform=processor.eval_image_transform,
    )
    padded = processor.collator([visual["eval"], short])["inputs"]
    tokenizer = processor.processor.tokenizer
    image_token_id = json.loads((VENDOR / "config.json").read_text())["image_token_index"]
    token_reports = {}
    for mode, item in (("eval", batch), ("train", train_batch), ("padded", padded)):
        records = []
        for b in range(item["input_ids"].shape[0]):
            ids = item["input_ids"][b].tolist()
            mask = item["attention_mask"][b].tolist()
            tokens = tokenizer.convert_ids_to_tokens(ids)
            for index, (token_id, token, valid) in enumerate(zip(ids, tokens, mask)):
                records.append({"batch": b, "position": index, "input_id": token_id,
                                "token": token, "attention_mask": valid,
                                "image_mask": int(token_id == image_token_id)})
        token_reports[mode] = records
        with (args.output / f"tokens_{mode}.tsv").open("w") as output:
            writer = csv.DictWriter(output, fieldnames=records[0].keys(), delimiter="\t")
            writer.writeheader()
            writer.writerows(records)
        print(mode, "SHAPES", {k: shape(v) for k, v in item.items()}, flush=True)
        for b in range(item["input_ids"].shape[0]):
            row = [r for r in records if r["batch"] == b]
            print(mode, b, "input_ids", [r["input_id"] for r in row], flush=True)
            print(mode, b, "tokens", [r["token"] for r in row], flush=True)
            print(mode, b, "attention_mask", [r["attention_mask"] for r in row], flush=True)
            print(mode, b, "image_mask", [r["image_mask"] for r in row], flush=True)

    result = {
        "checkpoint": {"repo": repo_id, "revision": revision, "weights_loaded": False},
        "dataset": {"parquet": str(source), "rows": len(df), "columns": list(df.columns),
                    "fps": info["fps"], "timestamp_deltas": sorted(set(np.round(np.diff(df.timestamp.to_numpy()), 7).tolist())),
                    "index_contiguous": bool(np.all(np.diff(df["index"]) == 1)),
                    "episode_ids": sorted(df.episode_index.unique().tolist()),
                    "raw_state_shape": shape(raw_state), "raw_action_shape": shape(raw_action),
                    "language": language, "processed_language": processed_language, "videos": videos,
                    "missing_converted_metadata": [n for n in ("modality.json", "stats.json") if not (args.dataset / "meta" / n).exists()]},
        "processor": {"padding_side": tokenizer.padding_side, "eos_token": tokenizer.eos_token,
                      "eos_token_id": tokenizer.eos_token_id, "pad_token": tokenizer.pad_token,
                      "pad_token_id": tokenizer.pad_token_id, "image_token_id": image_token_id,
                      "templates_equal": visual["eval"]["vlm_content"]["text"] == visual["train"]["vlm_content"]["text"],
                      "template": visual["eval"]["vlm_content"]["text"],
                      "token_ids_train_eval_equal": torch.equal(batch["input_ids"], train_batch["input_ids"]),
                      "shapes": {k: shape(v) for k, v in batch.items()},
                      "tokens": token_reports},
    }
    if args.vision:
        # This process executes plain torch inference, without DeepSpeed.
        # The installed DeepSpeed package otherwise fails at import because
        # CUDA_HOME/nvcc are absent. This changes no model tensor operations.
        if args.without_deepspeed:
            import transformers.integrations.deepspeed as ds_integration
            ds_integration.is_deepspeed_available = lambda: False
            result["probe_environment"] = {"deepspeed_availability_overridden": True}
        from transformers import AutoConfig, AutoModel
        cfg = AutoConfig.from_pretrained(VENDOR, trust_remote_code=True, local_files_only=True)
        print("BUILD RANDOM-WEIGHT EAGLE (NO CHECKPOINT WEIGHTS)", flush=True)
        torch.manual_seed(0)
        # Vendor's unused scratch initializer refers to undefined Siglip2Model.
        # Skip that initializer and fill parameters explicitly for this shape-
        # only run; this is not a replacement for checkpoint initialization.
        from transformers.modeling_utils import no_init_weights
        with torch.device("cuda:0"), no_init_weights():
            model = AutoModel.from_config(cfg, trust_remote_code=True, torch_dtype=torch.bfloat16)
        while len(model.language_model.model.layers) > configs["config.json"]["select_layer"]:
            model.language_model.model.layers.pop(-1)
        with torch.no_grad():
            for name, parameter in model.named_parameters():
                if name.endswith("bias"):
                    parameter.zero_()
                elif parameter.ndim == 1:
                    parameter.fill_(1)
                else:
                    parameter.normal_(mean=0, std=0.02)
        model.eval()
        hooks = []
        observed = {}
        captured = {}
        def vision_hook(mod, inputs, output):
            observed["vision_last_hidden_state"] = shape(output.last_hidden_state)
            observed["spatial_shapes"] = output.spatial_shapes.tolist()
        def mlp_pre(mod, inputs):
            observed["mlp1_input"] = shape(inputs[0])
        def mlp_post(mod, inputs, output):
            observed["mlp1_output"] = shape(output)
            captured["visual_embeddings"] = output.detach().reshape(-1, output.shape[-1])
        def language_pre(mod, inputs, kwargs):
            selected = device_batch["input_ids"] == image_token_id
            observed["vision_to_llm_positions_exact"] = bool(torch.equal(
                kwargs["inputs_embeds"][selected], captured["visual_embeddings"]))
            observed["llm_input_embeddings"] = shape(kwargs["inputs_embeds"])
        hooks.append(model.vision_model.register_forward_hook(vision_hook))
        hooks.append(model.mlp1.register_forward_pre_hook(mlp_pre))
        hooks.append(model.mlp1.register_forward_hook(mlp_post))
        hooks.append(model.language_model.register_forward_pre_hook(language_pre, with_kwargs=True))
        # Match EagleBackbone.forward's exact key filter (image_sizes is
        # processor metadata, not an Eagle model forward argument).
        device_batch = {k: v.cuda() if torch.is_tensor(v) else [x.cuda().bfloat16() for x in v]
                        for k, v in batch.items() if k in ("input_ids", "attention_mask", "pixel_values")}
        with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
            output = model(**device_batch, output_hidden_states=True)
        observed["llm_hidden_state"] = shape(output.hidden_states[-1])
        eos_positions = ((device_batch["input_ids"] == tokenizer.eos_token_id) & device_batch["attention_mask"].bool()).nonzero()
        cognitive_position = int(eos_positions[-1, 1])
        observed["cognitive_position"] = cognitive_position
        observed["cognitive_shape"] = shape(output.hidden_states[-1][:, cognitive_position:cognitive_position + 1])
        observed["hidden_states_finite"] = bool(output.hidden_states[-1].isfinite().all())
        observed["llm_layers"] = len(model.language_model.model.layers)
        observed["weights"] = "random initialization; shape validation only"
        observed["peak_cuda_bytes"] = torch.cuda.max_memory_allocated()
        result["eagle_shape_probe"] = observed
        print("EAGLE_SHAPES", observed, flush=True)
        for hook in hooks:
            hook.remove()
        padded_device = {k: v.cuda() if torch.is_tensor(v) else [x.cuda().bfloat16() for x in v]
                         for k, v in padded.items() if k in ("input_ids", "attention_mask", "pixel_values")}
        with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
            padded_output = model(**padded_device, output_hidden_states=True)
        observed["padded_llm_hidden_state"] = shape(padded_output.hidden_states[-1])
        observed["long_row_single_vs_batch_max_abs_diff"] = float(
            (padded_output.hidden_states[-1][0] - output.hidden_states[-1][0]).abs().max())
    (args.output / "input_audit.json").write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n")
    print("REPORT", args.output / "input_audit.json", flush=True)


if __name__ == "__main__":
    main()
