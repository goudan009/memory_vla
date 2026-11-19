"""Isolated, inference-only GR00T/B1K boundary probe; never contacts a server.

Run each source version in a separate process. Inputs are real demonstration
frames, not simulated observations. Artifacts preserve every inspected boundary.
No optimizer, training, service reset, or simulator is invoked.
"""

import argparse
import ast
import copy
import hashlib
import importlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n")


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


class Canonical(ast.NodeTransformer):
    def visit_Expr(self, node):
        if isinstance(node.value, ast.Constant) and isinstance(node.value.value, str):
            return None
        return self.generic_visit(node)


def source_identity(path):
    path = Path(path)
    if not path.exists():
        return {"exists": False}
    tree = ast.parse(path.read_text())
    tree = Canonical().visit(tree)
    for node in ast.walk(tree):
        if isinstance(node, ast.Name):
            node.id = node.id.replace("v1_groot_memoryvla", "gr00t")
        if isinstance(node, ast.ImportFrom) and node.module:
            node.module = node.module.replace("v1_groot_memoryvla", "gr00t")
        if isinstance(node, ast.alias):
            node.name = node.name.replace("v1_groot_memoryvla", "gr00t")
    return {"exists": True, "sha256": sha(path), "semantic_ast_sha256":
            hashlib.sha256(ast.dump(tree, include_attributes=False).encode()).hexdigest()}


def versions(args):
    roots = dict(item.split("=", 1) for item in args.roots)
    paths = ["model/gr00t_n1d6/gr00t_n1d6.py",
             "model/gr00t_n1d6/processing_gr00t_n1d6.py",
             "model/gr00t_n1d6/image_augmentations.py", "model/modules/dit.py",
             "model/modules/eagle_backbone.py", "policy/gr00t_policy.py",
             "data/state_action/state_action_processor.py",
             "data/state_action/action_chunking.py", "data/state_action/pose.py",
             "configs/data/embodiment_configs.py", "data/embodiment_tags.py",
             "eval/sim/BEHAVIOR/behavior_env.py", "eval/sim/BEHAVIOR/og_teleop_cfg.py",
             "eval/sim/BEHAVIOR/og_teleop_utils.py", "eval/sim/wrapper/multistep_wrapper.py"]
    base = Path(next(iter(roots.values()))) / "model/modules/nvidia/Eagle-Block2A-2B-v2"
    paths += ["model/modules/nvidia/Eagle-Block2A-2B-v2/" + p.name
              for p in base.glob("*.py")]
    result = {rel: {name: source_identity(Path(root)/rel) for name, root in roots.items()}
              for rel in paths}
    write_json(args.output, result)
    for rel, row in result.items():
        print(rel, {k: v.get("semantic_ast_sha256", "missing")[:12] for k, v in row.items()})


def inventory(args):
    result = {"time": time.strftime("%Y-%m-%d %H:%M:%S %z"), "host": os.uname().nodename,
              "processes": [], "checkpoints": {}, "repositories": {}}
    for pid in args.pids:
        p = Path("/proc")/str(pid)
        if not p.exists():
            result["processes"].append({"pid": pid, "exists": False})
            continue
        env = dict(s.split("=", 1) for s in (p/"environ").read_bytes().decode().split("\0") if "=" in s)
        result["processes"].append({"pid": pid, "cmdline": (p/"cmdline").read_bytes().decode().split("\0"),
            "cwd": os.readlink(p/"cwd"), "exe": os.readlink(p/"exe"),
            "environment": {k: env.get(k) for k in ("PYTHONPATH", "CUDA_VISIBLE_DEVICES", "HF_HUB_OFFLINE", "PYTHONNOUSERSITE")}})
    for path in args.checkpoints:
        cp = Path(path)
        row = {"files": {}, "configs": {}}
        for name in ("config.json", "processor_config.json", "statistics.json", "embodiment_id.json", "model.safetensors.index.json"):
            p = cp/name
            if p.exists():
                row["files"][name] = {"sha256": sha(p), "bytes": p.stat().st_size}
                if name != "statistics.json":
                    row["configs"][name] = json.loads(p.read_text())
        result["checkpoints"][str(cp)] = row
    for path in args.repositories:
        root = Path(path)
        data = {"has_git": (root/".git").exists()}
        if data["has_git"]:
            for name, cmd in {"head": ["rev-parse", "HEAD"], "branch": ["branch", "--show-current"],
                              "status": ["status", "--short"], "remotes": ["remote", "-v"]}.items():
                data[name] = subprocess.run(["git", "-C", str(root), *cmd], capture_output=True, text=True).stdout.strip()
        data["model_file"] = source_identity(root/"gr00t/model/gr00t_n1d6/gr00t_n1d6.py")
        result["repositories"][path] = data
    write_json(args.output, result)
    print(json.dumps({k: v for k, v in result.items() if k != "checkpoints"}, indent=2))


def offline(args):
    started = time.monotonic()
    out = Path(args.output).resolve()
    out.mkdir(parents=True, exist_ok=True)
    # Isolate generated remote-code modules; preserve the running services' caches.
    os.environ["HF_MODULES_CACHE"] = str(out/"hf_modules")
    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["TRANSFORMERS_OFFLINE"] = "1"
    os.environ["NO_ALBUMENTATIONS_UPDATE"] = "1"
    import numpy as np
    import torch
    from PIL import Image
    sys.path.insert(0, str(Path(args.source).resolve()))
    pkg = args.package
    policy_mod = importlib.import_module(pkg+".policy.gr00t_policy")
    types_mod = importlib.import_module(pkg+".data.types")
    loader_mod = importlib.import_module(pkg+".data.dataset.lerobot_episode_loader")
    extract = importlib.import_module(pkg+".data.dataset.sharded_single_step_dataset").extract_step_data
    tag = types_mod.EmbodimentTag.BEHAVIOR_R1_PRO
    print("LOAD", args.source, args.checkpoint, flush=True)
    torch.manual_seed(42)
    policy = policy_mod.Gr00tPolicy(tag, args.checkpoint, device="cuda")
    wrapper = policy_mod.Gr00tSimPolicyWrapper(policy)
    policy.model.requires_grad_(False)
    processor = policy.processor
    loader = loader_mod.LeRobotEpisodeLoader(args.dataset, policy.modality_configs, video_backend="ffmpeg")
    episode = args.episode
    data = loader._load_parquet_data(episode)
    frames = args.frames
    arrays = {}
    report = {"source": str(Path(args.source).resolve()), "package": pkg, "checkpoint": args.checkpoint,
              "dataset": args.dataset, "episode": episode, "frames": frames,
              "embodiment_tag": tag.value, "embodiment_id": processor.embodiment_id_mapping[tag.value],
              "model_config": policy.model.config.to_dict(), "samples": [],
              "loaded_sources": {"policy": policy_mod.__file__}, "services_contacted": False}
    captured = {}
    original_get = policy.model.get_action
    original_decode = processor.decode_action
    def capture_get(*a, **kw):
        result = original_get(*a, **kw)
        captured["raw_action"] = result["action_pred"].detach().float().cpu().numpy().copy()
        return result
    def capture_decode(*a, **kw):
        result = original_decode(*a, **kw)
        captured["decoded"] = {k: v.copy() for k, v in result.items()}
        return result
    policy.model.get_action = capture_get
    processor.decode_action = capture_decode
    def stats(x):
        x = np.asarray(x)
        return {"shape": list(x.shape), "dtype": str(x.dtype), "min": float(x.min()),
                "max": float(x.max()), "finite": bool(np.isfinite(x).all()),
                "sha256": hashlib.sha256(x.tobytes()).hexdigest()}
    for frame in frames:
        selected = {k: v for k, v in loader.modality_configs.items() if k != "video"}
        step = extract(data, frame, selected, tag)
        step.images = {k: list(v) for k, v in loader._load_video_data(episode, np.array([frame])).items()}
        observation = {"video": {k: np.asarray(v)[None] for k, v in step.images.items()},
                       "state": {k: np.asarray(v)[None] for k, v in step.states.items()},
                       "language": {policy.language_key: [[step.text]]}}
        flat = {f"{mod}.{k}": v for mod in ("video", "state") for k, v in observation[mod].items()}
        flat[policy.language_key] = [step.text]
        for k, v in flat.items():
            if k != policy.language_key:
                arrays[f"frame{frame}/input/{k}"] = v
        for key, imgs in step.images.items():
            Image.fromarray(np.asarray(imgs[0])).save(out/f"frame{frame}_{key.split('.')[-1]}.png")
        processed = processor([{"type": types_mod.MessageType.EPISODE_STEP.value, "content": copy.deepcopy(step)}])
        batch = processor.collator([processed])["inputs"]
        proc_stats = {}
        for k, v in batch.items():
            if torch.is_tensor(v):
                arr = v.detach().cpu().numpy()
                arrays[f"frame{frame}/processor/{k}"] = arr
                proc_stats[k] = stats(arr)
            elif isinstance(v, (list, tuple)) and all(torch.is_tensor(t) for t in v):
                proc_stats[k] = []
                for i, t in enumerate(v):
                    arr = t.detach().cpu().numpy()
                    arrays[f"frame{frame}/processor/{k}/{i}"] = arr
                    proc_stats[k].append(stats(arr))
        normalized_gt = processor.state_action_processor.apply_action(copy.deepcopy(step.actions), tag.value, copy.deepcopy(step.states))
        reconstructed_gt = processor.state_action_processor.unapply_action(copy.deepcopy(normalized_gt), tag.value, copy.deepcopy(step.states))
        roundtrip = {k: float(np.max(np.abs(step.actions[k]-reconstructed_gt[k]))) for k in step.actions}
        seed = 1234+frame
        policy.reset()
        torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)
        with torch.inference_mode():
            decoded, _ = policy.get_action(copy.deepcopy(observation))
        raw = captured["raw_action"].copy()
        decoded_before_wrapper = {k: v.copy() for k, v in decoded.items()}
        policy.reset()
        torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)
        with torch.inference_mode():
            wrapped, _ = wrapper.get_action(copy.deepcopy(flat))
        wrapper_diff = {k: float(np.max(np.abs(v-wrapped["action."+k]))) for k, v in decoded_before_wrapper.items()}
        # Execute the actual, unchanged env adapter function without importing Isaac Sim.
        env_ast = ast.parse(Path(args.env_adapter).read_text())
        amap = next(n for n in env_ast.body if isinstance(n, ast.Assign) and any(isinstance(t, ast.Name) and t.id=="ACTION_MAP" for t in n.targets))
        fn = next(n for n in env_ast.body if isinstance(n, ast.FunctionDef) and n.name=="preprocess_action")
        ns = {"np": np, "th": torch}
        exec(compile(ast.fix_missing_locations(ast.Module(body=[amap,fn], type_ignores=[])),args.env_adapter,"exec"), ns)
        final = np.stack([ns["preprocess_action"]({k:v[0,i] for k,v in wrapped.items()})["robot_r1"].numpy() for i in range(8)])
        expected = np.concatenate([wrapped["action."+k][0,:8] for k in policy.modality_configs["action"].modality_keys], axis=-1)
        arrays[f"frame{frame}/raw_action"] = raw
        arrays[f"frame{frame}/env_first8"] = final
        groups = {}
        for k, pred in decoded_before_wrapper.items():
            demo = np.asarray(step.actions[k])
            arrays[f"frame{frame}/decoded/{k}"] = pred
            arrays[f"frame{frame}/wrapped/{k}"] = wrapped["action."+k]
            arrays[f"frame{frame}/demo/{k}"] = demo
            groups[k] = {"prediction": stats(pred), "demo": stats(demo),
                         "prediction_first": pred[0,0].tolist(), "demo_first": demo[0].tolist(),
                         "first8_max_jump": float(np.max(np.abs(np.diff(pred[0,:8],axis=0))))}
        row = {"frame": frame, "language": step.text, "policy_input": {k:stats(v) for k,v in flat.items() if k!=policy.language_key},
               "processor": proc_stats, "raw_model_action": stats(raw), "groups": groups,
               "demo_normalization_roundtrip_max_abs": roundtrip, "wrapper_max_abs_delta": wrapper_diff,
               "actual_adapter_first8": stats(final), "adapter_vs_wrapper_max_abs": float(np.max(np.abs(final-expected)))}
        report["samples"].append(row)
        print("SAMPLE", json.dumps(row), flush=True)
        # Same frame/state/noise, canonical simulator instruction: isolate language only.
        if frame == frames[0]:
            sim_flat = copy.deepcopy(flat)
            sim_flat[policy.language_key] = ["Collecting childrens toys."]
            policy.reset(); torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)
            with torch.inference_mode():
                sim_action, _ = wrapper.get_action(sim_flat)
            arrays[f"frame{frame}/sim_language_raw_action"] = captured["raw_action"].copy()
            report["sim_language_comparison"] = {k:float(np.max(np.abs(sim_action["action."+k]-decoded_before_wrapper[k]))) for k in decoded_before_wrapper}
    report["loaded_sources"].update({name: {"path": m.__file__, "sha256": sha(m.__file__)}
        for name,m in list(sys.modules.items()) if getattr(m,"__file__",None) and name.startswith("transformers_modules.") and str(m.__file__).endswith(".py")})
    report["elapsed_s"] = time.monotonic()-started
    report["peak_cuda_bytes"] = torch.cuda.max_memory_allocated()
    np.savez_compressed(out/"boundaries.npz", **arrays)
    write_json(out/"report.json", report)
    print("DONE", out, report["elapsed_s"], flush=True)


def interfaces(args):
    """Exercise actual adapter/wrapper source on demo values, without a simulator."""
    import importlib.util
    from collections import OrderedDict
    from types import SimpleNamespace
    import gymnasium as gym
    import numpy as np
    import pandas as pd
    import torch as th

    def definitions(path, names, namespace):
        tree = ast.parse(Path(path).read_text())
        nodes = []
        for node in tree.body:
            name = node.name if isinstance(node, (ast.FunctionDef, ast.ClassDef)) else None
            if isinstance(node, ast.Assign):
                name = next((t.id for t in node.targets if isinstance(t, ast.Name)), None)
            if name in names:
                if isinstance(node, ast.FunctionDef):
                    node.decorator_list = []
                nodes.append(node)
        exec(compile(ast.fix_missing_locations(ast.Module(body=nodes, type_ignores=[])), str(path), "exec"), namespace)

    ns = {"np": np, "th": th, "OrderedDict": OrderedDict, "gym": gym}
    definitions(args.env_adapter, {"PROPRIOCEPTION_INDICES", "ACTION_MAP", "flatten_obs_dict", "preprocess_obs", "preprocess_action"}, ns)
    definitions(args.converter, {"STATE_SLICES", "ACTION_SLICES", "validate_raw_state"}, ns)
    df = pd.read_parquet(args.raw_parquet, columns=["observation.state", "action"])
    raw = np.stack(df["observation.state"])
    ns["validate_raw_state"](raw)
    result = {"scope": "demo-based adapter and synthetic wrapper checks, NOT live simulation", "source_sha256":
              {str(p): sha(p) for p in (args.env_adapter, args.converter, args.wrapper, args.release_wrapper, args.controller_base)},
              "raw_state_shape": list(raw.shape), "samples": [], "wrapper": {}}
    for frame in args.frames:
        x = raw[frame]
        # Insert the two grasp scalars explicitly requested by the runtime config.
        runtime = np.concatenate([x[:193], [-1.0], x[193:232], [-1.0], x[232:]]).astype(np.float32)
        cameras = {name: {"rgb": th.full((256, 256, 4), i, dtype=th.uint8)} for i, name in enumerate([
            "robot_r1:left_realsense_link:Camera:0", "robot_r1:right_realsense_link:Camera:0", "robot_r1:zed_link:Camera:0"], 11)}
        cameras["proprio"] = th.from_numpy(runtime)
        out = ns["preprocess_obs"](SimpleNamespace(task_instruction="Collecting childrens toys."), {"robot_r1": cameras})
        state_delta = {k: float(np.max(np.abs(out["state."+k]-x[a:b]))) for k, (a,b) in ns["STATE_SLICES"].items()}
        assert max(state_delta.values()) == 0
        action = np.asarray(df["action"].iloc[frame], dtype=np.float32)
        adapted = ns["preprocess_action"]({"action."+k: action[a:b] for k,(a,b) in ns["ACTION_SLICES"].items()})["robot_r1"].numpy()
        assert np.array_equal(action, adapted)
        result["samples"].append({"frame": frame, "reconstructed_runtime_state_dim": len(runtime),
            "state_field_max_abs_delta": state_delta, "action_max_abs_delta": float(np.max(np.abs(action-adapted))),
            "camera_sentinels_head_left_right": [int(out["video.observation.images.rgb."+k+"_256_256"][0,0,0]) for k in ("head", "left_wrist", "right_wrist")]})
    for label, path in (("actual", args.wrapper), ("release", args.release_wrapper)):
        spec = importlib.util.spec_from_file_location("probe_multistep_"+label, path)
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        class DummyEnv(gym.Env):
            observation_space = gym.spaces.Dict({"state.sentinel": gym.spaces.Box(-100, 100, (1,), dtype=np.float32)})
            action_space = gym.spaces.Dict({"action.probe": gym.spaces.Box(-100, 100, (1,), dtype=np.float32)})
            def __init__(self, truncate):
                self.actions, self.truncate = [], truncate
            def reset(self, *, seed=None, options=None):
                self.actions = []
                return {"state.sentinel": np.array([0], dtype=np.float32)}, {}
            def step(self, action):
                self.actions.append(float(action["action.probe"][0]))
                return {"state.sentinel": np.array([len(self.actions)], dtype=np.float32)}, 0.0, False, self.truncate, {"success": False}
        rows = []
        for truncate in (False, True):
            env = DummyEnv(truncate)
            wrapped = mod.MultiStepWrapper(env, np.array([0]), np.array([0]), n_action_steps=8)
            wrapped.reset()
            obs, _, _, truncated, _ = wrapped.step({"action.probe": np.arange(32, dtype=np.float32)[:,None]})
            rows.append({"native_truncate_first_step": truncate, "executed_indices": env.actions,
                         "latest_observation": obs["state.sentinel"].tolist(), "returned_truncated": bool(truncated)})
            if not truncate:
                assert env.actions == list(range(8))
                assert obs["state.sentinel"].reshape(-1)[-1] == 8
        result["wrapper"][label] = rows
    tree = ast.parse(Path(args.controller_base).read_text())
    cls = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name=="BaseController")
    fn = next(n for n in cls.body if isinstance(n, ast.FunctionDef) and n.name=="_preprocess_command")
    cns = {"cb": SimpleNamespace(array=np.array)}
    exec(compile(ast.Module(body=[fn], type_ignores=[]), args.controller_base, "exec"), cns)
    def controller_probe(command, inp, out):
        obj = SimpleNamespace(_command_input_limits=inp, _command_output_limits=out, _command_scale_factor=None)
        return cns["_preprocess_command"](obj, np.array(command, dtype=np.float32)).tolist()
    result["controller_math_only"] = {
        "base_command": [1.0, -1.0, 0.5],
        "base_physical_velocity": controller_probe([1,-1,.5], (np.full(3,-1),np.ones(3)), (-np.array([.75,.75,1]),np.array([.75,.75,1]))),
        "arm_absolute_position_passthrough": controller_probe([.3,-.7], None, None),
        "gripper_limit_source": "runtime joint limits; not instantiated or measured by this probe"}
    write_json(args.output, result)
    print(json.dumps(result, indent=2))


def compare(args):
    import numpy as np
    root = Path(args.root)
    result = {"sources": ["release", "archive", "v1"], "samples": [], "not_a_success_rate_test": True}
    for suffix in ("official", "motion"):
        reports = {s: json.loads((root/f"{s}_{suffix}/report.json").read_text()) for s in result["sources"]}
        arrays = {s: np.load(root/f"{s}_{suffix}/boundaries.npz") for s in result["sources"]}
        for sample in reports["release"]["samples"]:
            f = sample["frame"]
            keys = [k for k in arrays["release"].files if k.startswith(f"frame{f}/") and not k.endswith("sim_language_raw_action")]
            comparisons = {s: {k: float(np.max(np.abs(arrays["release"][k].astype(np.float64)-arrays[s][k].astype(np.float64)))) for k in keys} for s in ("archive", "v1")}
            assert all(v == 0 for row in comparisons.values() for v in row.values())
            result["samples"].append({"frame": f, "boundary_max_abs_delta_against_release":
                {s: max(row.values()) for s,row in comparisons.items()}, "boundaries_compared": len(keys),
                "wrapper_max_abs_delta": sample["wrapper_max_abs_delta"], "adapter_max_abs_delta": sample["adapter_vs_wrapper_max_abs"],
                "demo_normalization_roundtrip_max_abs": max(sample["demo_normalization_roundtrip_max_abs"].values()),
                "groups": sample["groups"]})
        for v in arrays.values():
            v.close()
    write_json(args.output, result)
    print("All sampled source-version boundaries are exactly equal:", [r["frame"] for r in result["samples"]])


def replay(args):
    """Replay saved live requests on another source, with exactly the saved noise seed."""
    import numpy as np
    out = Path(args.output).resolve()
    out.mkdir(parents=True, exist_ok=True)
    os.environ["HF_MODULES_CACHE"] = str(out/"hf_modules")
    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["TRANSFORMERS_OFFLINE"] = "1"
    import torch
    sys.path.insert(0, str(Path(args.source).resolve()))
    mod = importlib.import_module(args.package+".policy.gr00t_policy")
    types = importlib.import_module(args.package+".data.types")
    torch.manual_seed(42)
    policy = mod.Gr00tPolicy(types.EmbodimentTag.BEHAVIOR_R1_PRO, args.checkpoint, device="cuda")
    policy.model.requires_grad_(False)
    wrapped = mod.Gr00tSimPolicyWrapper(policy)
    original = policy.model.get_action
    captures = {}
    def hook(*a, **kw):
        inputs = kw.get("inputs", a[0] if a else {})
        for k,v in inputs.items():
            if torch.is_tensor(v):
                captures["processor/"+k] = v.detach().float().cpu().numpy().copy()
            elif isinstance(v, (list, tuple)) and all(torch.is_tensor(t) for t in v):
                for i,t in enumerate(v):
                    captures[f"processor/{k}/{i}"] = t.detach().float().cpu().numpy().copy()
        result = original(*a, **kw)
        captures["raw_action"] = result["action_pred"].detach().float().cpu().numpy().copy()
        return result
    policy.model.get_action = hook
    rows = []
    for path in sorted(Path(args.requests).glob("request*.npz")):
        index = int(path.stem.replace("request", ""))
        saved = np.load(path)
        obs = {k[len("input/"):]: saved[k] for k in saved.files if k.startswith("input/")}
        for k,v in list(obs.items()):
            if k.startswith("annotation."):
                obs[k] = v.tolist()
        seed = json.loads(path.with_suffix(".json").read_text())["seed"]
        captures.clear()
        torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)
        with torch.inference_mode():
            action, _ = wrapped.get_action(obs)
        raw_delta = float(np.max(np.abs(saved["raw_action"]-captures["raw_action"])))
        decoded_delta = {k:float(np.max(np.abs(saved["wrapped/"+k]-v))) for k,v in action.items()}
        np.savez_compressed(out/path.name, **captures)
        rows.append({"request": index, "seed": seed, "raw_max_abs_delta_against_live": raw_delta,
                     "decoded_max_abs_delta_against_live": decoded_delta})
        print("REPLAY", index, raw_delta, max(decoded_delta.values()), flush=True)
        saved.close()
    write_json(out/"report.json", {"source": args.source, "checkpoint": args.checkpoint, "requests": rows,
                                    "loaded_policy": mod.__file__, "services_contacted": False})


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    p = sub.add_parser("versions")
    p.add_argument("--roots", nargs="+", required=True, help="name=package-directory")
    p.add_argument("--output", required=True)
    p.set_defaults(run=versions)
    p = sub.add_parser("inventory")
    p.add_argument("--pids", nargs="*", type=int, default=[])
    p.add_argument("--checkpoints", nargs="*", default=[])
    p.add_argument("--repositories", nargs="*", default=[])
    p.add_argument("--output", required=True)
    p.set_defaults(run=inventory)
    p = sub.add_parser("offline")
    p.add_argument("--source", required=True)
    p.add_argument("--package", default="gr00t")
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--dataset", required=True)
    p.add_argument("--episode", type=int, default=210020)
    p.add_argument("--frames", type=int, nargs="+", default=[0,96,192])
    p.add_argument("--env-adapter", required=True)
    p.add_argument("--output", required=True)
    p.set_defaults(run=offline)
    p = sub.add_parser("interfaces")
    for key in ("env-adapter", "converter", "wrapper", "release-wrapper", "controller-base", "raw-parquet", "output"):
        p.add_argument("--"+key, required=True)
    p.add_argument("--frames", type=int, nargs="+", default=[0,96,192,800,808,1448,1456,8192])
    p.set_defaults(run=interfaces)
    p = sub.add_parser("compare")
    p.add_argument("--root", required=True)
    p.add_argument("--output", required=True)
    p.set_defaults(run=compare)
    p = sub.add_parser("replay")
    for key in ("source", "checkpoint", "requests", "output"):
        p.add_argument("--"+key, required=True)
    p.add_argument("--package", default="gr00t")
    p.set_defaults(run=replay)
    args = parser.parse_args()
    args.run(args)


if __name__ == "__main__":
    main()
