"""Read-only V1.2 contracts for the existing workbench.

The two benchmark protocols are deliberately distinct. An official 2026
checkpoint is not a 2025 Task40 checkpoint. All absolute paths, run identities,
GPU assignments and output allocations must come from the existing workbench.
"""
import json
import subprocess
from pathlib import Path

VERSION = "V_1_2"
FORK_URL = "https://github.com/wensi-ai/Isaac-GR00T"
FORK_REVISION = "ace36d935b376fbf25cd56371e23877b95407c40"
MODEL_ID = "nvidia/GR00T-N1.7-3B"
BACKBONE_ID = "nvidia/Cosmos-Reason2-2B"
OFFICIAL_RADIO_URL = "https://drive.google.com/file/d/1OXNm3SPLvWOSJR1e8In6xHMHxOYDp789/view?usp=sharing"
B1K2026 = "b1k2026_websocket"
B1K2025 = "b1k2025_zmq_candidate"
STATE_GROUPS_2026 = {"base_qvel": (0,3), "torso": (53,57), "left_arm": (3,10),
                     "left_gripper": (24,26), "right_arm": (28,35), "right_gripper": (49,51)}
ACTION_GROUPS = {"base": (0,3), "torso": (3,7), "left_arm": (7,14),
                 "left_gripper": (14,15), "right_arm": (15,22), "right_gripper": (22,23)}
UPSTREAM_FILES = {
    "scripts/b1k/serve_b1k.py": "6ae5314095339551e83f8785fb3844e89605b361",
    "scripts/b1k/train_b1k.py": "f5f60df9a4def3eaac1cd3b93270a94a11ce9961",
    "examples/b1k/r1pro.py": "d54a400f3f48348eaefff1b9bf0f34caa51721af",
    "examples/b1k/r1pro.json": "9287b28bade2af68a4e7442fcec1167845ee0432",
    "gr00t/policy/websocket_b1k_server.py": "cb3191307561e805c5b83c0196b4f1d80693cfb5",
}


def read(path):
    return json.loads(Path(path).read_text())


def absolute(value):
    path = Path(value)
    if not path.is_absolute():
        raise ValueError("paths must be explicitly resolved by the workbench")
    return path.resolve()


def inspect_upstream(root):
    """Check source identity without modifying a checkout or importing torch."""
    import hashlib
    root = absolute(root)
    head = subprocess.check_output(["git", "-C", str(root), "rev-parse", "HEAD"],
                                   text=True, timeout=10).strip()
    if head != FORK_REVISION:
        raise ValueError("N1.7 challenge fork revision mismatch")
    changes = subprocess.check_output(["git", "-C", str(root), "status", "--porcelain",
                                       "--untracked-files=no"], text=True, timeout=10)
    if changes.strip():
        raise ValueError("upstream tracked source changed; preserve official reference")
    for rel, expected in UPSTREAM_FILES.items():
        data = (root/rel).read_bytes()
        blob = hashlib.sha1(("blob %d\0" % len(data)).encode()+data).hexdigest()
        if blob != expected:
            raise ValueError("upstream file changed: "+rel)
    return dict(root=str(root), revision=head, verified_files=dict(UPSTREAM_FILES))


def model_files(checkpoint):
    checkpoint = absolute(checkpoint)
    index = checkpoint/"model.safetensors.index.json"
    names = sorted(set(read(index)["weight_map"].values())) if index.is_file() else ["model.safetensors"]
    if not names or any(Path(n).name != n for n in names):
        raise ValueError("unsafe or empty model index")
    for name in names:
        f = checkpoint/name
        if not f.is_file() or f.stat().st_size == 0:
            raise ValueError("incomplete model weights: "+name)
    return names


def inspect_checkpoint(checkpoint, *, role):
    """Do not infer a 2026 task, embodiment or source from a directory name."""
    checkpoint = absolute(checkpoint)
    config = read(checkpoint/"config.json")
    if str(config.get("model_type","")).lower() != "gr00tn1d7":
        raise ValueError("expected GR00T N1.7; N1.6 is not compatible")
    weights = model_files(checkpoint)
    processor = checkpoint if (checkpoint/"processor_config.json").is_file() else checkpoint/"processor"
    pc = read(processor/"processor_config.json")["processor_kwargs"]
    stats = read(processor/"statistics.json")
    result = dict(path=str(checkpoint),processor=str(processor),role=role,weights=weights,
                  model_horizon=config.get("action_horizon"),max_action_dim=config.get("max_action_dim"),
                  max_state_dim=config.get("max_state_dim"),inference_steps=config.get("num_inference_timesteps"))
    if role == "generic":
        return result
    if role != "official_2026_radio":
        raise ValueError("checkpoint role not yet supported")
    modes = pc.get("modality_configs",{}).get("new_embodiment")
    if not modes or "new_embodiment" not in stats:
        raise ValueError("official radio requires the fine-tuned NEW_EMBODIMENT processor/statistics")
    expected = {"state":list(STATE_GROUPS_2026),"action":list(ACTION_GROUPS),
                "video":["head","left_wrist","right_wrist"],"language":["annotation.human.task_description"]}
    for kind, keys in expected.items():
        if modes[kind]["modality_keys"] != keys:
            raise ValueError("2026 modality order mismatch: "+kind)
        if modes[kind]["delta_indices"] != (list(range(16)) if kind=="action" else [0]):
            raise ValueError("2026 modality horizon mismatch: "+kind)
    result.update(valid_action_steps=16,physical_action_dim=23,
                  raw_state_dim=61,selected_state_dim=25,protocol=B1K2026)
    return result


def official_serve_spec(*, python, upstream, checkpoint, host, port, gpu_index):
    """Native official reference process spec; never launches a service."""
    python, upstream, checkpoint = map(absolute,(python,upstream,checkpoint))
    if not python.is_file() or not isinstance(host,str) or not host.strip():
        raise ValueError("explicit target Python and bind host required")
    if type(port) is not int or not 1<=port<=65535 or type(gpu_index) is not int or gpu_index<0:
        raise ValueError("invalid workbench port/GPU")
    source = inspect_upstream(upstream)
    identity = inspect_checkpoint(checkpoint,role="official_2026_radio")
    argv = [str(python),"-u",str(upstream/"scripts/b1k/serve_b1k.py"),
            "--model-path",str(checkpoint),"--modality-config-path",str(upstream/"examples/b1k/r1pro.py"),
            "--embodiment-tag","NEW_EMBODIMENT","--device","cuda","--control-mode","temporal_ensemble",
            "--host",host,"--port",str(port)]
    return dict(version=VERSION,method="official",task_name="turning_on_radio",protocol=B1K2026,
                source=source,checkpoint=identity,argv=argv,cwd=str(upstream),
                env={"PYTHONPATH":str(upstream),"CUDA_VISIBLE_DEVICES":str(gpu_index),
                     "PYTHONNOUSERSITE":"1","PYTHONDONTWRITEBYTECODE":"1",
                     "HF_HUB_OFFLINE":"1","TRANSFORMERS_OFFLINE":"1"},
                health_path="/healthz",host=host,port=port,gpu_index=gpu_index,
                # The official server holds one shared temporal buffer. Use
                # one environment per service until isolation is proven.
                max_envs_per_service=1,control_mode="temporal_ensemble",rtc=False)


def official_eval_spec(*, python, behavior_root, output, endpoint, instance, rollouts):
    """Native 2026 evaluator. Output must already be allocated by workbench."""
    python,behavior_root,output=map(absolute,(python,behavior_root,output))
    if not python.is_file() or not behavior_root.is_dir():
        raise ValueError("separate B1K2026 environment/source required")
    if not output.is_dir() or any(output.iterdir()):
        raise ValueError("expected fresh workbench-reserved evaluation output")
    if endpoint.get("protocol")!=B1K2026 or endpoint.get("task_name")!="turning_on_radio" or endpoint.get("state")!="ready":
        raise ValueError("must bind a verified ready official 2026 radio service")
    if not endpoint.get("checkpoint_sha256"):
        raise ValueError("bind actual checkpoint hash before evaluation")
    if type(instance) is not int or not 0<=instance<20 or type(rollouts) is not int or rollouts<1:
        raise ValueError("explicit public-test index and rollout budget required")
    # Local pinned evaluator version and flags must be checked during the
    # workbench preflight (not assumed equivalent to 2025).
    return dict(version=VERSION,group="official",task_name="turning_on_radio",
                protocol=B1K2026,output=str(output),instance_indices=[instance],
                num_rollouts=rollouts,num_envs=1,policy_identity=dict(endpoint),
                argv=[str(python),"-m","omnigibson.eval.eval",
                      "--task-name","turning_on_radio","--mode","public_test",
                      "--instance-indices",str(instance),"--num-rollouts",str(rollouts),"--num-envs","1",
                      "--host",str(endpoint["host"]),"--port",str(endpoint["port"]),
                      "--output-dir",str(output),"--write-video"],
                cwd=str(behavior_root),control_mode="temporal_ensemble",rtc=False)
