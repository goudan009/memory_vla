"""Read-only V1.2 contracts for the existing workbench.

The two benchmark protocols are deliberately distinct. An official 2026
checkpoint is not a 2025 Task40 checkpoint. All absolute paths, run identities,
GPU assignments and output allocations must come from the existing workbench.
"""
import json
import hashlib
import math
import re
import subprocess
from pathlib import Path

VERSION = "V_1_2"
# VERSION remains the historical identity imported by the retired 2025 candidates.
# New native 2026 jobs use the user's concrete version within the V0 family.
B1K2026_VERSION = "V_0_0"
B1K2026_DISPLAY_VERSION = "V0_0"
B1K2026_FAMILY = "V0"
FORK_URL = "https://github.com/wensi-ai/Isaac-GR00T"
FORK_REVISION = "ace36d935b376fbf25cd56371e23877b95407c40"
MODEL_ID = "nvidia/GR00T-N1.7-3B"
BACKBONE_ID = "nvidia/Cosmos-Reason2-2B"
OFFICIAL_RADIO_URL = "https://drive.google.com/file/d/1OXNm3SPLvWOSJR1e8In6xHMHxOYDp789/view?usp=sharing"
B1K2026 = "b1k2026_websocket"
B1K2025 = "b1k2025_zmq_candidate"
RADIO_TASK_DESCRIPTION = "Turn on the radio receiver that's on the table in the living room."
OFFICIAL_DEFAULT_PROMPT = "pick up the object and place it on the table"
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
    "gr00t/eval/eval_b1k_wrapper.py": "f9167c74e2f3dca325d60005abb78bf63f56e351",
}


def read(path):
    return json.loads(Path(path).read_text())


def absolute(value):
    path = Path(value)
    if not path.is_absolute():
        raise ValueError("paths must be explicitly resolved by the workbench")
    return path.resolve()


def executable(value):
    """Make an executable path absolute without resolving its venv symlink."""
    path = Path(value)
    if not path.is_absolute():
        raise ValueError("paths must be explicitly resolved by the workbench")
    return path.absolute()


def inspect_upstream(root):
    """Check source identity without modifying a checkout or importing torch."""
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
    if not names or any(not isinstance(n,str) or Path(n).name != n or not n.endswith('.safetensors') for n in names):
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
    saved = read(processor/"processor_config.json")
    if saved.get("processor_class") != "Gr00tN1d7Processor":
        raise ValueError("expected native N1.7 processor class")
    pc = saved["processor_kwargs"]
    stats = read(processor/"statistics.json")
    if not isinstance(stats,dict) or not stats:
        raise ValueError("missing processor statistics")
    if pc.get("max_action_horizon") != config.get("action_horizon") or pc.get("max_action_dim") != config.get("max_action_dim") or pc.get("max_state_dim") != config.get("max_state_dim"):
        raise ValueError("model/processor padded dimension mismatch")
    result = dict(path=str(checkpoint),processor=str(processor),role=role,weights=weights,
                  model_horizon=config.get("action_horizon"),max_action_dim=config.get("max_action_dim"),
                  max_state_dim=config.get("max_state_dim"),inference_steps=config.get("num_inference_timesteps"))
    if role in ('generic','family_only'):
        # This is a family/schema check, not evidence of the pretrained origin.
        return result
    if role not in ("official_2026_radio", "trained_2026"):
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
    if not pc.get("use_relative_action"):
        raise ValueError("expected native relative arm/torso representation")
    configs=modes["action"].get("action_configs") or []
    if len(configs)!=len(ACTION_GROUPS):
        raise ValueError("missing action representation contract")
    for key,item in zip(ACTION_GROUPS,configs):
        relative=key in ("torso","left_arm","right_arm")
        if (item.get("rep")!=("RELATIVE" if relative else "ABSOLUTE")
            or item.get("type")!="NON_EEF" or item.get("format")!="DEFAULT"
            or (relative and item.get("state_key")!=key)
            or bool(item.get("is_gripper"))!=("gripper" in key)):
            raise ValueError("incorrect physical action representation: "+key)
    for group,layout in (("state",STATE_GROUPS_2026),("action",ACTION_GROUPS),
                         ("relative_action",{k:ACTION_GROUPS[k] for k in ("torso","left_arm","right_arm")})):
        table=stats["new_embodiment"].get(group,{})
        for key,(lo,hi) in layout.items():
            entry=table.get(key,{})
            # Upstream accepts vector absolute stats, and per-offset relative
            # stats. Do not accept empty or nonfinite normalization ranges.
            def vector(x):
                return isinstance(x,list) and len(x)==hi-lo and all(type(v) in (int,float) and math.isfinite(v) for v in x)
            a,b=entry.get("min"),entry.get("max")
            pairs=[(a,b)] if vector(a) and vector(b) else list(zip(a,b)) if isinstance(a,list) and isinstance(b,list) and len(a)==len(b)==16 and group=="relative_action" else []
            if not pairs or any(not vector(x) or not vector(y) or any(v>w for v,w in zip(x,y)) for x,y in pairs):
                raise ValueError("invalid normalization range: "+group+"/"+key)
    result.update(valid_action_steps=16,physical_action_dim=23,
                  raw_state_dim=61,selected_state_dim=sum(b-a for a,b in STATE_GROUPS_2026.values()),protocol=B1K2026)
    return result


def checkpoint_digest(checkpoint):
    """Hash actual weights + processor config/statistics; no trust in labels."""
    checkpoint=absolute(checkpoint)
    processor=checkpoint if (checkpoint/"processor_config.json").is_file() else checkpoint/"processor"
    files=[checkpoint/"config.json",*(checkpoint/n for n in model_files(checkpoint)),
           processor/"processor_config.json",processor/"statistics.json"]
    for optional in (checkpoint/"model.safetensors.index.json",processor/"embodiment_id.json"):
        if optional.is_file():files.append(optional)
    digest=hashlib.sha256()
    for f in sorted(files):
        h=hashlib.sha256()
        with f.open("rb") as stream:
            for block in iter(lambda:stream.read(1024*1024),b""):h.update(block)
        digest.update(str(f.relative_to(checkpoint)).encode()+b"\0"+h.digest())
    return digest.hexdigest()


def official_serve_spec(*, python, upstream, checkpoint, host, port, gpu_index, verified_radio_sha256):
    """Native official reference process spec; never launches a service."""
    return _native_serve_spec(python=python, upstream=upstream, checkpoint=checkpoint,
        host=host, port=port, gpu_index=gpu_index, verified_sha256=verified_radio_sha256,
        role='official_2026_radio', method='official', task='turning_on_radio')


def _native_serve_spec(*, python, upstream, checkpoint, host, port, gpu_index,
                       verified_sha256, role, method, task):
    python = executable(python)
    upstream, checkpoint = map(absolute, (upstream, checkpoint))
    if not python.is_file() or not isinstance(host,str) or not host.strip():
        raise ValueError("explicit target Python and bind host required")
    if type(port) is not int or not 1<=port<=65535 or type(gpu_index) is not int or gpu_index<0:
        raise ValueError("invalid workbench port/GPU")
    source = inspect_upstream(upstream)
    identity = inspect_checkpoint(checkpoint,role=role)
    actual=checkpoint_digest(checkpoint)
    # A local Agent must establish this digest from the actual official
    # download and archive its provenance. Matching modality alone cannot
    # distinguish a radio checkpoint from any other R1Pro task.
    if actual!=verified_sha256:
        raise ValueError("checkpoint download identity mismatch")
    argv = [str(python),"-u",str(upstream/"scripts/b1k/serve_b1k.py"),
            "--model-path",str(checkpoint),"--modality-config-path",str(upstream/"examples/b1k/r1pro.py"),
            "--embodiment-tag","NEW_EMBODIMENT","--device","cuda","--control-mode","temporal_ensemble",
            "--host",host,"--port",str(port)]
    return dict(version=B1K2026_VERSION,benchmark='b1k2026',display_version=B1K2026_DISPLAY_VERSION,
                method_family=B1K2026_FAMILY,
                method=method,task_name=task,protocol=B1K2026,
                source=source,checkpoint=identity,checkpoint_sha256=actual,argv=argv,cwd=str(upstream),
                env={"PYTHONPATH":str(upstream),"CUDA_VISIBLE_DEVICES":str(gpu_index),
                     "PYTHONNOUSERSITE":"1","PYTHONDONTWRITEBYTECODE":"1",
                     "HF_HUB_OFFLINE":"1","TRANSFORMERS_OFFLINE":"1"},
                health_path="/healthz",host=host,port=port,gpu_index=gpu_index,
                # The official server holds one shared temporal buffer. Use
                # one environment per service until isolation is proven.
                max_envs_per_service=1,control_mode="temporal_ensemble",rtc=False)


def official_eval_spec(*, python, behavior_root, output, endpoint, instance, rollouts):
    """Native 2026 evaluator. Output must already be allocated by workbench."""
    if endpoint.get('task_name') != 'turning_on_radio' or endpoint.get('method') != 'official':
        raise ValueError('must bind a verified ready official 2026 radio service')
    return _native_eval_spec(python=python,behavior_root=behavior_root,output=output,
                             endpoint=endpoint,instance=instance,rollouts=rollouts)


def _native_eval_spec(*, python, behavior_root, output, endpoint, instance, rollouts):
    python = executable(python)
    behavior_root, output = map(absolute, (behavior_root, output))
    if not python.is_file() or not behavior_root.is_dir():
        raise ValueError("separate B1K2026 environment/source required")
    if not output.is_dir() or any(output.iterdir()):
        raise ValueError("expected fresh workbench-reserved evaluation output")
    if endpoint.get("protocol")!=B1K2026 or endpoint.get("state")!="ready":
        raise ValueError("must bind a verified ready official 2026 radio service")
    if endpoint.get('benchmark', 'b1k2026') != 'b1k2026':
        raise ValueError('native endpoint belongs exclusively to benchmark b1k2026')
    if endpoint.get("version") not in (B1K2026_VERSION, VERSION) or not re.fullmatch(r"[0-9a-f]{64}",str(endpoint.get("checkpoint_sha256",""))):
        raise ValueError("bind actual V1.2 official checkpoint hash before evaluation")
    if not isinstance(endpoint.get("host"),str) or not endpoint["host"].strip() or type(endpoint.get("port")) is not int or not 1<=endpoint["port"]<=65535:
        raise ValueError("invalid observed policy endpoint")
    if type(instance) is not int or not 0<=instance<20 or type(rollouts) is not int or rollouts<1:
        raise ValueError("explicit public-test index and rollout budget required")
    # Local pinned evaluator version and flags must be checked during the
    # workbench preflight (not assumed equivalent to 2025).
    return dict(version=endpoint['version'],benchmark='b1k2026',group=endpoint['method'],task_name=endpoint['task_name'],
                protocol=B1K2026,output=str(output),instance_indices=[instance],
                num_rollouts=rollouts,num_envs=1,policy_identity=dict(endpoint),
                argv=[str(python),"-m","omnigibson.eval.eval",
                      "--task-name",endpoint['task_name'],"--mode","public_test",
                      "--instance-indices",str(instance),"--num-rollouts",str(rollouts),"--num-envs","1",
                      "--host",str(endpoint["host"]),"--port",str(endpoint["port"]),
                      "--output-dir",str(output),"--write-video"],
                cwd=str(behavior_root),control_mode="temporal_ensemble",rtc=False)


TRAINING_STATE_FILES = ('optimizer.pt', 'scheduler.pt', 'rng_state.pth',
                        'trainer_state.json', 'dataset_state.json')


def file_sha256(path):
    value = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b''):
            value.update(chunk)
    return value.hexdigest()


def inspect_training_checkpoint(checkpoint):
    """Validate a published checkpoint, including optimizer/data state integrity."""
    checkpoint = absolute(checkpoint)
    identity = read(checkpoint / 'b1k2026_training.json')
    if (identity.get('version') != B1K2026_VERSION or identity.get('benchmark') != 'b1k2026'
            or identity.get('complete') is not True or identity.get('parameter_update_verified') is not True
            or not re.fullmatch(r'v0_0__test\d{2}__exp\d{2}__run(?!0{3,}$)[0-9]{3,}', identity.get('run_id', ''))
            or type(identity.get('optimizer_step')) is not int or identity['optimizer_step'] < 1
            or identity.get('source_revision') != FORK_REVISION):
        raise ValueError('incomplete or inconsistent self-trained checkpoint identity')
    if not identity.get('tasks') or not set(identity['tasks']) <= {0,40}:
        raise ValueError('self-trained task whitelist missing or unsupported')
    inspect_checkpoint(checkpoint, role='trained_2026')
    if checkpoint_digest(checkpoint) != identity.get('checkpoint_sha256'):
        raise ValueError('self-trained checkpoint hash mismatch')
    for name in TRAINING_STATE_FILES:
        if file_sha256(checkpoint / name) != identity.get('training_state_sha256', {}).get(name):
            raise ValueError('training state hash mismatch: ' + name)
    if read(checkpoint / 'trainer_state.json')['global_step'] != identity['optimizer_step']:
        raise ValueError('checkpoint optimizer step mismatch')
    return identity


def trained_serve_spec(*, python, upstream, checkpoint, host, port, gpu_index,
                       task_index, verified_sha256):
    from .b1k2026_data import TASKS
    identity = inspect_training_checkpoint(checkpoint)
    if task_index not in identity['tasks']:
        raise ValueError('requested task was not in this training manifest')
    spec = _native_serve_spec(python=python,upstream=upstream,checkpoint=checkpoint,
        host=host,port=port,gpu_index=gpu_index,verified_sha256=verified_sha256,
        role='trained_2026',method='self_trained',task=TASKS[task_index])
    spec['training_identity'] = identity
    return spec


def trained_eval_spec(*, python, behavior_root, output, endpoint, instance, rollouts):
    from .b1k2026_data import TASKS
    identity = endpoint.get('training_identity', {})
    if (endpoint.get('method') != 'self_trained' or not identity.get('complete')
            or endpoint.get('task_name') not in [TASKS[i] for i in identity.get('tasks', []) if i in TASKS]
            or endpoint.get('checkpoint_sha256') != identity.get('checkpoint_sha256')):
        raise ValueError('must bind the exact self-trained task/checkpoint identity')
    return _native_eval_spec(python=python,behavior_root=behavior_root,output=output,
                             endpoint=endpoint,instance=instance,rollouts=rollouts)
