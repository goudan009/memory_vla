"""Offline FK audit, fresh-process checkpoint / policy / retrieval verification."""
import argparse
import json
from pathlib import Path
import time
import numpy as np
import pandas as pd
import torch
from .data import BodyMapper,slice_body
from .kinematics import quat_matrix
from .model import V2Model
from .pilot import influence,difference,context,save_json
from v1_groot_memoryvla.experiment.first_training import slice_features


def audit_fk(mapper,root):
    data=root/"assets/datasets/b1k2025_groot_task0_v1_200"
    df=pd.read_parquet(data/"data/task-0000/episode_00000010.parquet")
    raw=pd.read_parquet(root/"assets/datasets/b1k2025/data/task-0000/episode_00000010.parquet")
    meta=json.loads((data/"meta/modality.json").read_text())
    frames=[0,8,16,24,60,120,240,480,960]
    body=mapper.episode(df,meta,frames,raw)
    poses,valid=mapper.graph.fk(body["q"],body["q_mask"])
    state=np.stack(df["observation.state"])[frames]
    result={"frames":frames,"urdf_sha256":mapper.graph.sha256,"nodes":len(mapper.graph.names)}
    for side in ("left","right"):
        idx=mapper.graph.link_index[f"{side}_eef_link"]
        def field(name):
            m=meta["state"][name];return torch.tensor(state[:,m["start"]:m["end"]],dtype=torch.float32)
        pos=(poses[:,idx,:3,3]-field(f"eef_{side}_pos")).norm(dim=-1)
        rot=(poses[:,idx,:3,:3]-quat_matrix(field(f"eef_{side}_quat"))).abs()
        result[side]={"max_position_error_m":float(pos.max()),"max_rotation_matrix_error":float(rot.max())}
        assert valid[:,idx].all() and pos.max()<.003 and rot.max()<.01
    return result


@torch.no_grad()
def benchmark(model,seq,device):
    result={}
    count=len(seq["timesteps"])
    variants=["v1","v2a","v2b","v2c"]
    for variant in variants:
        old=model.config.variant;model.config.variant=variant
        measurements=[]
        for repeat in range(6):
            model.reset()
            features=[slice_features(seq["features"],i,i+1,device) for i in range(count)]
            ids=seq["input_ids"].to(device)
            bodies=[slice_body(seq["body"],i,i+1,device) for i in range(count)]
            torch.cuda.synchronize();start=time.perf_counter()
            with torch.autocast("cuda",dtype=torch.bfloat16):
                for i in range(count): model.fuse(features[i],ids[i:i+1],context(seq,i),bodies[i])
            torch.cuda.synchronize()
            if repeat: measurements.append((time.perf_counter()-start)*1000/count)
        result[variant]={"median_fusion_ms":float(np.median(measurements)),"samples":5*count}
        model.config.variant=old
    return result


def run(args):
    device=torch.device(args.device);root=Path.cwd()
    loader=V2Model.load_extension if (args.checkpoint/"manifest.json").exists() else V2Model.load
    model=loader(args.checkpoint).to(device).eval()
    model.base.requires_grad_(False)
    seq=torch.load(args.sequence,weights_only=False,map_location="cpu")
    result={"fk":audit_fk(model.mapper,root)}
    expected=np.load(args.expected)["normal"]
    normal,diag=influence(model,seq,device)
    result["fresh_checkpoint_action_exact"]=np.array_equal(expected,normal.numpy())
    assert result["fresh_checkpoint_action_exact"],"Fresh checkpoint changed predictions"
    interventions={}
    for condition in ("current_body","past_body","past_action","empty_embodied","vl_query"):
        pred,d=influence(model,seq,device,condition)
        row={"action":difference(normal[:,:8,:23],pred[:,:8,:23])}
        for key in ("query","attention"):
            if d[key].shape==diag[key].shape: row[key]=difference(diag[key],d[key])
        interventions[condition]=row
    result["influence"]=interventions
    assert all(v["action"]["changed"] for v in interventions.values())
    assert interventions["current_body"]["query"]["changed"]
    assert interventions["current_body"]["attention"]["changed"]
    result["fusion_cost"]=benchmark(model,seq,device)
    # Exercise actual RGB -> processor -> frozen Eagle -> V1/V2 -> full head ->
    # decode on a prior live input capture. No running simulation or server.
    from .policy import V2Policy,V2SimPolicyWrapper
    from v1_groot_memoryvla.model.gr00t_n1d6.processing_gr00t_n1d6 import Gr00tN1d6Processor
    processor=Gr00tN1d6Processor.from_pretrained(model.source)
    wrapper=V2SimPolicyWrapper(V2Policy(model,processor,device))
    capture=root/"outputs/diagnostics/groot_b1k_20261110"
    result["policy_requests"]=[]
    for i in (0,1):
        recorded=np.load(capture/f"live_server_400/request{i:03d}.npz")
        observation={k.removeprefix("input/"):recorded[k] for k in recorded.files if k.startswith("input/")}
        # NPZ serializes the live protocol's language tuple as an ndarray.
        observation["annotation.human.coarse_action"]=observation["annotation.human.coarse_action"].tolist()
        options={"observation_timestamps":[i*8/30],"session_id":"v2-offline-replay",
                 "episode_ids":["capture"],"env_ids":[0]}
        if i:
            executed=np.load(capture/"live_sim_400/step0007.npz")["env_action"].reshape(23)
            options.update(executed_actions=[executed],executed_action_timestamps=[7/30])
        actions,_=wrapper.get_action(observation,options)
        shapes={k:list(v.shape) for k,v in actions.items()}
        assert sum(v.shape[-1] for v in actions.values())==23
        assert all(np.isfinite(v).all() for v in actions.values())
        result["policy_requests"].append(shapes)
    wrapper.reset()
    save_json(args.output,result)
    print(json.dumps(result,indent=2),flush=True)


if __name__=="__main__":
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint",required=True,type=Path)
    parser.add_argument("--sequence",required=True,type=Path)
    parser.add_argument("--expected",required=True,type=Path)
    parser.add_argument("--output",required=True,type=Path)
    parser.add_argument("--device",default="cuda:0")
    run(parser.parse_args())
