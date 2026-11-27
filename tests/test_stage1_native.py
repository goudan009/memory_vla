"""CPU regression for the new isolated native baseline path."""
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from stage1_native_prepare import bounds, deltas, fit, prepare, read, verify_prepared, write, sha
from stage1_native_train import plan, assert_processor

def test_guarded_relative_bounds():
    # Tiny normal corrections; single large tail. Keep original raw range in audit.
    x=np.zeros((10000,3),dtype=np.float32)
    x[:,0]=np.linspace(-.02,.02,len(x))
    x[-1,0]=1.2
    x[:,1]=0.3 # constant
    x[:,2]=np.linspace(-1.,1.,len(x)) # no outlier-driven scaling
    out,audit=bounds(x,"guarded")
    assert audit["accepted"]==[True,False,False]
    assert out["max"][0]<.03 and audit["raw"]["max"][0]>1
    assert out["min"][0]<=0<=out["max"][0]
    assert out["min"][1]==out["max"][1]
    assert bounds(x,"minmax")[0]==audit["raw"]

def test_large_valid_tail_not_silently_clipped():
    x=np.linspace(-.02,.02,10000,dtype=np.float32).reshape(-1,1)
    x[-90:]=1.2
    out,audit=bounds(x,"guarded")
    assert audit["accepted"]==[False] and out["max"][0]>1

def test_same_anchor_for_every_offset():
    q=np.arange(40,dtype=np.float32)[:,None]
    a=q+.25
    arrays=[(1,q,a),(2,q+100,a+100)]
    for h in (0,1,31):
        x=deltas(arrays,slice(0,1),slice(0,1),h)
        assert x.shape==(18,1)
        np.testing.assert_allclose(x,h+.25)

def test_full_prepare_does_not_leak(tmp_path):
    source=tmp_path/"data";base=tmp_path/"GR00T-N1.6-3B"
    base.mkdir()
    write(base/"config.json",dict(action_horizon=50,max_action_dim=128,
          num_inference_timesteps=4,tune_top_llm_layers=4))
    write(base/"processor_config.json",dict(processor_kwargs=dict(max_action_horizon=50,
          max_action_dim=128,max_state_dim=128,use_relative_action=True,use_percentiles=False,
          clip_outliers=True)))
    (base/"model.safetensors").write_bytes(b"untouched")
    info=dict(fps=30,chunks_size=1000,data_path="data/task-{episode_chunk:04d}/episode_{episode_index:08d}.parquet",
              video_path="videos/task-{episode_chunk:04d}/{video_key}/episode_{episode_index:08d}.mp4",
              features={"observation.state":{"shape":[82],"dtype":"float32"},
                        "action":{"shape":[23],"dtype":"float32"}})
    mod={"state":{"trunk_qpos":{"start":78,"end":82},"arm_left_qpos":{"start":18,"end":25},
                  "arm_right_qpos":{"start":48,"end":55}},
         "action":{"base":{"start":0,"end":3},"torso":{"start":3,"end":7},
                   "left_arm":{"start":7,"end":14},"left_gripper":{"start":14,"end":15},
                   "right_arm":{"start":15,"end":22},"right_gripper":{"start":22,"end":23}},
         "video":{k:{"original_key":k} for k in ["head","left","right"]}}
    write(source/"meta/info.json",info);write(source/"meta/modality.json",mod)
    (source/"meta/tasks.jsonl").write_text('{"task_index":40,"task":"test"}\n')
    episodes=[];n=34
    for i in range(200):
        e=400000+i;fields=dict(episode_index=e,episode_chunk=e//1000)
        episodes.append(dict(episode_index=e,length=n,tasks=["test"]))
        p=source/info["data_path"].format(**fields);p.parent.mkdir(parents=True,exist_ok=True)
        q=np.zeros((n,82),np.float32);q[:,0]=999 if i>=180 else i
        a=np.zeros((n,23),np.float32);a[:,7]=.1 if i<180 else 50.
        pd.DataFrame({"observation.state":list(q),"action":list(a),"episode_index":e,
                      "frame_index":np.arange(n),"task_index":40}).to_parquet(p,index=False)
        for v in mod["video"]:
            p=source/info["video_path"].format(**fields,video_key=v)
            p.parent.mkdir(parents=True,exist_ok=True);p.write_bytes(b"synthetic test")
    (source/"meta/episodes.jsonl").write_text("".join(json.dumps(e)+"\n" for e in episodes))
    split=dict(task=40,seed=42,train=list(range(400000,400180)),validation=list(range(400180,400200)))
    write(tmp_path/"split.json",split)
    out=tmp_path/"prepared"
    m=prepare(source,tmp_path/"split.json",base,out,"guarded")
    assert m["train_frames"]==180*n and m["train_valid_anchors"]==180*3
    assert m["raw_hi"][0]==179
    s=read(out/"train/meta/relative_stats.json")
    assert len(s["left_arm"]["min"])==32
    np.testing.assert_allclose(np.array(s["left_arm"]["max"])[:,0],.1)
    assert m==verify_prepared(out)
    assert read(out/"generic_base/config.json")["tune_top_llm_layers"]==0
    assert read(base/"config.json")["tune_top_llm_layers"]==4
    assert (out/"generic_base/model.safetensors").resolve()==base/"model.safetensors"
    assert not (source/"meta/stats.json").exists()
    with pytest.raises(FileExistsError):prepare(source,tmp_path/"split.json",base,out,"guarded")
    # Read saved codec-compatible stats (no actual weights required).
    ckpt=tmp_path/"checkpoint";ckpt.mkdir()
    pc=dict(use_percentiles=False,use_relative_action=True,clip_outliers=True,max_action_dim=128,
            max_state_dim=128,max_action_horizon=50,modality_configs={"behavior_r1_pro":{
            "action":{"delta_indices":list(range(32)),"modality_keys":list(mod["action"])},
            "state":{"delta_indices":[0],"modality_keys":list(mod["state"])},
            "video":{"delta_indices":[0]}}})
    write(ckpt/"processor_config.json",dict(processor_kwargs=pc))
    absolute=read(out/"train/meta/stats.json")
    stats={"relative_action":s}
    for group,feature in [("state","observation.state"),("action","action")]:
        stats[group]={k:{f:v[sl["start"]:sl["end"]] for f,v in absolute[feature].items()} for k,sl in mod[group].items()}
    write(ckpt/"statistics.json",{"behavior_r1_pro":stats})
    assert_processor(ckpt,out)
    stats["relative_action"]["left_arm"]["max"][0][0]=10.
    write(ckpt/"statistics.json",{"behavior_r1_pro":stats})
    with pytest.raises(AssertionError):assert_processor(ckpt,out)
    (out/"train/meta/stats.json").write_text("{}")
    with pytest.raises(ValueError,match="changed"):verify_prepared(out)

def test_plan_batch_and_steps(tmp_path):
    p=plan(SimpleNamespace(output=tmp_path/"o",prepared=tmp_path/"p",official_root=tmp_path/"g"),6000)
    assert p["global_batch"]==64==p["num_gpus"]*p["per_device_batch"]*p["accumulation"]
    assert p["steps"]==6000 and p["inference_steps"]==4 and p["execute_steps"]==8
    assert p["learning_rate"]==1e-4 and p["lr_scheduler"]=="cosine"


def test_workbench_identity_and_canonical_checkpoint(tmp_path):
    from stage1_native_train import canonical_checkpoint, require_workbench_integration
    # Synthetic run number for tests only; never allocated on any device.
    run="v2_1__test02__exp01__run999"
    root=tmp_path/run
    cfg=dict(version="V_2_1",stage="stage1",recipe="native_random_v2",
             run_id=run,run_name=run,preflight_only=False)
    write(root/"config/web_launch.json",cfg)
    assert canonical_checkpoint(root,6000)==root/"run/checkpoints"/(run+"__step006000")
    cfg["run_id"]="v2.1__test01__exp14__run001"
    write(root/"config/web_launch.json",cfg)
    with pytest.raises(ValueError):canonical_checkpoint(root,6000)
    with pytest.raises(RuntimeError,match="BLOCKED_WORKBENCH_INTEGRATION"):
        require_workbench_integration()

def test_no_native_cli_bypass():
    import subprocess,sys
    repo=Path(__file__).resolve().parents[1]
    for name in ("stage1_native_prepare.py","stage1_native_train.py"):
        result=subprocess.run([sys.executable,str(repo/"scripts"/name),"--execute"],
                              capture_output=True,text=True)
        assert result.returncode!=0
        assert "BLOCKED_WORKBENCH_INTEGRATION" in result.stderr
