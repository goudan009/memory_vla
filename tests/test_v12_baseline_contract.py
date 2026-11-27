import json
from pathlib import Path
import sys

import pytest

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/"src"))
sys.path.insert(0,str(ROOT/"train"))
from run_identity import MODEL_VERSIONS, validate_run_identity
from v1_2_groot_baseline import contracts as c
from v1_2_groot_baseline.data_contract import STATE_WIDTHS,CAMERAS,audit_2025,modality_source
from v1_2_groot_baseline.training_contract import batch_accounting, native_train_candidate


def write(path,value):
    path.parent.mkdir(parents=True,exist_ok=True)
    path.write_text(json.dumps(value))


def test_independent_version():
    assert MODEL_VERSIONS["V_1_2"]["backend"]=="v1_2"
    assert MODEL_VERSIONS["V_1_2"]["launch_ready"] is False
    assert validate_run_identity("v1_2__test99__exp99__run999","V_1_2")["model_version"]=="V_1_2"
    with pytest.raises(ValueError):
        validate_run_identity("v2_1__test99__exp99__run999","V_1_2")
    assert "V_2_2" not in MODEL_VERSIONS


def checkpoint(root,model="Gr00tN1d7"):
    write(root/"config.json",dict(model_type=model,action_horizon=40,max_action_dim=132,max_state_dim=132))
    modes={k:dict(modality_keys=v,delta_indices=list(range(16)) if k=="action" else [0])
           for k,v in dict(state=list(c.STATE_GROUPS_2026),action=list(c.ACTION_GROUPS),
                video=["head","left_wrist","right_wrist"],language=["annotation.human.task_description"]).items()}
    modes["action"]["action_configs"]=[dict(rep="RELATIVE" if k in ("torso","left_arm","right_arm") else "ABSOLUTE",
        type="NON_EEF",format="DEFAULT",state_key=k if k in ("torso","left_arm","right_arm") else None,
        is_gripper="gripper" in k) for k in c.ACTION_GROUPS]
    write(root/"processor_config.json",{"processor_class":"Gr00tN1d7Processor","processor_kwargs":{
        "modality_configs":{"new_embodiment":modes},"max_action_horizon":40,
        "max_action_dim":132,"max_state_dim":132,"use_relative_action":True}})
    stats={}
    for kind,layout in (("state",c.STATE_GROUPS_2026),("action",c.ACTION_GROUPS),
                         ("relative_action",{k:c.ACTION_GROUPS[k] for k in ("torso","left_arm","right_arm")})):
        stats[kind]={k:dict(min=[-1]*(b-a),max=[1]*(b-a)) for k,(a,b) in layout.items()}
    write(root/"statistics.json",{"new_embodiment":stats})
    (root/"model.safetensors").write_bytes(b"fixture")


def test_model_and_protocol_gate(tmp_path,monkeypatch):
    ck=tmp_path/"ckpt";checkpoint(ck)
    info=c.inspect_checkpoint(ck,role="official_2026_radio")
    assert info["raw_state_dim"]==61 and info["selected_state_dim"]==25
    assert info["valid_action_steps"]==16 and info["physical_action_dim"]==23
    python=tmp_path/"python";python.touch()
    monkeypatch.setattr(c,"inspect_upstream",lambda p:{"root":str(p),"revision":c.FORK_REVISION})
    spec=c.official_serve_spec(python=python,upstream=tmp_path,checkpoint=ck,host="127.0.0.1",port=8000,gpu_index=0,verified_radio_sha256=c.checkpoint_digest(ck))
    assert spec["protocol"]==c.B1K2026 and spec["max_envs_per_service"]==1 and spec["rtc"] is False
    assert "temporal_ensemble" in spec["argv"]
    output=tmp_path/"eval";output.mkdir()
    endpoint=dict(spec,state="ready")
    plan=c.official_eval_spec(python=python,behavior_root=tmp_path,output=output,endpoint=endpoint,instance=0,rollouts=1)
    assert plan["num_envs"]==1 and "omnigibson.eval.eval" in plan["argv"]
    with pytest.raises(ValueError):
        c.official_eval_spec(python=python,behavior_root=tmp_path,output=output,
            endpoint=dict(endpoint,protocol=c.B1K2025),instance=0,rollouts=1)
    write(ck/"config.json",{"model_type":"Gr00tN1d6"})
    with pytest.raises(ValueError,match="N1.7"):c.inspect_checkpoint(ck,role="generic")


def test_2025_modality_keeps_original_joint_groups(tmp_path):
    offset=0;state={}
    for key,width in STATE_WIDTHS.items():
        state[key]={"start":offset,"end":offset+width};offset+=width
    mod=dict(state=state,action={k:{"start":a,"end":b} for k,(a,b) in c.ACTION_GROUPS.items()},
        video={k:{"original_key":k} for k in CAMERAS},annotation={"human.coarse_action":{"original_key":"annotation.human.coarse_action"}})
    write(tmp_path/"meta/info.json",{"fps":30,"features":{"observation.state":{"shape":[82]},"action":{"shape":[23]}}})
    write(tmp_path/"meta/modality.json",mod)
    info=audit_2025(tmp_path)
    assert info["state_dim"]==82 and not info["compatible_with_official_2026_radio"]
    for horizon in (16,32):
        source=modality_source(tmp_path,horizon=horizon)
        compile(source,"r1pro_2025.py","exec")
        assert f"range({horizon})" in source and "state_key='trunk_qpos'" in source
    with pytest.raises(ValueError):modality_source(tmp_path,horizon=50)
    mod["state"]["arm_left_qpos"]["start"]+=1
    write(tmp_path/"meta/modality.json",mod)
    with pytest.raises(ValueError):audit_2025(tmp_path)


def test_native_batch_is_pre_accumulation():
    assert batch_accounting(4,2,8)=={"global_batch_size":8,"effective_batch":64}
    with pytest.raises(ValueError):batch_accounting(4,2,0)


def test_reject_wrong_processor_and_checkpoint_identity(tmp_path,monkeypatch):
    ck=tmp_path/"ckpt";checkpoint(ck)
    python=tmp_path/"python";python.touch()
    monkeypatch.setattr(c,"inspect_upstream",lambda p:{"root":str(p),"revision":c.FORK_REVISION})
    digest=c.checkpoint_digest(ck)
    (ck/"model.safetensors").write_bytes(b"changed")
    assert digest!=c.checkpoint_digest(ck)
    with pytest.raises(ValueError,match="identity mismatch"):
        c.official_serve_spec(python=python,upstream=tmp_path,checkpoint=ck,host="localhost",port=8000,gpu_index=0,verified_radio_sha256=digest)
    pc=c.read(ck/"processor_config.json")
    pc["processor_kwargs"]["max_action_dim"]=128
    write(ck/"processor_config.json",pc)
    with pytest.raises(ValueError,match="dimension mismatch"):
        c.inspect_checkpoint(ck,role="official_2026_radio")
    checkpoint(ck)
    stats=c.read(ck/"statistics.json")
    stats["new_embodiment"]["relative_action"]["left_arm"]["min"]=[]
    write(ck/"statistics.json",stats)
    with pytest.raises(ValueError,match="normalization range"):
        c.inspect_checkpoint(ck,role="official_2026_radio")


def test_native_training_rejects_validation_file_leak(tmp_path,monkeypatch):
    from v1_2_groot_baseline import training_contract as t
    ck=tmp_path/"base";checkpoint(ck)
    dataset=tmp_path/"train";offset=0;state={}
    for k,w in STATE_WIDTHS.items():
        state[k]={"start":offset,"end":offset+w};offset+=w
    write(dataset/"meta/info.json",{"fps":30,"features":{"observation.state":{"shape":[82]},"action":{"shape":[23]}}})
    write(dataset/"meta/modality.json",dict(state=state,action={k:{"start":a,"end":b} for k,(a,b) in c.ACTION_GROUPS.items()},
         video={k:{"original_key":k} for k in CAMERAS},annotation={"human.coarse_action":{"original_key":"annotation.human.coarse_action"}}))
    (dataset/"meta/episodes.jsonl").write_text('{"episode_index":1}\n')
    data=dataset/"data/task-0000";data.mkdir(parents=True)
    (data/"episode_00000001.parquet").touch()
    split=tmp_path/"split.json";write(split,{"train":[1],"validation":[2]})
    python=tmp_path/"python";python.touch()
    run="v1_2__test99__exp99__run999"
    job=tmp_path/run;job.mkdir()
    modality=job/"modality.py";modality.write_text(modality_source(dataset,horizon=16))
    monkeypatch.setattr(t,"inspect_upstream",lambda p:{"root":str(p),"revision":c.FORK_REVISION})
    kw=dict(python=python,upstream=tmp_path,base=ck,dataset=dataset,modality=modality,
            job_root=job,run_id=run,gpus=[0],microbatch=2,accumulation=8,steps=4,
            save_steps=2,lr=1e-4,wd=1e-5,warmup_ratio=.05,horizon=16,split_file=split,
            verified_generic_sha256=c.checkpoint_digest(ck))
    plan=native_train_candidate(**kw)
    assert plan["effective_batch"]==16 and plan["launch_ready"] is False
    (data/"episode_00000002.parquet").touch()
    with pytest.raises(ValueError,match="held-out"):
        native_train_candidate(**kw)
