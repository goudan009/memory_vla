"""Fast tests use the real V1 memory and GR00T head, with smaller widths."""
from pathlib import Path
from types import SimpleNamespace
import copy
import json
import numpy as np
import pandas as pd
import pytest
import torch
from torch import nn
from transformers.feature_extraction_utils import BatchFeature
from v1_groot_memoryvla.configs.model.gr00t_n1d6 import Gr00tN1d6Config
from v1_groot_memoryvla.model.gr00t_n1d6.gr00t_n1d6 import Gr00tN1d6ActionHead
from v1_groot_memoryvla.model.memory.adapter import MemoryAdapter
from v2_groot_memoryvla.config import V2Config
from v2_groot_memoryvla.data import BodyMapper
from v2_groot_memoryvla.body import BodyEncoder
from v2_groot_memoryvla.kinematics import RobotGraph
from v2_groot_memoryvla.model import V2Model

ROOT = Path(__file__).resolve().parents[3]
URDF = ROOT/"assets/simulation/omnigibson-robot-assets/models/r1pro/urdf/r1pro.urdf"
DATA = ROOT/"assets/datasets/b1k2025_groot_task0_v1_200"


@pytest.fixture(scope="module")
def sample():
    mapper = BodyMapper(URDF)
    df = pd.read_parquet(DATA/"data/task-0000/episode_00000010.parquet")
    modality = json.loads((DATA/"meta/modality.json").read_text())
    body = mapper.episode(df,modality,[0,8,16,24])
    return mapper,body,df,modality


class SmallBase(nn.Module):
    def __init__(self, memory=True, conditioning="residual"):
        super().__init__()
        self.config = Gr00tN1d6Config(backbone_embedding_dim=32,hidden_size=32,
            input_embedding_dim=32,max_state_dim=128,max_action_dim=128,
            max_num_embodiments=32,action_horizon=8,attn_dropout=0.,
            diffusion_model_cfg=dict(num_layers=2,num_attention_heads=4,attention_head_dim=8,
                output_dim=32,dropout=0.,final_dropout=False,interleave_self_attention=True,norm_type="ada_norm"))
        self.action_head = Gr00tN1d6ActionHead(self.config)
        self.memory = MemoryAdapter(32,16,4,1,4,conditioning=conditioning) if memory else None
        self.backbone = nn.Identity()


def fixture_inputs():
    torch.manual_seed(41)
    features = BatchFeature(data={"backbone_features":torch.randn(4,245,32),
        "visual_features":torch.randn(4,3,81,32),
        "backbone_attention_mask":torch.ones(4,245,dtype=torch.bool),
        "image_mask":torch.arange(245).expand(4,-1)<243})
    ids = torch.zeros(4,245,dtype=torch.long); ids[:,-1]=151645
    actions = BatchFeature(data={"state":torch.randn(4,1,128),"action":torch.randn(4,8,128),
        "action_mask":(torch.arange(128)<23).expand(4,8,-1),"embodiment_id":torch.full((4,),24)})
    ctx = {"mode":"group","sequence_length":4,"episode_keys":["episode"]*4,"timesteps":[0,1,2,3]}
    return features,ids,actions,ctx


def make(variant,mapper,conditioning="residual"):
    return V2Model(SmallBase(variant!="baseline",conditioning), V2Config(variant=variant,
        body_dim=32,body_heads=4,memory_dim=32,memory_capacity=4),mapper if variant.startswith("v2") else None)


@pytest.mark.parametrize("conditioning",["residual","direct"])
@pytest.mark.parametrize("variant",["baseline","v1","v2a","v2b","v2c"])
def test_forward_gradient(variant,conditioning,sample):
    mapper,body,_,_ = sample
    model = make(variant,mapper,conditioning)
    features,ids,actions,context = fixture_inputs()
    out = model.forward_cached(features,ids,context,actions,body)
    assert torch.isfinite(out["loss"])
    out["loss"].backward()
    if variant.startswith("v2"):
        for name,p in model.extension.named_parameters():
            # V2b's body-query branch is deliberately masked for capacity match.
            if variant=="v2b" and name=="query.body.weight": continue
            assert p.grad is not None and torch.isfinite(p.grad).all(), name
            if name.endswith("weight") and not name.endswith("history.k.bias"):
                assert p.grad.norm()>0, name
    model.eval(); model.reset()
    with torch.no_grad(): pred=model.forward_cached(features,ids,context,actions,body,predict=True)
    assert pred["action_pred"].shape==(4,8,128)


@pytest.mark.parametrize("conditioning",["residual","direct"])
def test_short_groups_and_empty_history(conditioning,sample):
    mapper,body,_,_=sample
    model=make("v2c",mapper,conditioning).eval()
    features,ids,_,context=fixture_inputs()
    context.update(group_lengths=[1,3],episode_keys=["a","b","b","b"],timesteps=[0,0,1,2])
    model.fuse(features,ids,context,body)
    assert [d["history_times"] for d in model.extension.diagnostics]==[[],[],[0],[0,1]]
    assert not model.extension.history.bank
    context.update(mode="stream",episode_keys=["a"]*4,timesteps=[0,1,2,3])
    model.fuse(features,ids,context,body)
    bank=copy.deepcopy(model.extension.history.bank)
    v1_bank=copy.deepcopy(model.base.memory.cognitive.bank)
    context["mode"]="empty"
    # Empty accepts even the same timestamps: it must neither read nor write.
    empty=model.fuse(features,ids,context,body)
    assert all(d["history_times"]==[] for d in model.extension.diagnostics)
    assert all(d["attention"].numel()==0 for d in model.extension.diagnostics)
    for old,new in zip(bank["a"],model.extension.history.bank["a"]):
        assert old[0]==new[0]
        torch.testing.assert_close(old[1],new[1],rtol=0,atol=0)
    for old,new in zip(v1_bank["a"],model.base.memory.cognitive.bank["a"]):
        torch.testing.assert_close(old[1],new[1],rtol=0,atol=0)
    model.reset()
    torch.testing.assert_close(empty["backbone_features"],model.fuse(features,ids,context,body)["backbone_features"],rtol=0,atol=0)


def test_time_alignment_no_future(sample):
    mapper,body,df,modality = sample
    assert body["action_mask"][0].sum()==0
    actions=np.stack(df.action)
    for node,_,_,_,_,action in mapper.entries:
        assert body["previous_action"][1,node].item()==np.float32(actions[7,action])
    altered=df.copy(deep=True)
    # Modify all future rows, including current label action[8], but not past.
    for i in range(8,len(df)):
        altered.at[i,"action"]=np.ones(23)*100
    for i in range(9,len(df)):
        altered.at[i,"observation.state"]=np.ones(82)*100
    original=mapper.episode(df,modality,[8]); changed=mapper.episode(altered,modality,[8])
    for key in original: torch.testing.assert_close(original[key],changed[key],rtol=0,atol=0)
    with pytest.raises(ValueError,match="precede"):
        mapper.observation({},np.zeros(23),1.,1.)


def test_raw_joint_mapping_and_fk(sample):
    mapper,body,df,modality = sample
    raw=pd.read_parquet(ROOT/"assets/datasets/b1k2025/data/task-0000/episode_00000010.parquet")
    with_raw=mapper.episode(df,modality,[0,8,16,24],raw)
    torch.testing.assert_close(body["q"],with_raw["q"])
    assert with_raw["velocity_mask"].sum()==4*22
    transforms,valid=mapper.graph.fk(body["q"],body["q_mask"])
    state=np.stack(df["observation.state"])[[0,8,16,24]]
    from v2_groot_memoryvla.kinematics import quat_matrix
    for side in ("left","right"):
        idx=mapper.graph.link_index[f"{side}_eef_link"]
        assert valid[:,idx].all()
        def field(name):
            m=modality["state"][name];return torch.tensor(state[:,m["start"]:m["end"]],dtype=torch.float32)
        torch.testing.assert_close(transforms[:,idx,:3,3],field(f"eef_{side}_pos"),rtol=0,atol=0.003)
        torch.testing.assert_close(transforms[:,idx,:3,:3],quat_matrix(field(f"eef_{side}_quat")),rtol=0,atol=.01)


def test_variable_nodes_and_padding(sample,tmp_path):
    mapper,body,_,_=sample
    tiny=tmp_path/"tiny.urdf"
    tiny.write_text('<robot name="test"><link name="base"/><link name="slide"/><joint name="s" type="prismatic"><parent link="base"/><child link="slide"/><axis xyz="1 0 0"/><limit lower="0" upper="1"/></joint></robot>')
    graph=RobotGraph(tiny)
    data={k:v[:2].clone() for k,v in body.items()}
    data["q"][1,1]=.3; data["q_mask"][1,:2]=1
    encoder=BodyEncoder(32,2,4)
    out=encoder([mapper.graph,graph],data)
    assert out["padding_mask"][1].sum()==36
    assert torch.isfinite(out["tokens"]).all()
    assert out["poses"][1][0,1,0,3]==.3
    changed={k:v.clone() for k,v in data.items()}
    changed["q"][1,2:]=1000
    torch.testing.assert_close(out["pooled"],encoder([mapper.graph,graph],changed)["pooled"])


@pytest.mark.parametrize("conditioning",["residual","direct"])
def test_causal_attention_influence_and_checkpoint(conditioning,sample,tmp_path):
    mapper,body,_,_=sample
    model=make("v2c",mapper,conditioning).eval()
    features,ids,actions,context=fixture_inputs()
    def run(body,history_shift=0):
        model.reset()
        f=BatchFeature(data={k:v.clone() for k,v in features.items()})
        # Alter past embodied content only, retaining the same current frame.
        b={k:v.clone() for k,v in body.items()}; b["q"][:3]+=history_shift
        with torch.no_grad():
            fused=model.fuse(f,ids,context,b)
            diag=model.extension.diagnostics
            torch.manual_seed(17); pred=model.base.action_head.get_action(fused,actions)["action_pred"]
        return pred,diag
    pred,diag=run(body)
    for i,d in enumerate(diag): assert all(t<i for t in d["history_times"])
    changed={k:v.clone() for k,v in body.items()}; changed["q"][-1]+=0.4
    other,other_diag=run(changed)
    assert not torch.equal(diag[-1]["query"],other_diag[-1]["query"])
    assert not torch.equal(diag[-1]["attention"],other_diag[-1]["attention"])
    assert not torch.equal(pred[-1],other[-1])
    # Future-body perturbation cannot change any previous output.
    torch.testing.assert_close(pred[:3],other[:3],rtol=0,atol=0)
    past,past_diag=run(body,.4)
    assert not torch.equal(diag[-1]["attention"],past_diag[-1]["attention"])
    assert not torch.equal(pred[-1],past[-1])
    torch.save(model.state_dict(),tmp_path/"state.pt")
    restored=make("v2c",mapper,conditioning).eval()
    restored.load_state_dict(torch.load(tmp_path/"state.pt",weights_only=True),strict=True)
    model.reset(); restored.reset()
    torch.manual_seed(31); a=model.forward_cached(features,ids,context,actions,body,predict=True)["action_pred"]
    torch.manual_seed(31); b=restored.forward_cached(features,ids,context,actions,body,predict=True)["action_pred"]
    torch.testing.assert_close(a,b,rtol=0,atol=0)


def test_capacity_match_and_exact_v1_reuse(sample):
    mapper,_,_,_=sample
    b,c=make("v2b",mapper),make("v2c",mapper)
    assert sum(p.numel() for p in b.parameters())==sum(p.numel() for p in c.parameters())
    c.load_state_dict(b.state_dict(),strict=True)
    # Reusing V1 is composition, so neither its names nor parameter values change.
    original=SmallBase(); clone=copy.deepcopy(original.state_dict())
    wrapped=V2Model(original,V2Config(variant="v2c",body_dim=32,memory_dim=32),mapper)
    for k,v in clone.items(): torch.testing.assert_close(wrapped.base.state_dict()[k],v,rtol=0,atol=0)


def test_v2a_excludes_previous_commands_and_v2b_query_excludes_body(sample):
    mapper,body,_,_=sample
    features,ids,_,context=fixture_inputs()
    a=make("v2a",mapper).eval()
    original=a.fuse(features,ids,context,body)["backbone_features"]
    changed={k:v.clone() for k,v in body.items()}
    changed["previous_action"]+=100;changed["base"][:,6:]+=100
    a.reset()
    torch.testing.assert_close(original,a.fuse(features,ids,context,changed)["backbone_features"],rtol=0,atol=0)
    b=make("v2b",mapper).eval()
    b.fuse(features,ids,context,body); query=b.extension.diagnostics[-1]["query"]
    b.reset();changed={k:v.clone() for k,v in body.items()};changed["q"][-1]+=.4
    b.fuse(features,ids,context,changed)
    torch.testing.assert_close(query,b.extension.diagnostics[-1]["query"],rtol=0,atol=0)


@pytest.mark.parametrize("conditioning",["residual","direct"])
def test_v1_action_parity_and_query_only_effect(conditioning,sample):
    mapper,body,_,_=sample
    features,ids,actions,ctx=fixture_inputs()
    model=make("v1",mapper,conditioning).eval()
    fused=model.base.memory(features,ids,ctx)
    torch.manual_seed(32); original=model.base.action_head.get_action(fused,actions)["action_pred"]
    model.reset();torch.manual_seed(32)
    wrapped=model.forward_cached(features,ids,ctx,actions,body,predict=True)["action_pred"]
    torch.testing.assert_close(original,wrapped,rtol=0,atol=0)
    model=make("v2c",mapper,conditioning).eval()
    torch.manual_seed(32)
    conditioned=model.forward_cached(features,ids,ctx,actions,body,predict=True)["action_pred"]
    q=model.extension.diagnostics[-1]["query"]
    model.reset();model.config.variant="v2b";torch.manual_seed(32)
    ordinary=model.forward_cached(features,ids,ctx,actions,body,predict=True)["action_pred"]
    assert not torch.equal(q,model.extension.diagnostics[-1]["query"])
    assert not torch.equal(conditioned[-1],ordinary[-1])


def test_capacity_and_reset_isolation():
    from v2_groot_memoryvla.memory import EmbodiedMemory
    memory=EmbodiedMemory(8,2)
    encoder=nn.Linear(10,8)
    for i in range(8):
        _,_,times=memory.read_write(torch.randn(8),torch.randn(10),"a",i,encoder)
        assert all(t<i for t in times)
        assert len(memory.bank["a"])<=2
        assert all(not v.requires_grad for _,v in memory.bank["a"])
    with pytest.raises(ValueError,match="Non-increasing"):
        memory.read_write(torch.randn(8),torch.randn(10),"a",7,encoder)
    memory.read_write(torch.randn(8),torch.randn(10),"b",0,encoder)
    memory.reset(["a"])
    assert "b" in memory.bank and "a" not in memory.bank


def test_attention_reduction_keeps_float32_under_autocast():
    from v2_groot_memoryvla.memory import EmbodiedMemory
    memory=EmbodiedMemory(8,4)
    encoder=nn.Linear(10,8)
    memory.read_write(torch.randn(8),torch.randn(10),"a",0,encoder)
    memory.read_write(torch.randn(8),torch.randn(10),"a",1,encoder)
    with torch.autocast("cpu",dtype=torch.bfloat16):
        _,weights,_=memory.read_write(torch.randn(8).bfloat16(),torch.randn(10),"a",2,encoder)
    assert weights.dtype==torch.float32


def test_collator_body_alignment(sample):
    from v2_groot_memoryvla.dataset import V2HistoryCollator
    _,body,_,_=sample
    groups=[dict(steps=[None]*4,episode_key=key,group_id=i,frame_indices=[0,8,16,24],
                 timesteps=[0,1,2,3],body={k:v.clone()+i for k,v in body.items()})
            for i,key in enumerate(["a","b"])]
    collator=V2HistoryCollator(lambda steps:{"inputs":{}},group_size=4)
    batch=collator(groups)["inputs"]
    assert batch["memory_context"]["episode_keys"]==["a"]*4+["b"]*4
    torch.testing.assert_close(batch["body"]["q"][:4],body["q"])
    torch.testing.assert_close(batch["body"]["q"][4:],body["q"]+1)


def test_checkpoint_memory_protocol_resolution():
    from v2_groot_memoryvla.model import resolve_memory_config
    fresh=resolve_memory_config({},V2Config())
    assert fresh.memory_conditioning=="direct" and fresh.memory_length==64
    legacy=resolve_memory_config({"memory_enabled":True,"memory_length":16},V2Config())
    assert legacy.memory_conditioning=="residual" and legacy.memory_length==16
    direct=resolve_memory_config({"memory_enabled":True,"memory_length":256,"memory_conditioning":"direct"},V2Config())
    assert direct.memory_conditioning=="direct" and direct.memory_length==256
    with pytest.raises(ValueError,match="conditioning"):
        resolve_memory_config({"memory_enabled":True},V2Config(memory_conditioning="direct"))
    with pytest.raises(ValueError,match="capacity"):
        resolve_memory_config({"memory_enabled":True,"memory_length":16},V2Config(memory_length=64))


def test_tail_loss_uses_valid_action_elements(sample):
    from v2_groot_memoryvla.pilot import loss_sequence
    _,body,_,_=sample
    features,ids,actions,ctx=fixture_inputs()
    actions["state"][:,0,0]=torch.tensor([1.,2.,3.,4.])
    actions["action_mask"]=torch.zeros_like(actions["action_mask"])
    for i,count in enumerate([8,8,4,1]): actions["action_mask"][i,:count,:23]=True
    class LossHead(nn.Module):
        def __init__(self):
            super().__init__(); self.scale=nn.Parameter(torch.tensor(1.))
        def forward(self,f,a): return {"loss":self.scale*a["state"][0,0,0]}
    head=LossHead()
    model=SimpleNamespace(reset=lambda:None,fuse=lambda f,*args:f,base=SimpleNamespace(action_head=head))
    seq=dict(features=features,input_ids=ids,actions=actions,body=body,episode_key="a",timesteps=[0,1,2,3])
    value=loss_sequence(model,seq,torch.device("cpu"),backward=True)
    expected=(1*8+2*8+3*4+4*1)/21
    assert value==pytest.approx(expected)
    assert head.scale.grad.item()==pytest.approx(expected)


@pytest.mark.parametrize("source",["original","legacy","direct"])
def test_hf_v1_initialization_and_v2_full_reload(source,sample,tmp_path,monkeypatch):
    from v1_groot_memoryvla.model.gr00t_n1d6 import gr00t_n1d6 as module
    from v1_groot_memoryvla.model.gr00t_n1d6 import processing_gr00t_n1d6 as processing
    from v1_groot_memoryvla.tests.test_migration_training import TinyBackbone
    monkeypatch.setattr(module,"get_backbone_cls",lambda config:TinyBackbone)
    monkeypatch.setattr(processing,"Gr00tN1d6DataCollator",lambda **kwargs:None)
    cfg=Gr00tN1d6Config(memory_enabled=source!="original",
        memory_conditioning="direct" if source=="direct" else "residual",memory_length=16,
        backbone_embedding_dim=32,hidden_size=32,input_embedding_dim=32,
        max_state_dim=128,max_action_dim=128,action_horizon=8,max_num_embodiments=32,
        load_bf16=False,use_flash_attention=False,attn_dropout=0.,
        diffusion_model_cfg=dict(num_layers=2,num_attention_heads=4,attention_head_dim=8,
            output_dim=32,dropout=0.,final_dropout=False,interleave_self_attention=True,norm_type="ada_norm"))
    base=module.Gr00tN1d6(cfg)
    base.save_pretrained(tmp_path/"source")
    if source=="legacy":
        p=tmp_path/"source/config.json"
        values=json.loads(p.read_text()); values.pop("memory_conditioning");p.write_text(json.dumps(values))
    model=V2Model.from_v1(tmp_path/"source",V2Config(variant="v2c",body_dim=32,memory_dim=32),URDF).eval()
    assert model.base.memory.conditioning==("residual" if source=="legacy" else "direct")
    assert model.base.memory.cognitive.capacity==(64 if source=="original" else 16)
    if source=="original":
        model.base.requires_grad_(False)
        with pytest.raises(ValueError,match="newly initialized"):
            model.save_extension(tmp_path/"invalid-extension")
    else:
        assert not model.loading_info["missing_keys"]
        for k,v in base.memory.state_dict().items():
            torch.testing.assert_close(model.base.memory.state_dict()[k],v,rtol=0,atol=0)
    _,body,_,_=sample
    features,ids,actions,ctx=fixture_inputs()
    torch.manual_seed(87)
    before=model.forward_cached(features,ids,ctx,actions,body,predict=True)["action_pred"]
    model.save(tmp_path/"full")
    restored=V2Model.load(tmp_path/"full").eval()
    torch.manual_seed(87)
    after=restored.forward_cached(features,ids,ctx,actions,body,predict=True)["action_pred"]
    torch.testing.assert_close(before,after,rtol=0,atol=0)
    assert restored.config.memory_conditioning==model.config.memory_conditioning
    assert restored.config.memory_length==model.config.memory_length


def test_training_augmentation_reencoded_and_eval_restored(monkeypatch):
    from v2_groot_memoryvla import pilot
    processor=nn.Identity().eval()
    model=SimpleNamespace(base=SimpleNamespace(backbone=nn.Identity()))
    reference=dict(episode_key="a",frame_indices=[0,8])
    seeds=[]
    def recache(base,proc,dataset,index,microbatch,augmentation_seed):
        assert proc.training
        seeds.append(augmentation_seed)
        return dict(reference)
    monkeypatch.setattr(pilot,"cache_sequence",recache)
    for step in range(2):
        pilot.augmented_training_sequence(model,processor,None,0,torch.device("cpu"),42+step,reference)
        assert not processor.training
    assert seeds==[42,43]
    with pytest.raises(ValueError,match="selected training window"):
        pilot.augmented_training_sequence(model,processor,None,0,torch.device("cpu"),44,
                                          dict(episode_key="wrong",frame_indices=[0,8]))
    assert not processor.training


def test_diagnostic_benchmark_uses_real_short_window(sample,monkeypatch):
    from contextlib import nullcontext
    from v2_groot_memoryvla.diagnose import benchmark
    mapper,body,_,_=sample
    features,ids,actions,ctx=fixture_inputs()
    seq=dict(features=features,input_ids=ids,actions=actions,body=body,episode_key="a",timesteps=[0,1])
    monkeypatch.setattr(torch.cuda,"synchronize",lambda:None)
    monkeypatch.setattr(torch,"autocast",lambda *a,**kw:nullcontext())
    result=benchmark(make("v2c",mapper).eval(),seq,torch.device("cpu"))
    assert all(row["samples"]==10 for row in result.values())


@pytest.mark.parametrize("variant",["baseline","v1"])
def test_no_body_variants_accept_shared_body_batch(variant,sample):
    mapper,body,_,_=sample
    model=make(variant,mapper,"direct").eval()
    features,ids,actions,ctx=fixture_inputs()
    class CachedBackbone(nn.Module):
        def forward(self,inputs): return features
    model.base.backbone=CachedBackbone()
    model.base.prepare_input=lambda inputs:({"input_ids":ids},actions)
    out=model(dict(body=body,memory_context=ctx),predict=True)
    assert out["action_pred"].shape==(4,8,128)
