"""Source-level regression against the audited V1 action implementation."""
from pathlib import Path
import ast
ROOT=Path(__file__).resolve().parents[3]
def test_all_v21_python_parses():
    for p in (ROOT/'src/v2_1_groot_memoryvla').rglob('*.py'):
        ast.parse(p.read_text(),filename=str(p))
def test_velocity_insertion_sites():
    text=(ROOT/'src/v2_1_groot_memoryvla/model/velocity.py').read_text()
    assert text.index('u0 = pred_actions')<text.index('pred_actions, delta = refiner')<text.index('action_loss = F.mse_loss')
    assert text.index('u0 = pred_velocity')<text.index('pred_velocity, delta = refiner')<text.index('actions = actions + dt * pred_velocity')
    assert 'refiner(noisy_trajectory, u0, t[:,0,0], cache, robot, 1.)' in text
    assert 'refiner(actions, u0, t_cont, cache, robot, scale)' in text


def test_copied_velocity_uses_original_fm_and_euler():
    """Execute both actual function bodies with a small fake frozen DiT, no HF dependencies."""
    import torch
    import torch.nn.functional as F
    from types import SimpleNamespace
    class BatchFeature(dict):
        def __init__(self,data=None,**kwargs):super().__init__(data or {},**kwargs)
        def __getattr__(self,k):return self[k]
    env={"torch":torch,"F":F,"BatchFeature":BatchFeature}
    def extract(path,cls,names):
        tree=ast.parse(path.read_text())
        items=tree.body if cls is None else next(n.body for n in tree.body if isinstance(n,ast.ClassDef) and n.name==cls)
        funcs=[n for n in items if isinstance(n,ast.FunctionDef) and n.name in names]
        # Only evaluated annotations refer to imported names; the bodies are unchanged.
        for fn in funcs:
            fn.returns=None
            for a in fn.args.args+fn.args.kwonlyargs:a.annotation=None
        local=dict(env)
        exec(compile(ast.fix_missing_locations(ast.Module(body=funcs,type_ignores=[])),str(path),'exec'),local)
        return local
    base=extract(ROOT/'src/v1_groot_memoryvla/model/gr00t_n1d6/gr00t_n1d6.py','Gr00tN1d6ActionHead',
                 {'forward','get_action_with_features'})
    ours=extract(ROOT/'src/v2_1_groot_memoryvla/model/velocity.py',None,{'training_velocity','inference_velocity'})
    class Fake:
        config=SimpleNamespace(add_pos_embed=False,use_alternate_vl_dit=False,action_horizon=50)
        state_dropout_prob=0;state_additive_noise_scale=0;training=False
        num_timestep_buckets=1000;num_inference_timesteps=4;action_horizon=50;action_dim=128
        openvla_memory=False;memory_fusion_mode='legacy'
        def set_frozen_modules_to_eval_mode(self):pass
        def process_backbone_output(self,b):return b
        def state_encoder(self,s,e):return s
        def sample_time(self,b,device,dtype):return torch.rand(b,device=device,dtype=dtype)
        def action_encoder(self,x,t,e):return x+t[:,None,None]*.0001
        def _fusion_kwargs(self,*a,**kw):return {}
        def model(self,hidden_states,timestep,return_all_hidden_states=False,**kw):
            h=hidden_states*.25+timestep[:,None,None]*.0001
            return (h,None) if return_all_hidden_states else h
        def action_decoder(self,x,e):return x
    fake=Fake();features=BatchFeature(backbone_features=torch.randn(2,3,2048),backbone_attention_mask=torch.ones(2,3,dtype=torch.bool))
    action=BatchFeature(state=torch.randn(2,1,128),action=torch.randn(2,50,128),
                        action_mask=torch.zeros(2,50,128),embodiment_id=torch.full((2,),24))
    action.action_mask[:,:32,:23]=1
    cache=SimpleNamespace(logits=torch.zeros(2,8))
    calls=[]
    def residual(x,u,t,c,r,scale):
        calls.append((x.detach().clone(),float(scale)))
        d=torch.zeros_like(u);d[:,:,:23]=.2
        return u+scale*d,d
    def zero(x,u,t,c,r,scale):return u,torch.zeros_like(u)
    torch.manual_seed(42);orig=base['forward'](fake,features,action)
    torch.manual_seed(42);new=ours['training_velocity'](fake,features,action,zero,cache,None)
    assert torch.equal(orig['action_loss'],new['action_loss'])
    torch.manual_seed(42);new=ours['training_velocity'](fake,features,action,residual,cache,None)
    assert not torch.equal(orig['action_loss'],new['action_loss'])
    args=(fake,features.backbone_features,action.state,action.embodiment_id,features)
    torch.manual_seed(42);orig=base['get_action_with_features'](*args)
    calls.clear()
    torch.manual_seed(42);new=ours['inference_velocity'](*args,residual,cache,None,0.)
    assert torch.equal(orig['action_pred'],new['action_pred']) and len(calls)==4
    assert all(scale==0. for _,scale in calls)
    calls.clear()
    torch.manual_seed(42);new=ours['inference_velocity'](*args,residual,cache,None,1.)
    assert not torch.equal(orig['action_pred'][:,:,:23],new['action_pred'][:,:,:23])
    assert torch.equal(orig['action_pred'][:,:,23:],new['action_pred'][:,:,23:])
    assert len(calls)==4 and all(scale==1. for _,scale in calls)

def test_two_slot_schedule_is_chronological_and_complete():
    import numpy as np
    from types import SimpleNamespace
    p=ROOT/'src/v2_1_groot_memoryvla/data/windows.py'
    tree=ast.parse(p.read_text())
    fn=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='schedule')
    env={'np':np};exec(compile(ast.Module(body=[fn],type_ignores=[]),str(p),'exec'),env)
    ds=SimpleNamespace(by_episode={1:[0,1,2,3],2:[4],3:[5,6,7],4:[8,9]})
    owner={i:e for e,seq in ds.by_episode.items() for i in seq}
    batches=list(env['schedule'](ds,42,0))
    seen=[i for b in batches for i in b]
    assert sorted(seen)==list(range(10))
    for b in batches:assert 1<=len(b)<=2 and len(set(owner[i] for i in b))==len(b)
    for e,seq in ds.by_episode.items():assert [i for i in seen if owner[i]==e]==seq
