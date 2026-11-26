import gc
import json
from pathlib import Path
import pytest
import torch
from v2_1_groot_memoryvla.config import Config,ALL_STATES,check_scale
from v2_1_groot_memoryvla.model.embodiment import R1Pro,quat_matrix
from v2_1_groot_memoryvla.model.memory import Banks,Memory,Refinement,Cache
from v2_1_groot_memoryvla.data.labels import state_terms,normalized_state_loss

torch.set_num_threads(2)
def bounds():
    return torch.full((82,),-5.),torch.full((82,),5.)
def test_config():
    assert len(Config().states)==8 and len(ALL_STATES)==21
    assert len(Config(states=ALL_STATES).states)==21
    for v in [-1,2,float('nan'),float('inf'),True,'0.5']:
        with pytest.raises(ValueError):check_scale(v)
def test_quality_and_normalization():
    x=torch.zeros(3,2,requires_grad=True)
    y=torch.tensor([[1.,0.],[1.,float('nan')],[float('nan'),0.]])
    q=torch.tensor([[1,1],[2,0],[0,2]])
    n,c=state_terms(x,y,q,torch.ones(2),.5)
    loss=normalized_state_loss(n,c)
    assert torch.allclose(loss,torch.log(torch.tensor(2.))*.75)
    loss.backward()
    assert x.grad[1,1]==0 and x.grad[2,0]==0
    assert torch.allclose(x.grad[1,0],x.grad[0,0]*.5)
def test_banks():
    lo,hi=bounds();b=Banks(2,.2,lo,hi)
    q=torch.zeros(82);q[3:6]=1.;q[10]=1.
    for t in range(5):
        p,c,prev=b.read('a',t)
        assert all(s.end<t for s in p+c)
        assert sum(s.count for s in p)==t and sum(s.count for s in c)==t
        v=torch.ones(3,4,requires_grad=True)*t
        b.write('a',t,v,q,torch.ones(6)*t,torch.ones(2)*t)
        assert len(b.p['a'])<=2 and len(b.c['a'])<=2
        assert not b.p['a'][0].value.requires_grad
    assert sum(s.count for s in b.p['a'])==5
    assert b.read('b',0)==([],[],None)
    with pytest.raises(ValueError):b.read('a',4)
    b.reset(['a']);assert not b.p and not b.c and not b.prev and not b.last
def test_fk_audited_samples():
    robot=R1Pro(*bounds(),'fk')
    samples=json.loads(Path(__file__).with_name('fk_samples.json').read_text())
    q=torch.zeros(len(samples),82)
    for i,s in enumerate(samples):
        raw=torch.tensor(s['q'])
        q[i,78:82]=raw[6:10];q[i,18:25]=raw[10:24:2];q[i,48:55]=raw[11:24:2]
        q[i,46:48]=raw[24:26];q[i,76:78]=raw[26:28]
    poses=robot.fk(q)
    for i,s in enumerate(samples):
        for name,ref in s['links'].items():
            assert torch.allclose(poses[name][i,:3,3],torch.tensor(ref['position_m']),atol=1e-5,rtol=0)
            assert torch.allclose(poses[name][i,:3,:3],quat_matrix(torch.tensor(ref['quaternion_xyzw'])),atol=1e-5,rtol=0)
    assert (~robot.local_mask).sum()==73  # 23 self loops + 2*25 undirected edges
@pytest.mark.parametrize("prior",["topology","fk","none"])
def test_refinement_modes(prior):
    cfg=Config(prior=prior);robot=R1Pro(*bounds(),prior)
    assert robot(torch.zeros(2,82)).shape==(2,23,256)
    head=Refinement(cfg)
    x=torch.randn(2,50,128);u=torch.randn_like(x)
    cache=Cache(torch.randn(2,23,256),torch.randn(2,2048),torch.randn(2,23,256),torch.tensor([True,False]),torch.randn(2,8),[])
    z,d=head(x,u,.25,cache,robot)
    assert torch.equal(z,u) and torch.count_nonzero(d)==0
    with torch.no_grad():head.output.weight.normal_(std=.01)
    z,d=head(x,u,.25,cache,robot)
    assert torch.count_nonzero(d[0,:,:23])>0
    assert torch.equal(z[1],u[1]) and torch.equal(z[:,:,23:],u[:,:,23:])
    for gain in [0,.25,.5,1]:
        out,delta=head(x,u,.25,cache,robot,gain)
        assert torch.allclose(out,u+gain*d,atol=1e-6)
    assert (head.local is not None)==(prior=='topology')
def test_causal_memory_gradient_and_empty():
    torch.manual_seed(42)
    cfg=Config(capacity=2)
    model=Memory(cfg,*bounds())
    head=Refinement(cfg);opt=torch.optim.AdamW(list(model.parameters())+list(head.parameters()),lr=1e-4)
    def batch(t):
        feat={'visual_features':torch.randn(2,3,81,2048),
              'backbone_features':torch.randn(2,4,2048),'backbone_attention_mask':torch.ones(2,4,dtype=torch.bool)}
        ids=torch.tensor([[1,2,151645,3]]*2)
        q=torch.randn(2,82)*.2
        context={'episode_keys':['a','a'],'timesteps':[t,t+1],'sequence_start':t==0}
        return feat,ids,q,context
    for t in [0,2]:
        opt.zero_grad(set_to_none=True)
        feat,ids,q,context=batch(t)
        c=model.prepare(feat,ids,q,context)
        assert c.memory.shape==(2,23,256) and c.logits.shape==(2,8)
        assert c.active.tolist()==([False,True] if t==0 else [True,True])
        assert all(s is None or torch.allclose(s[2].sum(-1),torch.ones(23)) for s in c.scores)
        x=torch.randn(2,50,128);u=torch.randn_like(x)
        corrected,_=head(x,u,.25,c,model.robot)
        loss=corrected[:,:,:23].square().mean()+c.logits.square().mean()*.01
        loss.backward()
        if t==2:
            for module in [model.visual,model.enhance,model.state_head,model.robot.body,model.query,model.key,model.value,model.cog_read,head.action,head.fusion,head.output]:
                assert any(p.grad is not None and torch.isfinite(p.grad).all() and p.grad.abs().sum()>0 for p in module.parameters())
        opt.step()
    assert len(model.banks.p['a'])==2 and sum(s.count for s in model.banks.p['a'])==4
    model.eval()
    feat,ids,q,context=batch(4);context['history_mode']='empty'
    with torch.no_grad():
        c=model.prepare(feat,ids,q,context)
        assert not c.active.any()
        out,d=head(x,u,.25,c,model.robot)
        assert torch.equal(out,u) and torch.count_nonzero(d)==0
    model.reset();assert not model.banks.last
    del model,head,opt;gc.collect()


def test_runtime_roundtrip():
    from types import SimpleNamespace
    from v2_1_groot_memoryvla.runtime import export_runtime,import_runtime
    lo,hi=bounds()
    b=Banks(2,.2,lo,hi)
    q=torch.zeros(82);q[3:6]=1.;q[10]=1.
    for t in range(5):
        b.write('a',t,torch.ones(3,4)*t,q,torch.ones(6)*t,torch.ones(2)*t)
    model=SimpleNamespace(memory=SimpleNamespace(banks=b,robot=SimpleNamespace(lo=lo)),reset=b.reset)
    state=export_runtime(model)
    b.reset()
    import_runtime(model,state)
    assert b.last['a']==4 and sum(s.count for s in b.p['a'])==5
    p,c,prev=b.read('a',5)
    assert all(s.end<5 for s in p+c)
    assert torch.equal(prev,torch.full((2,),4.))
    assert torch.equal(p[0].value,state['p']['a'][0][3])
    assert not p[0].value.requires_grad
    assert b.read('other',0)==([],[],None)
