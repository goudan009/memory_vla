"""Dependency and mutation journal checks; no remote/GPU imports."""
import importlib.util
from pathlib import Path

SPEC=importlib.util.spec_from_file_location('v21_overnight',Path(__file__).parents[1]/'scripts/v21_overnight.py')
M=importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(M)


class Fake:
    def __init__(self):
        self.calls=[]
        self.jobs={}
        self.details={}
    def api(self,method,path,body=None):
        self.calls.append((method,path,body))
        if path.startswith('/api/training/jobs'):return {'jobs':list(self.jobs.values())}
        if path.startswith('/api/training/detail'):
            from urllib.parse import parse_qs,urlsplit
            return self.details[parse_qs(urlsplit(path).query)['run'][0]]
        return {'ok':True}
    def resources(self,run):
        return dict(free_bytes=100*2**30,stop=None,live=[],resume_events=[],gpu='four A100')


def detail(stage='stage1',status='training',step=1,checkpoint=True):
    run=M.RUNS[stage]
    return dict(run=run,arm='memory_v1' if stage=='memory' else 'baseline',
        state=dict(status=status,step=step,last=dict(fm=.1,grad_norm=.2)),
        config=dict(steps=6000 if stage=='stage1' else 3000,source='/source/'+run),
        checkpoints=[dict(step=step,path='/checkpoint/'+run)] if checkpoint else [],metrics=[])


def chain(w,state):return M.Chain(w,state,lambda:None,lambda *a:None)


def test_running_stage1_cannot_dispatch_dependencies():
    w=Fake();w.details[M.RUNS['stage1']]=detail()
    state={};chain(w,state).tick()
    assert state['phase']=='wait_stage1'
    assert not any(c[0]=='POST' for c in w.calls)


def test_completed_without_final_checkpoint_blocks():
    w=Fake();w.details[M.RUNS['stage1']]=detail(status='completed',step=6000,checkpoint=False)
    state={};chain(w,state).tick()
    assert state['status']=='blocked'
    assert not any(c[0]=='POST' for c in w.calls)


def test_complete_stage1_dispatches_only_memory_preflight_preview_first():
    w=Fake();w.details[M.RUNS['stage1']]=detail(status='completed',step=6000)
    state={};chain(w,state).tick()
    posts=[c for c in w.calls if c[0]=='POST']
    assert [p[1] for p in posts]==['/api/training/preview','/api/training/start']
    request=posts[1][2]
    assert request['stage']=='memory' and request['preflight_only'] is True
    assert request['steps']==3000 and request['gpus']==[0,1,2,3]
    assert request['checkpoint']=='/checkpoint/'+M.RUNS['stage1']


def test_ambiguous_start_never_resubmits():
    w=Fake();w.details[M.RUNS['stage1']]=detail(status='completed',step=6000)
    state={'actions':{M.RUNS['memory']+'__preflight001-start':{'status':'sending'}}}
    chain(w,state).tick()
    assert state['status']=='blocked'
    assert not any(c[0]=='POST' for c in w.calls)


def test_existing_running_preflight_adopted():
    w=Fake();w.details[M.RUNS['stage1']]=detail(status='completed',step=6000)
    pre=M.RUNS['memory']+'__preflight001'
    w.jobs[pre]={'run':pre};w.details[pre]=detail('memory',status='initializing',step=0)
    state={};chain(w,state).tick()
    assert not any(c[0]=='POST' for c in w.calls)


def test_user_pause_holds_future_stages():
    w=Fake();w.details[M.RUNS['stage1']]=detail(status='stopped',step=400)
    state={};chain(w,state).tick()
    assert state['status']=='blocked'
    assert not any(c[0]=='POST' for c in w.calls)


def test_nonfinite_training_requests_stop_and_blocks():
    w=Fake();d=detail();d['state']['last']['fm']=float('nan')
    w.details[M.RUNS['stage1']]=d
    state={};chain(w,state).tick()
    assert state['status']=='blocked'
    assert [c[1] for c in w.calls if c[0]=='POST']==['/api/training/stop']


def test_failed_preflight_cannot_be_promoted_or_deleted():
    w=Fake();state={'checkpoints':{'stage1':{'checkpoint':'/s1'}}}
    pre=M.RUNS['memory']+'__preflight001'
    w.details[pre]=detail('memory',status='failed',step=3)
    w.archive=lambda run:{'files':{'preflight.json':{'passed':False}}}
    chain(w,state).training('memory',{pre:{'run':pre}})
    assert state['status']=='blocked'
    assert not any(c[0]=='POST' for c in w.calls)


def test_archive_required_before_preflight_purge():
    w=Fake();state={'checkpoints':{'stage1':{'checkpoint':'/s1'}}}
    pre=M.RUNS['memory']+'__preflight001'
    d=detail('memory',status='completed',step=3);d['preflight']={'passed':True};w.details[pre]=d
    w.archive=lambda run:{'path':'/independent/preflight.json','files':{'preflight.json':{'passed':True}}}
    c=chain(w,state);c.training('memory',{pre:{'run':pre}})
    assert not any(c[0]=='POST' for c in w.calls)
    c.training('memory',{pre:{'run':pre}})
    post=[c for c in w.calls if c[0]=='POST'][0]
    assert post[1]=='/api/training/delete'
    assert post[2]['evidence_path']=='/independent/preflight.json'
    assert post[2]['purge_preflight'] is True
