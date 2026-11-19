from copy import deepcopy

import torch

from v1_groot_memoryvla.tests.test_history_controls import populated
from v1_groot_memoryvla.tests.test_memory_v1 import features


def stochastic_head(backbone):
    x=backbone.backbone_features
    action=x[:,:,:8].mean(dim=1,keepdim=True).expand(-1,4,-1)
    return {'action_pred':action+torch.randn_like(action)}


def test_model_path_runs_backbone_once_and_preserves_actual_head_output():
    import ast
    from pathlib import Path
    from types import SimpleNamespace
    module=Path(__file__).resolve().parents[1]/'model/gr00t_n1d6/gr00t_n1d6.py'
    tree=ast.parse(module.read_text())
    method=next(node for cls in tree.body if isinstance(cls,ast.ClassDef) and cls.name=='Gr00tN1d6'
                for node in cls.body if isinstance(node,ast.FunctionDef) and node.name=='get_action')
    method.returns=None
    namespace={}
    exec(compile(ast.fix_missing_locations(ast.Module(body=[method],type_ignores=[])),str(module),'exec'),namespace)
    memory=populated();sample,ids=features(2);calls=[]
    def backbone(inputs):calls.append(inputs);return sample
    model=SimpleNamespace(memory=memory,prepare_input=lambda x:({'input_ids':ids},{}),
        backbone=backbone,action_head=SimpleNamespace(get_action=lambda f,a:stochastic_head(f)))
    result=namespace['get_action'](model,{'memory_context':dict(episode_keys=['a','b'],timesteps=[1,1],memory_action_controls=True)})
    assert len(calls)==1 and len(result['memory_action_controls']['comparisons'])==2


def test_fixed_input_controls_keep_normal_actions_rng_and_stream_writes():
    from v1_groot_memoryvla.model.memory.diagnostics import action_controls
    memory=populated();reference=deepcopy(memory)
    sample,ids=features(2)
    context=dict(episode_keys=['a','b'],timesteps=[1,1],history_mode='normal',memory_diagnostics=True)
    torch.manual_seed(42)
    expected=stochastic_head(reference(sample,ids,context))
    expected_rng=torch.get_rng_state().clone()
    torch.manual_seed(42)
    actual=action_controls(memory,sample,ids,context,stochastic_head)
    assert torch.equal(actual['action_pred'],expected['action_pred'])
    assert torch.equal(torch.get_rng_state(),expected_rng)
    assert actual['memory_action_controls']['conditions']==['empty','other_episode']
    assert all(x['normalized_action_shift_rms']>0 for x in actual['memory_action_controls']['comparisons'])
    assert memory.last_history_control['mode']=='normal'
    assert memory.last_diagnostics==reference.last_diagnostics
    for bank,other in [(memory.perceptual,reference.perceptual),(memory.cognitive,reference.cognitive)]:
        assert bank.last_written==other.last_written and bank.merge_count==other.merge_count
        for key in bank.bank:
            assert all(a[0]==b[0] and torch.equal(a[1],b[1]) for a,b in zip(bank.bank[key],other.bank[key]))


def test_first_observation_has_zero_effect_and_unpaired_wrong_history_is_explicit():
    from v1_groot_memoryvla.model.memory.diagnostics import action_controls
    memory=populated();memory.reset();sample,ids=features(1)
    with torch.no_grad():
        result=action_controls(memory,sample,ids,dict(episode_keys=['a'],timesteps=[0]),stochastic_head)
    report=result['memory_action_controls']
    assert report['conditions']==['empty']
    assert report['comparisons'][0]['normalized_action_shift_rms']==0
    assert report['other_episode_unavailable']=='requires_distinct_paired_environments'
    assert memory.cognitive.last_written=={'a':0}
