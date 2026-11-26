import numpy as np
import pytest
import torch
from unittest.mock import patch
from .test_state_memory import TinyBackbone,tiny_config,tiny_inputs
from v2_groot_memoryvla.model.gr00t_n1d6 import Gr00tN1d6
from v2_groot_memoryvla.task40_training import cached_forward,frozen_hash,part_loss_sums

from v2_groot_memoryvla.task40_training import (validate_label_manifest,window_labels,joint_weights)


def test_only_certified_physical_labels_and_complete_split():
    split = dict(train=list(range(180)),validation=list(range(180,200)))
    manifest = dict(labels_certified=True,label_names=['started'],episodes=[dict(episode=i) for i in range(200)])
    assert len(validate_label_manifest(manifest,split)) == 200
    for extra in (dict(labels_certified=False),dict(label_names=['door_open']),dict(episodes=[])):
        with pytest.raises(ValueError):
            validate_label_manifest(dict(manifest,**extra),split)
    with pytest.raises(ValueError):
        validate_label_manifest(manifest,dict(split,validation=list(range(20))))


def test_real_labels_use_current_observation_frame_and_keep_unknown():
    labels = np.array([[0.],[1.],[0.],[1.]],dtype=np.float32)
    known = np.array([[True],[False],[True],[True]])
    y,mask = window_labels(dict(labels=labels,known=known),[0,1,3])
    assert y[:,0].tolist() == [0,1,1]
    assert mask[:,0].tolist() == [True,False,True]
    with pytest.raises(ValueError):
        window_labels(dict(labels=labels,known=known),[4])


def test_action_tail_and_unknown_state_normalize_separately():
    seq = dict(actions=dict(action_mask=torch.tensor([[1.,1.],[1.,1.],[1.,0.],[0.,0.]])),
               state_known=torch.tensor([[True],[False],[True],[True]]))
    first,second = joint_weights(seq,0,2),joint_weights(seq,2,4)
    assert first == pytest.approx((4/5,1/3))
    assert second == pytest.approx((1/5,2/3))


def test_body_loss_logging_uses_masked_actual_dimensions_without_changing_graph():
    raw = torch.tensor([[[2.,4.,0.,99.]],[[6.,0.,0.,99.]]],requires_grad=True)
    parts = dict(base=dict(start=0,end=2), arm=dict(start=2,end=3))
    assert part_loss_sums(raw,parts) == [12.,0.]
    raw.sum().backward()
    assert torch.equal(raw.grad,torch.ones_like(raw))


def test_cached_eagle_output_bypasses_image_encoder_but_keeps_joint_graph():
    with patch('v1_groot_memoryvla.model.gr00t_n1d6.gr00t_n1d6.get_backbone_cls',return_value=TinyBackbone), \
         patch('v1_groot_memoryvla.model.gr00t_n1d6.processing_gr00t_n1d6.Gr00tN1d6DataCollator',return_value=None):
        model = Gr00tN1d6(tiny_config())
    model.train()
    batch = tiny_inputs()
    with patch('v2_groot_memoryvla.task40_training.inputs',return_value=batch), \
         patch.object(model.backbone,'forward',side_effect=AssertionError('Cached features must not reenter Eagle')):
        output = cached_forward(model,None,0,3)
    output['loss'].backward()
    assert model.memory.cognitive.state_head.proj.weight.grad.abs().sum()>0
    assert output['action_loss_unreduced'].shape[:2] == (3,2)


def test_frozen_hash_supports_bfloat16_and_zero_dimensional_residual_scale():
    model = torch.nn.Module()
    model.register_parameter('scalar',torch.nn.Parameter(torch.tensor(.2),requires_grad=False))
    model.register_parameter('bf16',torch.nn.Parameter(torch.ones(3,dtype=torch.bfloat16),requires_grad=False))
    first = frozen_hash(model)
    assert frozen_hash(model) == first
    model.scalar.data.add_(.1)
    assert frozen_hash(model) != first
