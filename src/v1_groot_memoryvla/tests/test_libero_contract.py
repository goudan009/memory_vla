from types import SimpleNamespace

import numpy as np
import pytest
import torch

from v1_groot_memoryvla.model.modules.eagle_backbone import EagleBackbone
from v1_groot_memoryvla.benchmarks.libero.protocol import camera_image, native_action, quat_axis_angle


@pytest.mark.parametrize('cameras',[1,2,3])
def test_visual_grids_keep_camera_and_batch_identity(cameras):
    batch, grid, dim = 2,81,4
    visual=torch.arange(batch*cameras*grid*dim).reshape(batch*cameras*grid,dim).float()
    class FakeEagle(torch.nn.Module):
        config=SimpleNamespace(image_token_index=99)
        def forward(self,**kwargs):
            return dict(hidden_states=[torch.zeros(batch,cameras*grid+2,dim)],visual_features=visual)
    backbone=EagleBackbone.__new__(EagleBackbone)
    torch.nn.Module.__init__(backbone)
    backbone.model=FakeEagle(); backbone.return_visual_features=True; backbone.eval()
    ids=torch.full((batch,cameras*grid+2),99);ids[:,-2:]=1
    result=backbone(dict(input_ids=ids,attention_mask=torch.ones_like(ids),pixel_values=[torch.zeros(3,252,252)]*(batch*cameras)))
    assert torch.equal(result.visual_features,visual.reshape(batch,cameras,grid,dim))
    assert torch.all(result.image_mask.sum(1)==cameras*grid)


def test_native_gripper_is_not_inverted_and_cameras_rotate_both_axes():
    assert native_action([0,0,0,0,0,0,-.5])[-1]==-1
    assert native_action([0,0,0,0,0,0,.5])[-1]==1
    with pytest.raises(ValueError): native_action([float('nan')]*7)
    image=np.arange(18,dtype=np.uint8).reshape(2,3,3)
    assert np.array_equal(camera_image(image)[0,0],image[-1,-1])
    assert camera_image(image).flags.c_contiguous
    assert np.allclose(quat_axis_angle([0,0,0,1]),[0,0,0])
    assert np.allclose(quat_axis_angle([0,0,1,0]),[0,0,np.pi])


def test_native_demonstration_groups_preserve_order_episode_and_tail_mask(tmp_path):
    import h5py
    from v1_groot_memoryvla.benchmarks.libero.dataset import LiberoGroups
    from v1_groot_memoryvla.benchmarks.libero.prepare import inspect_dataset
    from v1_groot_memoryvla.benchmarks.libero.protocol import atomic_json,manifest_path
    file=tmp_path/'demo.hdf5'
    with h5py.File(file,'w') as handle:
        data=handle.create_group('data')
        for i in range(2):
            demo=data.create_group('demo_'+str(i));demo.attrs['success']=i==0
            demo.create_dataset('actions',data=np.arange(17*7).reshape(17,7).astype(np.float32))
            obs=demo.create_group('obs')
            obs.create_dataset('ee_states',data=np.zeros((17,6),np.float32))
            obs.create_dataset('gripper_states',data=np.zeros((17,2),np.float32))
            for camera in ('agentview_rgb','eye_in_hand_rgb'):
                obs.create_dataset(camera,data=np.zeros((17,2,2,3),np.uint8))
    manifest=inspect_dataset(file)
    assert manifest['rejected']==['demo_1'] and manifest['episodes']==[dict(name='demo_0',length=17)]
    atomic_json(manifest_path('libero_spatial',0,tmp_path),dict(suite='libero_spatial',task=0,
        name='put_the_bowl_on_the_plate',file=str(file),**manifest))
    def processor(messages):
        value=messages[0]['content']
        assert len(value.images)==2
        return dict(action=np.ones((16,32)),action_mask=np.pad(np.ones((16,7)),((0,0),(0,25))))
    dataset=LiberoGroups('libero_spatial',[0],processor=processor,group_size=16,seed=42,data_root=tmp_path)
    group=dataset[0]
    assert group['frame_indices']==[0,8,16] and group['timesteps']==[0,1,2]
    assert group['episode_start'] and group['episode_end'] and group['episode']=='demo_0'
    assert [int(s['action_mask'].sum()) for s in group['steps']]==[16*7,9*7,7]
    assert np.count_nonzero(group['steps'][-1]['action'][1:])==0
    assert dataset[0]['frame_indices']==group['frame_indices']
