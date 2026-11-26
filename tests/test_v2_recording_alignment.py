import sys
from pathlib import Path
import h5py
import numpy as np
import pytest

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'evaluation/scripts'))
from v2_recording_alignment import matching_demo


def put(data,name,actions):
    group = data.create_group(name)
    group.create_dataset('action',data=actions)
    group.create_dataset('state',data=np.zeros((len(actions)+1,2)))
    group.create_dataset('state_size',data=np.full(len(actions)+1,2))


def test_multiple_attempts_select_exact_full_action_sequence(tmp_path):
    actions = np.arange(12,dtype=np.float32).reshape(4,3)
    with h5py.File(tmp_path/'record.hdf5','w') as f:
        put(f,'demo_0',actions[:1])
        put(f,'demo_1',actions+1)
        put(f,'demo_2',actions)
        f.create_group('demo_3')  # Official wrapper skips malformed attempts.
        name,inventory = matching_demo(f,actions)
        assert name == 'demo_2' and len(inventory)==4


def test_missing_or_ambiguous_identity_is_rejected(tmp_path):
    actions = np.ones((4,3),dtype=np.float32)
    with h5py.File(tmp_path/'record.hdf5','w') as f:
        put(f,'demo_0',actions+1)
        with pytest.raises(ValueError,match='unique exact'):
            matching_demo(f,actions)
        put(f,'demo_1',actions)
        put(f,'demo_2',actions)
        with pytest.raises(ValueError,match='unique exact'):
            matching_demo(f,actions)
