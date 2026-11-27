"""Shared LIBERO contract. Importable by the web server without ML dependencies."""
import ast
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[4]
DATA_ROOT = ROOT/'assets/datasets/libero'
DATA_REPO = 'yifengzhu-hf/LIBERO-datasets'
DATA_REVISION = 'f13aa24a3da8c43c7225569f28c562979fa0e35a'
LIBERO_REVISION = '8f1084e3132a39270c3a13ebe37270a43ece2a01'
HORIZONS = {'libero_spatial': 220, 'libero_object': 280, 'libero_goal': 300,
            'libero_10': 520, 'libero_90': 400}
KEYS = ('x', 'y', 'z', 'roll', 'pitch', 'yaw', 'gripper')
CONTRACT = dict(version=1, embodiment='libero_panda', state_dim=8, action_dim=7,
                images=['image', 'wrist_image'], image_transform='rotate180',
                action_format='native_LIBERO_OSC_POSE', gripper='-1=open,+1=close',
                state_rotation='axis_angle', action_horizon=16, action_steps=8)


def task_names(root=ROOT):
    file = Path(root)/'third_party/LIBERO/libero/libero/benchmark/libero_suite_task_map.py'
    tree = ast.parse(file.read_text())
    return ast.literal_eval(next(n.value for n in tree.body if isinstance(n, ast.Assign)))


def dataset_path(suite, task, data_root=DATA_ROOT):
    return Path(data_root)/suite/(task_names()[suite][task]+'_demo.hdf5')


def manifest_path(suite, task, data_root=DATA_ROOT):
    return Path(data_root)/'manifests'/suite/f'task-{task}.json'


def read_json(path, default=None):
    return json.loads(Path(path).read_text()) if Path(path).is_file() else default


def atomic_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix('.partial.json')
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2)+'\n')
    temporary.replace(path)


def camera_image(image):
    import numpy as np
    return np.ascontiguousarray(image[::-1, ::-1])


def state_dict(eef, gripper):
    import numpy as np
    eef, gripper = np.asarray(eef, dtype=np.float32), np.asarray(gripper, dtype=np.float32)
    assert eef.shape[-1] == 6 and gripper.shape[-1] == 2
    return {**{name: eef[..., i:i+1] for i, name in enumerate(KEYS[:6])}, 'gripper': gripper}


def action_dict(actions):
    import numpy as np
    actions = np.asarray(actions, dtype=np.float32)
    assert actions.shape[-1] == 7
    return {name: actions[..., i:i+1] for i, name in enumerate(KEYS)}


def quat_axis_angle(quaternion):
    import numpy as np
    q = np.asarray(quaternion, dtype=np.float64).copy()
    q /= np.linalg.norm(q)
    if q[3] < 0: q = -q
    denominator = np.sqrt(max(0., 1.-float(q[3])**2))
    return (np.zeros(3) if denominator < 1e-8 else q[:3]*2*np.arccos(np.clip(q[3],-1,1))/denominator).astype(np.float32)


def native_action(values):
    """Our HDF5 labels already use native gripper signs: no RLDS inversion."""
    import numpy as np
    value = np.asarray(values, dtype=np.float32)
    if value.shape != (7,) or not np.isfinite(value).all():
        raise ValueError('Expected seven finite native LIBERO action values')
    value = np.clip(value, -1, 1)
    value[-1] = 1. if value[-1] >= 0 else -1.
    return value
