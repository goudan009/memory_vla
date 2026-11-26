"""The real reset check must use the selected physical GPU and no policy."""
import sys
from pathlib import Path
import json
from types import SimpleNamespace
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'evaluation/scripts'))
from test_benchmark_dashboard import manager


def test_scene_reset_plan(manager):
    plan = manager.prepare(dict(kind='task40_scene_reset', gpu=0, arm='original'))
    argv = plan['stages'][0]['argv']
    assert argv[:3] == ['env', '-u', 'CUDA_VISIBLE_DEVICES']
    assert 'OMNIGIBSON_GPU_ID=0' in argv
    assert '--arm' in argv and 'original' in argv
    assert '--checkpoint' not in argv
    assert plan['config']['category'] == 'reset_regression'


def test_zero_action_images_accept_multistep_history(tmp_path):
    np = pytest.importorskip('numpy')
    Image = pytest.importorskip('PIL.Image')
    from groot_task40_trace import append_reset_trace
    chair = SimpleNamespace(name='chair', category='straight_chair',
        get_position_orientation=lambda: (np.array([1., 2., 3.]), np.array([0., 0., 0., 1.])),
        get_linear_velocity=lambda: np.zeros(3), get_angular_velocity=lambda: np.zeros(3))
    obs = {f'video.observation.images.rgb.{camera}_256_256': np.full((1, 8, 8, 3), 80, np.uint8)
           for camera in ('head', 'left_wrist', 'right_wrist')}
    scene = SimpleNamespace(objects=[chair], _initial_file={'state': {'registry': {
        'object_registry': {'chair': {'root_link': {'pos': [1., 2., 3.], 'ori': [0., 0., 0., 1.]}}}}}})
    scene._groot_background_baseline = scene._initial_file['state']['registry']['object_registry']
    append_reset_trace(SimpleNamespace(scene=scene), tmp_path,
                       dict(env_idx=0, episode_number=1, instance_id=4), 1, obs)
    row = json.loads((tmp_path/'scene_reset_gpu1_env0.jsonl').read_text())
    assert row['action_steps'] == 0 and row['furniture']['chair']['position'] == [1., 2., 3.]
    assert row['reset_baseline']['chair']['pos'] == [1., 2., 3.]
    assert row['background_baseline_fixed'] is True
    assert Image.open(tmp_path/row['images']['head']).size == (8, 8)
