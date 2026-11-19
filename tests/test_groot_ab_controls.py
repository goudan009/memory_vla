"""Exercise retained A/B launch controls without Isaac startup."""

import importlib.util
from pathlib import Path
from types import SimpleNamespace
import json
import subprocess

from test_groot_eval_launcher import _run_dry_run


def test_ab_launch_flags_preserve_original_default_and_native_horizon(tmp_path):
    base = _run_dry_run(tmp_path, '--tasks', '21')
    candidate = _run_dry_run(tmp_path, '--tasks', '21', '--episodes', '1', '--disable-viewer', '--audit-semantics', '--diagnostic-only', '--profile-cycles', '150')
    assert base.returncode == candidate.returncode == 0
    assert 'VIEWER_ENABLED: True' in base.stdout
    assert 'VIEWER_ENABLED: False; DIAGNOSTIC_ONLY: True' in candidate.stdout
    assert '--max_episode_steps 999999999' in candidate.stdout
    assert '--n_action_steps 8' in candidate.stdout
    invalid = _run_dry_run(tmp_path, '--diagnostic-only')
    assert invalid.returncode != 0
    assert '--diagnostic-only requires --profile-cycles' in invalid.stderr
    multi = _run_dry_run(tmp_path, '--n-envs', '2', '--disable-viewer')
    assert multi.returncode != 0 and 'require --n-envs 1' in multi.stderr


def test_array_audit_hash_is_content_based_for_object_text_and_leaves_inputs_unchanged():
    import numpy as np
    import sys
    scripts = Path(__file__).resolve().parents[1] / 'scripts'
    sys.path.insert(0, str(scripts))
    spec = importlib.util.spec_from_file_location('ab_audit_test', scripts / 'groot_ab_audit.py')
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    first = {'text': np.array(['task21'], dtype=object), 'state': np.array([1.0, 2.0], dtype=np.float32)}
    duplicate = {k: v.copy() for k, v in first.items()}
    assert module.describe_arrays(first) == module.describe_arrays(duplicate)
    assert np.array_equal(first['state'], duplicate['state'])


def test_dual_entry_assigns_distinct_instances_and_rejects_shared_cache(tmp_path):
    script = Path(__file__).resolve().parents[1] / 'scripts/run_groot_dual_eval.py'
    def run(*args):
        return subprocess.run(['/usr/bin/python3', str(script), '--dry-run', *args], capture_output=True, text=True)
    normal = run('--run-tag', 'test_pair')
    assert normal.returncode == 0, normal.stderr
    plan = json.loads(normal.stdout)
    assert plan['instances'] == [[0], [1]]
    for gpu, command in enumerate(plan['commands']):
        assert command[command.index('--gpu-index') + 1] == str(gpu)
        assert command[command.index('--instance-ids') + 1] == str(gpu)
        assert command[command.index('--seed') + 1] == '0'
        assert '--diagnostic-only' in command and '--audit-semantics' in command
    overlap = run('--instances-gpu0', '0', '--instances-gpu1', '0')
    assert overlap.returncode != 0 and 'disjoint' in overlap.stderr
    cache = run('--appdata-gpu0', str(tmp_path), '--appdata-gpu1', str(tmp_path))
    assert cache.returncode != 0 and 'cannot share' in cache.stderr
    full = json.loads(run('--full-episodes', '--episodes-per-gpu', '5', '--instances-gpu0', '0', '2', '4', '6', '8', '--instances-gpu1', '1', '3', '5', '7', '9').stdout)
    assert all('--diagnostic-only' not in command and '--monitor-resources' in command for command in full['commands'])


def test_viewer_pause_preserves_graph_and_other_cameras():
    scripts = Path(__file__).resolve().parents[1] / 'scripts'
    spec = importlib.util.spec_from_file_location('viewer_test', scripts / 'groot_viewer_control.py')
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    changes = []
    texture = SimpleNamespace(set_updates_enabled=lambda enabled: changes.append(enabled))
    product = SimpleNamespace(path='/viewer/replicator', hydra_texture=texture)
    viewport = SimpleNamespace(visible=True, viewport_api=SimpleNamespace(updates_enabled=True, render_product_path='/viewer/viewport'))
    viewer = SimpleNamespace(_render_product=product, _viewport=viewport, prim=object(), _annotators={'rgb': object()})
    original_prim, original_annotator = viewer.prim, viewer._annotators['rgb']
    assert module.pause_viewer(viewer) == ['/viewer/replicator', '/viewer/viewport']
    assert changes == [False] and not viewport.visible and not viewport.viewport_api.updates_enabled
    assert viewer.prim is original_prim and viewer._annotators['rgb'] is original_annotator
