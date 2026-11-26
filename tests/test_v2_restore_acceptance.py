"""Approximate restored RGB is sufficient for ordinary closed-loop evaluation."""
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'evaluation/scripts'))
from v2_task40_paired_evaluation import simulator_command, simulator_environment


def test_restore_command_defaults_to_existing_approximate_checks():
    protocol = {'closed_loop': {'cycles': 60}}
    item = {'checkpoint': 'anchor', 'prefix': '/recorded/history'}
    argv = simulator_command(ROOT, item, Path('/output'), Path('/seeds.json'), 5598, protocol)
    assert '--require-exact-restore-rgb' not in argv
    assert '--renderer-audit' not in argv  # Do not bypass the existing RGB tolerance.
    assert '--rebuild-policy-history' in argv
    assert argv[argv.index('--restore-checkpoint') + 1] == str(ROOT / 'anchor')
    assert argv[argv.index('--parent-trial') + 1] == item['prefix']
    assert argv[argv.index('--cycles') + 1] == '60'


def test_exact_rgb_is_available_only_as_explicit_protocol_opt_in():
    protocol = {'closed_loop': {'cycles': 60, 'require_exact_restore_rgb': True}}
    item = {'checkpoint': 'anchor', 'prefix': '/recorded/history'}
    argv = simulator_command(ROOT, item, Path('/output'), Path('/seeds.json'), 5598, protocol)
    assert argv.count('--require-exact-restore-rgb') == 1


def test_simulator_uses_selected_physical_gpu_not_hardcoded_zero(monkeypatch):
    monkeypatch.setenv('CUDA_VISIBLE_DEVICES', '1')
    gpu, env = simulator_environment(Path('/output'))
    assert gpu == 1
    assert env['OMNIGIBSON_GPU_ID'] == '1'
    assert 'CUDA_VISIBLE_DEVICES' not in env
    assert env['OMNIGIBSON_APPDATA_PATH'] == '/output/appdata'
