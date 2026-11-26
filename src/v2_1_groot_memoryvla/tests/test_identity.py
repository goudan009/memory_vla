"""Execute training identity helpers without loading HF/GPU dependencies."""
import ast
import hashlib
from pathlib import Path
from types import SimpleNamespace


def helpers():
    path = Path(__file__).resolve().parents[1] / 'train.py'
    tree = ast.parse(path.read_text())
    names = {'identity', 'verify_calibrated_weight'}
    nodes = [n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name in names]
    env = {'source_digest': lambda: 'source',
           'parallel': SimpleNamespace(world_size=lambda: 1),
           'hash_file': lambda p: hashlib.sha256(p.read_bytes()).hexdigest()}
    exec(compile(ast.Module(body=nodes, type_ignores=[]), str(path), 'exec'), env)
    return env


def arguments(tmp_path):
    labels = tmp_path / 'labels'
    labels.write_bytes(b'labels')
    return SimpleNamespace(stage='memory', prior='topology', fusion_layers=0,
        states=['started'], seed=42, microbatch=2, lr=1e-5, weak_weight=.5,
        state_weight=.01, checkpoint=tmp_path, dataset=tmp_path, labels=labels,
        calibration_reference=None)


def test_identity_binds_requested_state_weight(tmp_path):
    args = arguments(tmp_path)
    identity = helpers()['identity']
    first = identity(args)
    args.state_weight = .1
    assert identity(args) != first


def test_calibration_does_not_change_requested_identity(tmp_path):
    args = arguments(tmp_path)
    args.requested_state_weight = args.state_weight
    identity = helpers()['identity']
    first = identity(args)
    args.state_weight = .075
    assert identity(args) == first


def test_identity_binds_reference_contents(tmp_path):
    args = arguments(tmp_path)
    args.calibration_reference = tmp_path / 'calibration.json'
    args.calibration_reference.write_text('{"lambda_s":0.01}')
    identity = helpers()['identity']
    first = identity(args)
    args.calibration_reference.write_text('{"lambda_s":0.02}')
    assert identity(args) != first


def test_gate_verifies_actual_calibrated_weight(tmp_path):
    import pytest
    verify = helpers()['verify_calibrated_weight']
    args = arguments(tmp_path)
    verify({'lambda_s': .01}, args)
    args.state_weight = .02
    with pytest.raises(ValueError, match='calibrated'):
        verify({'lambda_s': .01}, args)
