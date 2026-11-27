"""Calibration failure retains real measurements without weakening its gate."""
import ast
import copy
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest


def test_failed_recheck_retains_both_passes(tmp_path):
    path = Path(__file__).resolve().parents[1] / 'train.py'
    tree = ast.parse(path.read_text())
    node = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == 'calibration')
    records = []
    updates = []
    model = SimpleNamespace(state_dict=lambda: {}, reset=lambda: None,
                            load_state_dict=lambda *a, **k: None)
    def update(*a):
        updates.append(1)
        return dict(fm_shared_grad=1e-5, state_shared_grad=100.)
    def sequence(*a):
        while True:
            yield 0, [1, 2]
    env = dict(np=np, optimizer_for=lambda *a: object(), stream=sequence,
               run_update=update, seeded=lambda *a: None,
               jsonwrite=lambda p, x: records.append((p, copy.deepcopy(x))))
    exec(compile(ast.Module(body=[node], type_ignores=[]), str(path), 'exec'), env)
    args = SimpleNamespace(seed=42, state_weight=.01, output=tmp_path)
    with pytest.raises(RuntimeError, match='state/FM calibration failed'):
        env['calibration'](None, None, model, None, None, args)
    file, diagnostic = records[-1]
    assert file == tmp_path / 'calibration_diagnostics.json'
    assert diagnostic['passed'] is False
    assert len(updates) == 40
    assert [p['lambda_s'] for p in diagnostic['passes']] == [.01, .001]
    assert all(len(p['steps']) == 20 for p in diagnostic['passes'])
    assert diagnostic['passes'][-1]['ratio'] == pytest.approx(10000.)
