"""Isaac workers must leave after Gym closes, before native static teardown."""

import os
from pathlib import Path
import subprocess


def test_isaac_worker_flushes_after_gym_close_and_skips_static_teardown():
    root = Path(__file__).resolve().parents[1]
    environment = os.environ.copy()
    environment['PYTHONPATH'] = str(root / 'scripts')
    result = subprocess.run([
        '/nvme3/envs/b1k2025-sim/bin/python', '-c', r'''
import atexit
import os
import sys
import types
import gymnasium.vector.async_vector_env as implementation
from groot_eval_results import run_eval_async_worker
implementation._async_worker = lambda *args, **kwargs: print('GYM_CLOSED')
sys.modules['omnigibson'] = types.SimpleNamespace(app=object())
atexit.register(os.abort)
run_eval_async_worker()
''',
    ], env=environment, capture_output=True, text=True, timeout=20)
    assert result.returncode == 0, result.stdout + result.stderr
    assert 'GYM_CLOSED' in result.stdout
    assert 'EVAL_WORKER_EXIT' in result.stdout
