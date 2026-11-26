"""V2.1 routing and process identity through the canonical workbench launcher."""
import json
from pathlib import Path
import shlex
import subprocess
import sys

import pytest

from test_groot_inference_launch import ROOT, free_port, local_args

sys.path.insert(0, str(ROOT/'evaluation/scripts'))
import groot_inference_remote as remote


@pytest.mark.parametrize('stage', ['stage1', 'continued', 'memory'])
def test_v21_canonical_launcher_routes_all_stages(tmp_path, stage):
    args = local_args(tmp_path, free_port())
    checkpoint = tmp_path/'checkpoint'
    (checkpoint/'v2_1_run.json').write_text(json.dumps({'stage': stage}))
    (checkpoint/'COMPLETE.json').write_text('{}')
    if stage == 'stage1':
        (checkpoint/'v2_1_stage1.json').write_text('{}')
    if stage == 'memory':
        (checkpoint/'v2_1.json').write_text('{}')
        (checkpoint/'v2_1.pt').write_bytes(b'sidecar')
    serve = tmp_path/'source/src/v2_1_groot_memoryvla/serve.py'
    serve.parent.mkdir(parents=True)
    serve.write_text('')
    result = subprocess.run(args+['--check-only'], capture_output=True, text=True)
    assert result.returncode == 0, result.stderr
    info = json.loads(result.stdout)
    command = shlex.split(info['command'])
    assert command[command.index('-m')+1] == 'v2_1_groot_memoryvla.serve'
    assert command[command.index('--residual-scale')+1] == '1'
    assert command[command.index('--history-mode')+1] == 'normal'
    (checkpoint/'COMPLETE.json').unlink()
    result = subprocess.run(args+['--check-only'], capture_output=True, text=True)
    assert result.returncode != 0 and '完成标记' in result.stderr


def test_v21_catalog_preserves_canonical_identity(tmp_path, monkeypatch):
    run = 'v2.1__test01__exp14__run003'
    root = tmp_path/run
    config = root/'config'
    config.mkdir(parents=True)
    (config/'web_launch.json').write_text(json.dumps({
        'version': 'v2.1', 'train_number': 3, 'method': 'baseline', 'seed': 42}))
    checkpoint = root/'run/checkpoints'/f'{run}__step003000'
    checkpoint.mkdir(parents=True)
    monkeypatch.setattr(remote, 'TRAINING', tmp_path)
    info = remote.checkpoint_info(checkpoint)
    assert info['run'] == run
    assert info['version'] == 'V_2_1'
    assert info['training_id'] == run
    assert info['method'] == 'baseline'


def test_v21_process_is_discovered_and_blocks_port(tmp_path):
    args = local_args(tmp_path, free_port())
    port = args[args.index('--port')+1]
    checkpoint = str(tmp_path/'checkpoint')
    child = subprocess.Popen(['/usr/bin/python3', '-c', 'import time;print("ready",flush=True);time.sleep(20)',
                              '-m', 'v2_1_groot_memoryvla.serve', '--model-path', checkpoint,
                              '--port', port], stdout=subprocess.PIPE, stderr=subprocess.DEVNULL)
    try:
        assert child.stdout.readline() == b'ready\n'
        actual = remote.process_info(child.pid)
        assert actual and actual['port'] == int(port)
        result = subprocess.run(args+['--check-only'], capture_output=True, text=True)
        assert result.returncode != 0 and '可能正在加载' in result.stderr
    finally:
        child.terminate()
        child.wait(timeout=3)
