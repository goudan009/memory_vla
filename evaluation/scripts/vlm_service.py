"""Manual lifecycle of the existing Qwen diagnostic service on RTX 6000 Ada.

Use the service's systemd unit, never a process-name kill or the GR00T launcher.
The remote installation and model are kept in their existing locations.
"""
from __future__ import annotations

import json
import shlex
import subprocess
import threading

import vlm_diagnosis

HOST = '1024201092WYH@100.100.2.30'
SERVICE_ID = 'a6000-qwen3-vl'
LOCK = threading.Lock()


def endpoint(request: dict) -> str:
    if 'service_id' in request and request['service_id'] != SERVICE_ID:
        raise ValueError('请选择可用的视觉诊断服务')
    return vlm_diagnosis.DEFAULT_ENDPOINT if 'service_id' in request else (
        request.get('endpoint') or vlm_diagnosis.DEFAULT_ENDPOINT)

REMOTE = r'''
import json, pathlib, subprocess, sys
action = sys.argv[1]
unit = 'VLMService.service'
directory = '/home/1024201092WYH/vlm_service'
script = directory + '/vlm_service.py'
def run(args):
    return subprocess.run(args, capture_output=True, text=True, timeout=20)
def status():
    result = run(['systemctl', '--user', 'show', unit, '-p', 'ActiveState',
                  '-p', 'SubState', '-p', 'MainPID', '-p', 'Result'])
    fields = dict(line.split('=', 1) for line in result.stdout.splitlines() if '=' in line)
    return dict(active=fields.get('ActiveState', 'inactive'),
                substate=fields.get('SubState', 'dead'),
                pid=int(fields.get('MainPID') or 0), result=fields.get('Result', ''))
value = status()
if action in ('start', 'stop'):
    if value['pid']:
        args = pathlib.Path('/proc', str(value['pid']), 'cmdline').read_bytes().split(b'\0')
        if script.encode() not in args:
            raise ValueError('VLMService unit belongs to another program; refusing to change it')
    if action == 'stop':
        result = run(['systemctl', '--user', 'stop', unit]) if value['active'] != 'inactive' else None
    elif value['active'] not in ('active', 'activating'):
        run(['systemctl', '--user', 'reset-failed', unit])
        result = run(['systemd-run', '--user', '--unit=VLMService', '--collect',
                      '--working-directory=' + directory, '--setenv=CUDA_VISIBLE_DEVICES=0',
                      '/usr/bin/python3', '-u', script, '--port', '8901'])
    else:
        result = None
    if result is not None and result.returncode:
        raise RuntimeError(result.stderr.strip() or result.stdout.strip())
    value = status()
if action == 'logs':
    value['logs'] = run(['journalctl', '--user', '-u', unit, '-n', '50', '--no-pager']).stdout
print(json.dumps(value))
'''


def _remote(action: str) -> dict:
    result = subprocess.run(
        ['ssh', '-o', 'BatchMode=yes', '-o', 'ConnectTimeout=6', HOST,
         'python3 -c ' + shlex.quote(REMOTE) + ' ' + shlex.quote(action)],
        capture_output=True, text=True, timeout=30,
    )
    if result.returncode:
        raise RuntimeError(result.stderr.strip() or '诊断设备连接失败')
    return json.loads(result.stdout)


def service(action: str = 'status') -> dict:
    if action not in ('status', 'start', 'stop', 'logs'):
        raise ValueError('不支持的诊断服务操作')
    with LOCK:
        remote = _remote(action)
    health = vlm_diagnosis.health()
    state = ('ready' if health['ok'] else 'loading' if remote['active'] in ('active', 'activating')
             else 'failed' if remote['active'] == 'failed' else 'stopped')
    return dict(health, **remote, id=SERVICE_ID, state=state, host=HOST,
                label='A6000 · Qwen3-VL-8B-Instruct', gpu_index=0, port=8901,
                model=health.get('model') or '/home/1024201092WYH/vlm_service/models/Qwen3-VL-8B-Instruct')
