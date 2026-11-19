"""Diagnostic service traffic and lifecycle, without loading a VLM."""
import json
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
import sys
import threading
import urllib.request
import subprocess

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'evaluation/scripts'))
import vlm_diagnosis
import vlm_service


def test_private_vlm_requests_ignore_broken_shell_proxy(monkeypatch):
    class Handler(BaseHTTPRequestHandler):
        def do_GET(self):
            self.send_response(200)
            self.end_headers()
            self.wfile.write(b'{"ok":true,"model":"Qwen"}')

        def do_POST(self):
            self.payload = json.loads(self.rfile.read(int(self.headers['Content-Length'])))
            self.do_GET()

        def log_message(self, *args):
            pass

    server = ThreadingHTTPServer(('127.0.0.1', 0), Handler)
    worker = threading.Thread(target=server.serve_forever, daemon=True)
    worker.start()
    monkeypatch.setenv('no_proxy', '')
    monkeypatch.setenv('NO_PROXY', '')
    monkeypatch.setattr(urllib.request, '_opener', urllib.request.build_opener(
        urllib.request.ProxyHandler({'http': 'http://127.0.0.1:1'})))
    endpoint = f'http://127.0.0.1:{server.server_port}'
    try:
        assert vlm_diagnosis.health(endpoint)['ok'] is True
        assert vlm_diagnosis._post_json(endpoint + '/analyze', {'frames': []})['ok'] is True
    finally:
        server.shutdown()
        server.server_close()
        worker.join()


@pytest.mark.parametrize('active,ok,state', [
    ('active', True, 'ready'), ('active', False, 'loading'),
    ('inactive', False, 'stopped'), ('failed', False, 'failed'),
])
def test_status_distinguishes_loading_closed_and_failed(monkeypatch, active, ok, state):
    monkeypatch.setattr(vlm_service, '_remote', lambda action: dict(active=active, pid=123 if active=='active' else 0))
    monkeypatch.setattr(vlm_diagnosis, 'health', lambda: dict(ok=ok, endpoint=vlm_diagnosis.DEFAULT_ENDPOINT))
    value = vlm_service.service()
    assert value['state'] == state
    assert value['port'] == 8901 and value['gpu_index'] == 0
    assert vlm_service.endpoint({'service_id': vlm_service.SERVICE_ID, 'endpoint': 'ignored'}) == vlm_diagnosis.DEFAULT_ENDPOINT
    with pytest.raises(ValueError):
        vlm_service.endpoint({'service_id': ''})


def test_remote_lifecycle_is_idempotent_and_only_stops_its_systemd_unit(monkeypatch, capsys):
    active, pid, calls = 'inactive', 0, []
    original_read = Path.read_bytes
    monkeypatch.setattr(Path, 'read_bytes', lambda path: (
        b'/usr/bin/python3\0-u\0/home/1024201092WYH/vlm_service/vlm_service.py\0'
        if str(path)=='/proc/123/cmdline' else original_read(path)))

    def run(args, **kwargs):
        nonlocal active, pid
        calls.append(args)
        if args[:3] == ['systemctl', '--user', 'show']:
            return subprocess.CompletedProcess(args, 0, f'ActiveState={active}\nMainPID={pid}\n', '')
        if args[0] == 'systemd-run':
            active, pid = 'active', 123
        if args[:3] == ['systemctl', '--user', 'stop']:
            active, pid = 'inactive', 0
        return subprocess.CompletedProcess(args, 0, '', '')

    monkeypatch.setattr(subprocess, 'run', run)
    for action in ('start', 'start', 'stop', 'start'):
        monkeypatch.setattr(sys, 'argv', ['vlm-control', action])
        exec(vlm_service.REMOTE, {})
    assert len([call for call in calls if call[0]=='systemd-run']) == 2
    assert [call for call in calls if call[:3]==['systemctl','--user','stop']] == [
        ['systemctl', '--user', 'stop', 'VLMService.service']]
    assert all('pkill' not in call and 'kill' not in call for call in calls)
    assert json.loads(capsys.readouterr().out.splitlines()[-1])['pid'] == 123


def test_remote_refuses_to_stop_foreign_program(monkeypatch):
    monkeypatch.setattr(sys, 'argv', ['vlm-control', 'stop'])
    calls = []
    def run(args, **kwargs):
        calls.append(args)
        return subprocess.CompletedProcess(args, 0, 'ActiveState=active\nMainPID=123\n', '')
    monkeypatch.setattr(subprocess, 'run', run)
    monkeypatch.setattr(Path, 'read_bytes', lambda path: b'python\0groot_server.py\0')
    with pytest.raises(ValueError, match='another program'):
        exec(vlm_service.REMOTE, {})
    assert len(calls) == 1


def test_service_http_routes(monkeypatch):
    import view_results
    actions = []
    def service(action='status'):
        actions.append(action)
        return dict(ok=True, state='ready', logs='model ready')
    monkeypatch.setattr(vlm_service, 'service', service)
    server = ThreadingHTTPServer(('127.0.0.1', 0), view_results.Handler)
    worker = threading.Thread(target=server.serve_forever, daemon=True)
    worker.start()
    opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
    try:
        for path, method in [('', 'GET'), ('/start', 'POST'), ('/stop', 'POST'), ('/logs', 'GET')]:
            request = urllib.request.Request(f'http://127.0.0.1:{server.server_port}/api/vlm/service{path}', method=method)
            with opener.open(request, timeout=3) as response:
                assert json.load(response)['state'] == 'ready'
        assert actions == ['status', 'start', 'stop', 'logs']
    finally:
        server.shutdown(); server.server_close(); worker.join()
