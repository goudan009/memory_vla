"""Checkpoint metadata persists independently from GPU workloads and weight files."""
import json
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import sys
import threading
from urllib.error import HTTPError
from urllib.request import Request, urlopen

import pytest

sys.path[:0] = [str(Path(__file__).resolve().parents[1] / 'evaluation/scripts')]
from checkpoint_notes import CheckpointNotes
import view_results as viewer

DEVICES = {'a100': {'host': '172.28.62.17'}, 'a6000': {'host': '100.100.2.30'}}


def test_persist_edit_clear_and_checkpoint_device_isolation(tmp_path):
    store = CheckpointNotes(tmp_path / 'notes.json')
    requests = [dict(device=device, checkpoint=path, notes=text) for device, path, text in [
        ('a100', '/models/run/checkpoint1', '验证集较好'),
        ('a100', '/models/run/checkpoint2', '待评估'),
        ('a6000', '/models/run/checkpoint1', '另一台设备的权重')]]
    for request in requests:
        store.save(request, DEVICES)
    restored = CheckpointNotes(store.path)
    assert {item['notes'] for item in restored.read().values()} == {'验证集较好', '待评估', '另一台设备的权重'}
    restored.save(dict(requests[0], notes='  已完成评估\n成功 8/10  '), DEVICES)
    assert any(item['notes'] == '已完成评估\n成功 8/10' for item in restored.read().values())
    restored.save(dict(requests[0], notes=''), DEVICES)
    assert {item['notes'] for item in restored.read().values()} == {'待评估', '另一台设备的权重'}


def test_parallel_edits_preserve_other_checkpoints(tmp_path):
    store = CheckpointNotes(tmp_path / 'notes.json')
    def save(index):
        store.save(dict(device='a100', checkpoint=f'/models/checkpoint{index}', notes=f'备注{index}'), DEVICES)
    with ThreadPoolExecutor(max_workers=4) as pool:
        list(pool.map(save, range(12)))
    assert len(store.read()) == 12


@pytest.mark.parametrize('changes', [dict(device='unknown'), dict(checkpoint='relative/checkpoint'),
    dict(checkpoint='/'), dict(checkpoint='/models/../other'), dict(checkpoint='/models/\nweight'),
    dict(notes='a' * 2001), dict(notes=None), dict(extra='field')])
def test_invalid_notes_do_not_touch_store(tmp_path, changes):
    store = CheckpointNotes(tmp_path / 'notes.json')
    with pytest.raises(ValueError):
        store.save(dict(dict(device='a100', checkpoint='/models/checkpoint', notes='备注'), **changes), DEVICES)
    assert not store.path.exists()


def test_corrupt_store_is_not_silently_overwritten(tmp_path):
    store = CheckpointNotes(tmp_path / 'notes.json')
    store.path.write_text('{broken')
    with pytest.raises(ValueError):
        store.save(dict(device='a100', checkpoint='/models/checkpoint', notes='备注'), DEVICES)
    assert store.path.read_text() == '{broken'


def test_http_persistence_device_map_and_origin(tmp_path, monkeypatch):
    monkeypatch.setattr(viewer, 'CHECKPOINT_NOTES', CheckpointNotes(tmp_path / 'notes.json'))
    monkeypatch.setattr(viewer, 'load_workbench_devices', lambda _: {
        'a6000': dict(label='RTX 6000 Ada', host='user@100.100.2.30')})
    class Handler(viewer.Handler):
        inference_devices = None
    server = viewer.ThreadingHTTPServer(('127.0.0.1', 0), Handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    url = f'http://127.0.0.1:{server.server_port}/api/checkpoint-notes'
    payload = json.dumps(dict(device='a100', checkpoint='/models/checkpoint1', notes='训练中备注')).encode()
    try:
        assert json.load(urlopen(url))['devices']['a6000']['host'] == '100.100.2.30'
        request = Request(url, data=payload, headers={'Content-Type':'application/json'})
        assert json.load(urlopen(request))['notes'] == '训练中备注'
        data = json.load(urlopen(url))
        assert len(data['notes']) == 1
        assert next(iter(data['notes'].values()))['notes'] == '训练中备注'
        with pytest.raises(HTTPError) as error:
            urlopen(Request(url, data=payload, headers={'Origin':'http://other-host'}))
        assert error.value.code == 400
        assert len(json.load(urlopen(url))['notes']) == 1
    finally:
        server.shutdown(); server.server_close(); thread.join()
