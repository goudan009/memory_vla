"""Resource browsing and mutations only use isolated temporary directories."""
import importlib.util
import json
from pathlib import Path
import shlex
import subprocess
import sys
import threading
from urllib.error import HTTPError
from urllib.request import Request, urlopen

import pytest

SOURCE = Path(__file__).resolve().parents[1] / 'evaluation/scripts/registry_storage.py'


@pytest.fixture
def storage():
    spec = importlib.util.spec_from_file_location('registry_storage_test', SOURCE)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def call(storage, root, action='browse', **payload):
    return storage.operate(dict(action=action, roots=[str(root)], protected=[str(root)], path=str(root), **payload))


def test_deep_listing_pagination_preview_and_disk(storage, tmp_path):
    root = tmp_path/'models'; root.mkdir()
    deep = root/'run'/'checkpoints'/'step-000001'; deep.mkdir(parents=True)
    (deep/'config.json').write_text('{"memory": true}', encoding='utf-8')
    (deep/'weights.bin').write_bytes(b'\x00\x01\x02')
    result = storage.operate(dict(action='browse', roots=[str(root)], path=str(deep), limit=1))
    assert result['total'] == 2 and result['has_more']
    assert result['disk']['total'] > result['disk']['free'] >= 0
    assert result['parent'] == str(deep.parent)
    result = storage.operate(dict(action='browse', roots=[str(root)], path=str(deep/'config.json')))
    assert result['preview'] == '{"memory": true}'
    assert not result['truncated']
    result = storage.operate(dict(action='browse', roots=[str(root)], path=str(deep/'weights.bin')))
    assert result['preview'] is None


def test_preview_is_bounded(storage, tmp_path):
    path=tmp_path/'log.txt'; path.write_text('x'*100_000)
    result=storage.operate(dict(action='browse',roots=[str(tmp_path)],path=str(path)))
    assert result['truncated'] and len(result['preview']) == 65_536


def test_create_trash_restore_and_protect_root(storage, tmp_path):
    call(storage,tmp_path,'mkdir',name='new folder')
    folder=tmp_path/'new folder'
    (folder/'important.txt').write_text('keep me')
    request=dict(roots=[str(tmp_path)],protected=[str(tmp_path)],path=str(folder))
    deleted=storage.operate(dict(request,action='trash'))
    assert not folder.exists()
    assert (Path(deleted['trash_path'])/'important.txt').read_text() == 'keep me'
    storage.operate(dict(request,action='restore',token=deleted['token']))
    assert (folder/'important.txt').read_text() == 'keep me'
    with pytest.raises(ValueError,match='根目录'):
        call(storage,tmp_path,'trash')


def test_path_escape_and_symlink_escape_rejected(storage,tmp_path):
    root=tmp_path/'models';root.mkdir()
    outside=tmp_path/'private';outside.mkdir()
    (outside/'secret.txt').write_text('private')
    (root/'escape').symlink_to(outside,target_is_directory=True)
    for path in [outside,root/'..'/'private',root/'escape'/'secret.txt']:
        with pytest.raises(ValueError,match='范围'):
            storage.operate(dict(action='browse',roots=[str(root)],path=str(path)))
    with pytest.raises(ValueError):
        call(storage,root,'mkdir',name='../bad')


def test_restore_does_not_overwrite_new_data(storage,tmp_path):
    target=tmp_path/'file.txt';target.write_text('original')
    request=dict(roots=[str(tmp_path)],path=str(target))
    deleted=storage.operate(dict(request,action='trash'))
    target.write_text('new data')
    with pytest.raises(ValueError,match='已存在'):
        storage.operate(dict(request,action='restore',token=deleted['token']))
    assert target.read_text() == 'new data'


def test_local_registration_persists_without_rescan(storage,tmp_path):
    project=tmp_path/'project';project.mkdir()
    dataset=project/'assets/datasets/custom';dataset.mkdir(parents=True)
    service=storage.StorageService(project,{},lambda:{'items':[]})
    response=service.request('4090',dict(action='register',path=str(dataset),kind='dataset',name='新数据'))
    assert response['item']['name']=='新数据'
    assert storage.StorageService(project,{},lambda:{'items':[]}).registered()[0]['path']==str(dataset)
    assert service.request('4090',dict(action='browse',path=str(dataset)))['type']=='directory'


def test_ssh_worker_uses_same_operations_and_preserves_spaces(storage,tmp_path,monkeypatch):
    root=tmp_path/'remote folder';root.mkdir()
    (root/'config.json').write_text('{"remote":true}')
    service=storage.StorageService(tmp_path,{'a100':{'host':'test-host','paths':{'project':str(tmp_path)}}},
        lambda:{'items':[dict(host='A100',kind='weights',path='A100:'+str(root))]})
    run=subprocess.run
    def ssh(command,**kwargs):
        assert command[-2]=='test-host'
        code=shlex.split(command[-1])[-1]
        return run([sys.executable,'-c',code],**kwargs)
    monkeypatch.setattr(storage.subprocess,'run',ssh)
    result=service.request('A100',dict(action='browse',path='A100:'+str(root)))
    assert result['entries'][0]['name']=='config.json'


def test_http_resource_operations_and_origin_guard(storage,tmp_path,monkeypatch):
    from http.server import ThreadingHTTPServer
    monkeypatch.syspath_prepend(str(SOURCE.parent))
    spec=importlib.util.spec_from_file_location('registry_http_test',SOURCE.with_name('view_results.py'))
    viewer=importlib.util.module_from_spec(spec);spec.loader.exec_module(viewer)
    dataset=tmp_path/'assets/datasets/demo';dataset.mkdir(parents=True)
    catalog=dict(items=[dict(id='demo',host='4090',kind='dataset',path=str(dataset))])
    monkeypatch.setattr(viewer.experiment_registry,'cached',lambda **kwargs:catalog)
    class Handler(viewer.Handler):
        storage_service=storage.StorageService(tmp_path,{},lambda:catalog)
        def log_message(self,*args):pass
    server=ThreadingHTTPServer(('127.0.0.1',0),Handler)
    thread=threading.Thread(target=server.serve_forever,daemon=True);thread.start()
    base='http://127.0.0.1:'+str(server.server_port)
    def post(payload,origin=None):
        headers={'Content-Type':'application/json'}
        if origin:headers['Origin']=origin
        request=Request(base+'/api/registry/storage',data=json.dumps(payload).encode(),headers=headers)
        with urlopen(request) as response:return json.load(response)
    try:
        post(dict(action='mkdir',path=str(dataset),name='added'))
        with urlopen(base+'/api/registry/storage?path='+str(dataset)) as response:
            assert json.load(response)['entries'][0]['name']=='added'
        with pytest.raises(HTTPError) as rejected:
            post(dict(action='mkdir',path=str(dataset),name='blocked'),origin='https://other.example')
        assert rejected.value.code==400
        assert not (dataset/'blocked').exists()
        result=post(dict(action='register',path=str(dataset),kind='dataset',name='custom'))
        with urlopen(base+'/api/registry') as response:
            data=json.load(response)
        assert len(data['items'])==1 and data['items'][0]['id']==result['item']['id']
    finally:
        server.shutdown();server.server_close();thread.join()
