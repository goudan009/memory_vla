"""Stage API uses the existing server; raw official metrics remain immutable."""
from http.server import ThreadingHTTPServer
import json
from pathlib import Path
import sys
import threading
import urllib.error
import urllib.request

import pytest

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'evaluation/scripts'))
import view_results


@pytest.fixture
def server(tmp_path):
    run=tmp_path/'V_1/memory/task40_eval01';run.mkdir(parents=True)
    (run/'results.json').write_text(json.dumps(dict(task_id=40,evaluation_mode='official-eval',configuration={})))
    (run/'episodes.jsonl').write_text(json.dumps(dict(episode_index=1,instance_id=0,seed=42,gpu_index=0,env_idx=0,
        sim_steps=1000,valid=True,q_score=.5,native_termination={'success':False})))
    class Handler(view_results.Handler):
        root=tmp_path
    httpd=ThreadingHTTPServer(('127.0.0.1',0),Handler)
    thread=threading.Thread(target=httpd.serve_forever,daemon=True);thread.start()
    yield f'http://127.0.0.1:{httpd.server_port}',run
    httpd.shutdown();httpd.server_close();thread.join()


def test_report_and_three_export_formats(server):
    base,run=server
    before={p:p.read_bytes() for p in run.iterdir()}
    url=base+'/api/vlm/stages?run=V_1/memory/task40_eval01'
    data=json.load(urllib.request.urlopen(url))
    assert data['summary']['official_mean_q']==.5
    assert data['summary']['official_success_rate']['numerator']==0
    assert data['episodes'][0]['trace_samples']==0
    request=urllib.request.Request(base+'/api/vlm/stages/batch',data=json.dumps({'run':'V_1/memory/task40_eval01'}).encode(),headers={'Content-Type':'application/json'})
    archived=json.load(urllib.request.urlopen(request))
    assert len(archived['artifacts'])==3
    assert (run/'stage_diagnostics/reports/task40_stages.csv').is_file()
    for format in ['json','csv','md']:
        response=urllib.request.urlopen(base+'/api/vlm/stages/export?run=V_1/memory/task40_eval01&format='+format)
        assert response.status==200 and response.read()
    assert all(p.read_bytes()==content for p,content in before.items())
    with pytest.raises(urllib.error.HTTPError) as e:
        urllib.request.urlopen(base+'/api/vlm/stages?run=../../outside')
    assert e.value.code==400


def test_review_roundtrip_and_identity_validation(server):
    base,run=server
    payload=dict(run='V_1/memory/task40_eval01',episode_index=1,decision='unknown',events=[],notes='遮挡')
    url=base+'/api/vlm/stages/review'
    request=urllib.request.Request(url,data=json.dumps(payload).encode(),headers={'Content-Type':'application/json'})
    result=json.load(urllib.request.urlopen(request))
    assert result['source']=='human'
    data=json.load(urllib.request.urlopen(base+'/api/vlm/stages?run=V_1/memory/task40_eval01'))
    assert data['episodes'][0]['human_review']['decision']=='unknown'
    payload['episode_index']=2
    request=urllib.request.Request(url,data=json.dumps(payload).encode(),headers={'Content-Type':'application/json'})
    with pytest.raises(urllib.error.HTTPError) as e:urllib.request.urlopen(request)
    assert e.value.code==400


def test_asset_is_served_by_existing_server(server):
    base,_=server
    response=urllib.request.urlopen(base+'/task40_diagnostics.js')
    assert 'javascript' in response.headers['Content-Type']
    assert b'task40Diagnostics' in response.read()
