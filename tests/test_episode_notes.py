"""Free text belongs to one recorded episode and never changes its score/review."""
import json
from pathlib import Path
import sys
import threading
from urllib.request import Request,urlopen
from urllib.error import HTTPError

import pytest

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'evaluation/scripts'))
import episode_notes
import view_results


@pytest.fixture
def run(tmp_path):
    folder=tmp_path/'V_1/baseline/task0_eval01';folder.mkdir(parents=True)
    (folder/'results.json').write_text('{"status":"interrupted","mean_q_score":0.2}')
    (folder/'episodes.jsonl').write_text(''.join(json.dumps(dict(episode_index=i,valid=True,
        instance_id=i,seed=i+42,success=False,task_progress=20,sim_steps=32,q_score=.2))+'\n' for i in (1,2)))
    (folder/'human_reviews').mkdir();(folder/'human_reviews/fixture.json').write_text('{"fixture":true}')
    return folder


def test_save_edit_clear_and_result_projection(run):
    originals={path:path.read_bytes() for path in run.rglob('*') if path.is_file()}
    first=episode_notes.save(run,1,' 抓取后滑落 <script> ')
    assert first['notes']=='抓取后滑落 <script>' and first['seed']==43
    assert episode_notes.read(run,2) is None
    episode_notes.save(run,2,'第二回合')
    episode_notes.save(run,1,'修改观察')
    assert episode_notes.read(run,2)['notes']=='第二回合'
    results=view_results.load_runs(run.parents[2])
    assert [e['notes'] for e in results[0]['_episodes']]==['修改观察','第二回合']
    episode_notes.save(run,1,'')
    assert episode_notes.read(run,1)['notes']==''
    for path,data in originals.items(): assert path.read_bytes()==data


@pytest.mark.parametrize('index,notes',[(0,'a'),(99,'a'),(True,'a'),(1,'x'*2001),(1,42)])
def test_note_requires_existing_episode_and_bounded_text(run,index,notes):
    with pytest.raises(ValueError):episode_notes.save(run,index,notes)


def test_real_http_notes_do_not_require_failure_classification(run):
    class Handler(view_results.Handler):root=run.parents[2]
    server=view_results.ThreadingHTTPServer(('127.0.0.1',0),Handler)
    thread=threading.Thread(target=server.serve_forever,daemon=True);thread.start()
    base=f'http://127.0.0.1:{server.server_port}/api/episode-notes'
    payload=dict(run=run.relative_to(Handler.root).as_posix(),episode_index=1,notes='仅保存备注')
    try:
        req=Request(base,data=json.dumps(payload).encode(),headers={'Content-Type':'application/json'})
        assert json.load(urlopen(req,timeout=3))['notes']=='仅保存备注'
        url=base+'?run='+payload['run']+'&episode_index=1'
        assert json.load(urlopen(url,timeout=3))['notes']=='仅保存备注'
        for field,value in [('run','../../etc'),('episode_index',99)]:
            bad=Request(base,data=json.dumps(dict(payload,**{field:value})).encode(),headers={'Content-Type':'application/json'})
            with pytest.raises(HTTPError) as error:urlopen(bad,timeout=3)
            assert error.value.code==400
    finally:server.shutdown();server.server_close();thread.join()
