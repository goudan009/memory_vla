"""Controller signals and permanent deletes use CPU-only fixture processes."""
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time

import pytest

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'evaluation/scripts'))
from groot_eval_control import EvaluationControl,process_info


def make_run(tmp_path):
    run=tmp_path/'evaluation/V_2/memory/task21_eval01'
    run.mkdir(parents=True)
    (run/'results.json').write_text('{"status":"running"}')
    (run/'episodes.jsonl').write_text('{"episode_index":1,"success":false}\n')
    return run,EvaluationControl(tmp_path),dict(run='V_2/memory/task21_eval01')


def test_pause_preserves_results_then_delete_permanently_removes_selected_run(tmp_path):
    run,control,request=make_run(tmp_path)
    code='import signal,time\nsignal.signal(signal.SIGTERM,lambda *a:exit(0))\nwhile True:time.sleep(.05)\n'
    process=subprocess.Popen([sys.executable,'-u','-'],stdin=subprocess.PIPE,
        env={**os.environ,'GROOT_ENTRY_ROOT':str(tmp_path)})
    other=subprocess.Popen([sys.executable,'-c','import time;time.sleep(30)'])
    try:
        process.stdin.write(code.encode());process.stdin.close()
        (run/'.controller.json').write_text(json.dumps(dict(pid=process.pid,start_ticks=process_info(process.pid)['start'])))
        with pytest.raises(ValueError,match='先暂停'):control.delete(request)
        # The dashboard runs in system Python (pidfd support); the simulator's
        # Python used by pytest was built without this OS API.
        reply=subprocess.check_output(['/usr/bin/python3','-c',
            'import sys,json;sys.path.insert(0,sys.argv[1]);from groot_eval_control import EvaluationControl;print(json.dumps(EvaluationControl(sys.argv[2]).pause(json.loads(sys.argv[3]))))',
            str(ROOT/'evaluation/scripts'),str(tmp_path),json.dumps(request)],text=True)
        assert json.loads(reply)['state']=='stopping'
        process.wait(timeout=4)
        assert other.poll() is None
        assert (run/'episodes.jsonl').is_file()
        sibling=run.with_name('task21_eval02')
        sibling.mkdir(); (sibling/'keep.txt').write_text('another evaluation')
        deleted=control.delete(request)
        assert deleted['state']=='deleted' and not run.exists()
        assert 'trash' not in deleted and not (control.evaluation/'.trash').exists()
        assert (sibling/'keep.txt').read_text()=='another evaluation'
    finally:
        for proc in [process,other]:
            if proc.poll() is None:proc.terminate()
            proc.wait(timeout=5)


def test_pause_discovers_controller_of_existing_queue_without_marker(tmp_path):
    run,control,request=make_run(tmp_path)
    queue=tmp_path/'evaluation/scripts/run_groot_priority_eval.sh'
    queue.parent.mkdir(parents=True)
    code='import subprocess,os,time\np=subprocess.Popen(["/usr/bin/python3","-c","import time;time.sleep(30)","--public-output-dir",'+repr(str(run))+'],env={**os.environ,"LAUNCHER_PATH":'+repr(str(queue))+'})\nwhile True:time.sleep(.05)\n'
    process=subprocess.Popen([sys.executable,'-u','-'],stdin=subprocess.PIPE,env={**os.environ,'GROOT_ENTRY_ROOT':str(tmp_path)})
    child=None
    try:
        process.stdin.write(code.encode());process.stdin.close()
        for _ in range(80):
            controllers,active=control.processes(run)
            if process.pid in controllers:
                child=next(pid for pid in active if pid!=process.pid);break
            time.sleep(.02)
        assert process.pid in controllers
    finally:
        if child:
            try:os.kill(child,signal.SIGTERM)
            except ProcessLookupError:pass
        process.terminate();process.wait(timeout=5)


def test_stale_identity_does_not_signal_reused_pid_or_unrelated_policy(tmp_path):
    run,control,request=make_run(tmp_path)
    process=subprocess.Popen([sys.executable,'-u','-'],stdin=subprocess.PIPE,env={**os.environ,'GROOT_ENTRY_ROOT':str(tmp_path)})
    try:
        process.stdin.write(b'import time;time.sleep(30)\n');process.stdin.close()
        (run/'.controller.json').write_text(json.dumps(dict(pid=process.pid,start_ticks=process_info(process.pid)['start']+1)))
        assert control.pause(request)['state']=='stopped'
        assert process.poll() is None
    finally:process.terminate();process.wait(timeout=5)


@pytest.mark.parametrize('path',['','../','/tmp','V_2','V_2/memory','../../etc','.trash/run'])
def test_control_rejects_broad_and_non_result_targets(tmp_path,path):
    _,control,_=make_run(tmp_path)
    with pytest.raises(ValueError):control.delete(dict(run=path))


def test_finish_retains_partial_results_and_all_artifacts(tmp_path):
    run,control,request=make_run(tmp_path)
    original=dict(status='interrupted',requested_episodes=100,completed_episodes=17,
                  recorded_episodes=18,invalid_episodes=1,successful_episodes=3,
                  success_rate=3/17,mean_q_score=.3,configuration={'repetitions':10})
    (run/'results.json').write_text(json.dumps(original))
    (run/'video.mp4').write_bytes(b'fixture video')
    journal=(run/'episodes.jsonl').read_bytes()
    reply=control.finish(request)
    assert reply['state']=='paused'
    saved=json.loads((run/'results.json').read_text())
    assert saved['resume_allowed'] is True and saved['ended_by_user'] is False
    assert saved['paused_at']
    for key,value in original.items():
        if key!='status': assert saved[key]==value
    assert (run/'episodes.jsonl').read_bytes()==journal
    assert (run/'video.mp4').read_bytes()==b'fixture video'
    assert control.finish(request)['state']=='paused'
    assert json.loads((run/'results.json').read_text())==saved



def test_finish_rejects_active_or_completed_evaluation(tmp_path,monkeypatch):
    run,control,request=make_run(tmp_path)
    (run/'results.json').write_text('{"status":"interrupted"}')
    monkeypatch.setattr(control,'processes',lambda run:([123],[123]))
    with pytest.raises(ValueError,match='先暂停'):control.finish(request)
    monkeypatch.setattr(control,'processes',lambda run:([],[]))
    for status in ('running','complete'):
        (run/'results.json').write_text(json.dumps(dict(status=status)))
        with pytest.raises(ValueError,match='只能结束'):control.finish(request)
        assert json.loads((run/'results.json').read_text())['status']==status
    with pytest.raises(ValueError):control.finish(dict(run='../'))


def test_finish_http_route_updates_only_selected_run(tmp_path):
    import threading
    from types import SimpleNamespace
    from urllib.request import Request, urlopen
    import view_results as viewer
    run,control,request=make_run(tmp_path)
    (run/'results.json').write_text('{"status":"interrupted","completed_episodes":2}')
    class Handler(viewer.Handler):
        launch_manager=SimpleNamespace(finish=control.finish)
        evaluation_devices={}
    with viewer.ThreadingHTTPServer(('127.0.0.1',0),Handler) as server:
        thread=threading.Thread(target=server.serve_forever,daemon=True);thread.start()
        try:
            req=Request(f'http://127.0.0.1:{server.server_port}/api/launch/finish',
                        data=json.dumps(request).encode(),headers={'Content-Type':'application/json'})
            with urlopen(req,timeout=3) as response:
                assert json.load(response)['state']=='paused'
            assert json.loads((run/'results.json').read_text())['completed_episodes']==2
        finally:server.shutdown();thread.join()


@pytest.mark.parametrize('action', ['finish', 'delete'])
def test_remote_control_supports_frozen_launcher_and_updates_local_cache(tmp_path,monkeypatch,action):
    from types import SimpleNamespace
    import groot_web_launch
    import groot_device_eval_remote as remote
    from groot_device_eval import DeviceEvaluationManager
    run,control,request=make_run(tmp_path)
    (run/'results.json').write_text('{"status":"interrupted","completed_episodes":2}')
    remote_root=tmp_path/'remote'
    remote_run=remote_root/'evaluation'/request['run']
    remote_run.mkdir(parents=True)
    (remote_run/'results.json').write_text((run/'results.json').read_text())
    setup=remote_root/'setup';setup.mkdir()
    (setup/'evaluation_jobs.json').write_text(json.dumps([
        dict(folder=request['run'],source=str(remote_root),output=str(remote_run))]))
    # Use current control even if the frozen launcher predates these behaviors.
    monkeypatch.setattr(groot_web_launch,'LaunchManager',lambda *args:SimpleNamespace())
    monkeypatch.setenv('GROOT_EVAL_ROOT',str(remote_root/'evaluation'))
    for name in ('GROOT_REPO','OMNIGIBSON_ROOT','BDDL_ROOT','GROOT_TASK40_EXPERIMENT_ROOT'):
        monkeypatch.setenv(name,'fixture')
    manager=object.__new__(DeviceEvaluationManager)
    manager.root=tmp_path;manager.device_id='a6000'
    (run/'.device.json').write_text(json.dumps(dict(device='a6000',remote_run=request['run'])))
    manager._remote=lambda action,payload:remote.dispatch(dict(project=str(remote_root),
        action=action,payload=payload,environment={'GROOT_EVAL_ROOT':str(remote_root/'evaluation')}))
    result=getattr(manager,action)(request)
    if action=='delete':
        assert result['state']=='deleted'
        assert not run.exists() and not remote_run.exists()
        assert not (control.evaluation/'.trash').exists()
        assert not (remote_root/'evaluation/.trash').exists()
        assert json.loads((setup/'evaluation_jobs.json').read_text())==[]
        return
    assert result['state']=='paused'
    for path in (run,remote_run):
        saved=json.loads((path/'results.json').read_text())
        assert saved['status']=='paused' and saved['resume_allowed'] is True
        assert saved['completed_episodes']==2
