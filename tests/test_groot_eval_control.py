"""Controller signals and recoverable deletes use CPU-only fixture processes."""
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


def test_pause_signals_only_selected_controller_then_delete_preserves_all_files(tmp_path):
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
        deleted=control.delete(request)
        assert deleted['state']=='deleted' and not run.exists()
        assert (Path(deleted['trash'])/'episodes.jsonl').read_text()=='{"episode_index":1,"success":false}\n'
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
