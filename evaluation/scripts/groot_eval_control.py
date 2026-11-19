"""Pause one evaluation through its controller; move stopped results to trash."""
import json
import os
from pathlib import Path
import signal
import time
import uuid


def process_info(pid):
    try:
        folder = Path('/proc')/str(pid)
        stat = (folder/'stat').read_text().rsplit(')', 1)[1].split()
        if stat[0] == 'Z':
            return None
        argv = (folder/'cmdline').read_bytes().decode().rstrip('\0').split('\0')
        env = dict(part.split('=',1) for part in (folder/'environ').read_bytes().decode().split('\0') if '=' in part)
        return dict(pid=int(pid),parent=int(stat[1]),start=int(stat[19]),argv=argv,env=env)
    except (OSError, ValueError, IndexError):
        return None


def argument(argv, name):
    try:
        return argv[argv.index(name)+1]
    except (ValueError, IndexError):
        return None


class EvaluationControl:
    def __init__(self, project_root, evaluation_root=None):
        self.root = Path(project_root).resolve()
        self.evaluation = Path(evaluation_root).resolve() if evaluation_root else self.root/'evaluation'

    def run(self, request):
        if not isinstance(request,dict) or set(request) != {'run'} or not isinstance(request['run'],str) or not request['run']:
            raise ValueError('请选择一个评估目录')
        run = (self.evaluation/request['run']).resolve()
        if not run.is_relative_to(self.evaluation) or run == self.evaluation or '.trash' in run.relative_to(self.evaluation).parts:
            raise ValueError('只能操作 evaluation 下的单次评估目录')
        if not (run/'results.json').is_file():
            raise ValueError('评估目录不存在')
        return run

    def is_controller(self, process):
        return bool(process and process['env'].get('GROOT_ENTRY_ROOT') == str(self.root) and
                    Path(process['argv'][0]).name.startswith('python') and '-' in process['argv'])

    def processes(self, run):
        controllers, active = {}, set()
        marker = run/'.controller.json'
        if marker.exists():
            record = json.loads(marker.read_text())
            proc = process_info(record['pid'])
            if self.is_controller(proc) and proc['start'] == record['start_ticks']:
                controllers[proc['pid']] = proc
        for folder in Path('/proc').iterdir():
            if not folder.name.isdigit():
                continue
            proc = process_info(folder.name)
            if proc is None:
                continue
            public = argument(proc['argv'],'--public-output-dir')
            runtime = proc['env'].get('GROOT_EVAL_OUTPUT_DIR')
            if runtime and Path(runtime).is_relative_to(run):
                active.add(proc['pid'])
            if public and Path(public).resolve() == run:
                launcher = proc['env'].get('LAUNCHER_PATH','')
                if Path(launcher).resolve() != self.root/'evaluation/scripts/run_groot_priority_eval.sh':
                    continue
                active.add(proc['pid'])
                parent = process_info(proc['parent'])
                if self.is_controller(parent):
                    controllers[parent['pid']] = parent
        active.update(controllers)
        return controllers, active

    def pause(self, request):
        run = self.run(request)
        controllers, active = self.processes(run)
        if not controllers:
            if active:
                raise ValueError('检测到仿真进程，但未找到对应调度器，未向其他进程发送信号')
            return dict(state='stopped',run=request['run'],message='该评估已停止，可以续跑或删除')
        for proc in controllers.values():
            fd = os.pidfd_open(proc['pid'])
            try:
                current = process_info(proc['pid'])
                if not self.is_controller(current) or current['start'] != proc['start']:
                    raise ValueError('评估进程已变化，请刷新后重试')
                # The existing controller forwards SIGTERM and waits for its
                # own runner to close encoders and preserve episode records.
                signal.pidfd_send_signal(fd,signal.SIGTERM)
            finally:
                os.close(fd)
        return dict(state='stopping',run=request['run'],message='正在暂停，等待仿真清理；已完成回合和录像保留')

    def delete(self, request):
        run = self.run(request)
        _, active = self.processes(run)
        if active:
            raise ValueError('评估仍在运行或清理，请先暂停，停止后再删除')
        trash = self.evaluation/'.trash'
        trash.mkdir(exist_ok=True)
        destination = trash/(time.strftime('%Y%m%d_%H%M%S')+'_'+run.name+'_'+uuid.uuid4().hex[:8])
        run.rename(destination)
        return dict(state='deleted',run=request['run'],trash=str(destination),
                    message='已移入 evaluation/.trash，可恢复')
