"""Standalone A100 catalog and exact-process policy control, sent over SSH."""

import json
import os
from pathlib import Path
import re
import signal
import subprocess
import sys

if '__file__' in globals():
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]/'train'))
from run_identity import CANONICAL_RUN, V21_RUN, training_identity, training_base

PROJECT = Path('/nvme3/projects/memory_vla')
TRAINING = Path('/nvme3/data_wyh/outputs/memory_vla/v1_groot_memoryvla/training')
OFFICIAL = Path('/nvme3/huggingface/train_model/GR00T/GR00T-N1.6-BEHAVIOR1k')
_paths=globals().get('DEVICE_PATHS',{})
PROJECT=Path(_paths.get('project',PROJECT))
TRAINING=Path(_paths.get('training',TRAINING))
OFFICIAL=Path(_paths.get('official',OFFICIAL))


def read_json(path):
    try:
        return json.loads(path.read_text())
    except (OSError, ValueError):
        return {}


def checkpoint_info(checkpoint):
    path = Path(checkpoint).resolve()
    if path == OFFICIAL.resolve():
        return dict(version='official', method='official', training_id=None, train_number=None, source_digest=None,
                    run='官方模型', label='Official · GR00T-N1.6-BEHAVIOR1k · 未微调', source_dir=None)
    if path.parent.name != 'checkpoints':
        return dict(version='unversioned', run=path.name, label=path.name, source_dir=None)
    root, arm = path.parents[2], path.parents[1].name
    if not root.is_relative_to(TRAINING):
        return dict(version='unversioned', run=root.name, label=root.name+' / '+path.name, source_dir=None)
    identity=training_identity(root,arm)
    run = root.name
    if not CANONICAL_RUN.fullmatch(run) and not V21_RUN.fullmatch(run):
        run = identity.get('display_name') or training_base(identity['version'],identity['task_id'],identity['method'],legacy=True)+'_train%02d'%identity['train_number']
    info = dict(identity,run=run,label=run+' / '+path.name)
    if identity.get('version') == 'V_2_1' and not (path/'v2_1.json').is_file():
        source = Path(identity.get('source_dir') or PROJECT/'deployments/web_live')
        entry = source/'src/v2_1_groot_memoryvla/serve.py'
        live = PROJECT/'deployments/web_live'
        live_entry = live/'src/v2_1_groot_memoryvla/serve.py'
        if (not entry.is_file() or 'BaselinePolicy' not in entry.read_text()) and live_entry.is_file() and 'BaselinePolicy' in live_entry.read_text():
            info.update(inference_source_dir=str(live), source_compatibility='旧训练快照缺少 baseline 推理入口，使用兼容推理源码；训练快照保留。')
    return info


def catalog():
    checkpoints = [dict(path=str(OFFICIAL), **checkpoint_info(str(OFFICIAL)))] if (OFFICIAL/'config.json').is_file() else []
    for path in sorted(TRAINING.glob('*/*/checkpoints/*/config.json'), key=lambda p:p.stat().st_mtime, reverse=True):
        if path.parents[3].is_symlink():
            continue
        checkpoints.append(dict(path=str(path.parent), **checkpoint_info(str(path.parent))))
    sources = [str(p) for p in sorted((PROJECT/'deployments').glob('*')) if (p/'src/v1_groot_memoryvla/serve.py').is_file()]
    response = subprocess.run(['nvidia-smi','--query-gpu=index,name,memory.total,memory.used','--format=csv,noheader,nounits'],
                              capture_output=True, text=True, check=True, timeout=4)
    gpus = []
    for line in response.stdout.splitlines():
        index, name, total, used = map(str.strip, line.split(','))
        gpus.append(dict(index=int(index), name=name, total_mb=int(total), used_mb=int(used)))
    return dict(checkpoints=checkpoints, sources=sources, gpus=gpus)


def argument(args, *names, default=None):
    for i, value in enumerate(args):
        for name in names:
            if value == name and i+1 < len(args):
                return args[i+1]
            if value.startswith(name+'='):
                return value.split('=',1)[1]
    return default


def process_info(pid):
    path = Path('/proc')/str(pid)
    try:
        args = (path/'cmdline').read_bytes().decode(errors='replace').rstrip(chr(0)).split(chr(0))
        module = argument(args, '-m')
        script = next((v for v in args[1:] if v.endswith('.py')), '')
        diagnostic = script.endswith('/policy_server_diagnostic.py') and 'experiment_07/scripts/' in script
        restore_rng = any('from task40_memory_instrumentation import run_stateless_server; run_stateless_server()' in value
                          for value in args[1:4])
        native_trace = any('run_native_memory_server()' in value for value in args[1:4])
        if not Path(args[0]).name.startswith('python') or not (
            module in ('v1_groot_memoryvla.serve', 'v2_1_groot_memoryvla.serve', 'gr00t.eval.run_gr00t_server', 'baselines.groot_n1d6.serve') or
            script.endswith('run_gr00t_server.py') or diagnostic or restore_rng or native_trace or
            (script.endswith('serve.py') and ('groot' in script.lower() or 'memoryvla' in script))):
            return None
        stat = (path/'stat').read_text().rsplit(')',1)[1].split()
        if stat[0] in ('Z','X'):
            return None
        checkpoint = argument(args, '--weights') if diagnostic else argument(args, '--model-path', '--model_path')
        if not checkpoint:
            return None
        environment = {}
        try:
            environment = dict(item.split('=',1) for item in (path/'environ').read_bytes().decode(errors='replace').split(chr(0)) if '=' in item)
        except PermissionError:
            pass
        try:
            log = os.readlink(path/'fd/1')
            if not log.startswith('/') or not Path(log).is_file():
                log = None
        except OSError:
            log = None
        info=checkpoint_info(checkpoint)
        runtime=environment.get('PYTHONPATH','').split(':')[0]
        if runtime.endswith('/src'):
            source=Path(runtime).parent
            manifest=read_json(source/'web_source_manifest.json')
            info.update(training_source_digest=info.get('source_digest'),source_dir=str(source),source_digest=manifest.get('digest'))
        return dict(pid=pid, start_ticks=int(stat[19]), port=int(argument(args,'--port',default='5555')),
                    checkpoint_step=Path(checkpoint).name,checkpoint_resolved=str(Path(checkpoint).resolve()),policy_seed=argument(args,'--seed'),
                    history_mode=argument(args,'--history-mode',default='normal'),
                    checkpoint=checkpoint, gpu_index=environment.get('CUDA_VISIBLE_DEVICES', argument(args,'--device',default='—')),
                    log=log, task40_diagnostics=diagnostic or restore_rng or native_trace,
                    task40_trace=native_trace or (diagnostic and '--task40-trace' in args),
                    **info)
    except (OSError, ValueError, IndexError):
        return None


def same_process(left, right):
    return all(left.get(key) == right.get(key) for key in ('pid','port','checkpoint')) and (
        left.get('start_ticks') is None or left['start_ticks'] == right.get('start_ticks'))


def snapshot(jobs):
    response = subprocess.run(['ss','-H','-ltnp'],capture_output=True,text=True,check=True,timeout=3)
    ports = set()
    for line in response.stdout.splitlines():
        fields = line.split()
        if len(fields) >= 5:
            for pid in re.findall(r'pid=(\d+),', line):
                ports.add((int(pid), int(fields[3].rsplit(':',1)[1])))
    current = [row for path in Path('/proc').iterdir() if path.name.isdigit() and (row := process_info(int(path.name)))]
    rows = []
    for job in jobs:
        actual = next((row for row in current if same_process(job,row)), None)
        if actual:
            current.remove(actual)
            row = dict(job, **actual)
            row['state'] = 'stopping' if job.get('state') == 'stopping' else 'ready' if (row['pid'],row['port']) in ports else 'loading'
        else:
            row = dict(job, state='stopped' if job.get('state') in ('ready','stopping','stopped') else 'failed')
            info = checkpoint_info(job['checkpoint'])
            row.update({key:info[key] for key in ('version', 'run', 'label')})
        rows.append(row)
    rows += [dict(row, state='ready' if (row['pid'],row['port']) in ports else 'loading') for row in current]
    for row in rows:
        try:
            with Path(row.get('log') or '').open('rb') as log:
                log.seek(0,2);log.seek(max(0,log.tell()-12000))
                tail = log.read().decode(errors='replace')
        except OSError:
            tail = '未找到可读取的推理日志。'
        row['log_tail'] = '\n'.join(tail.splitlines()[-20:])
    return dict(jobs=rows)


def stop_process(request):
    pid = request['pid']
    try:
        # Hold a reference to this process before validating its identity, so
        # PID reuse between validation and signaling cannot target a new process.
        fd = os.pidfd_open(pid)
    except ProcessLookupError:
        return dict(request, state='stopped')
    try:
        actual = process_info(pid)
        if actual is None or not same_process(request, actual):
            raise ValueError('进程身份已变化，未关闭任何服务；请刷新列表。')
        signal.pidfd_send_signal(fd, signal.SIGTERM)
        return dict(actual, state='stopping')
    finally:
        os.close(fd)


def dispatch(action, request):
    try:
        result = catalog() if action == 'catalog' else snapshot(request) if action == 'snapshot' else checkpoint_info(request) if action == 'identity' else stop_process(request)
        print(json.dumps(result, ensure_ascii=False))
    except (OSError, ValueError, subprocess.SubprocessError) as error:
        import sys
        print(str(error), file=sys.stderr)
        raise SystemExit(1)
