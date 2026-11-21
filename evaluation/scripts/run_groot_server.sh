#!/usr/bin/env bash
# Start a GR00T checkpoint for B1K evaluation on the selected GPU and port.
set -euo pipefail
exec /usr/bin/python3 -u - "$@" <<'PY'
import argparse
import json
import os
from pathlib import Path
import shlex
import shutil
import socket
import subprocess

PROJECT = Path('/nvme3/projects/memory_vla')
DEFAULT_SOURCE = PROJECT / 'deployments/20261109_2100_task21_ddp_async_s42_resume_r4'
parser = argparse.ArgumentParser(
    prog='run_groot_server.sh',
    description='指定checkpoint、GPU和端口，在A100启动B1K模型服务。')
parser.add_argument('--checkpoint', type=Path, required=True, help='完整checkpoint目录，不自动选择权重')
parser.add_argument('--gpu-index', type=int, required=True, help='物理GPU编号')
parser.add_argument('--port', type=int, required=True, help='服务监听端口')
parser.add_argument('--host', default='172.28.62.17', help='监听地址，默认172.28.62.17')
parser.add_argument('--source-dir', type=Path, default=DEFAULT_SOURCE, help='模型源码目录')
parser.add_argument('--python-path', type=Path, default=PROJECT / 'envs/n1d6-p0/bin/python',
                    help='模型Python环境')
parser.add_argument('--log-dir', type=Path, help='日志目录，训练checkpoint默认归所属run的logs')
parser.add_argument('--seed', type=int, default=42, help='动作随机种子，默认42')
parser.add_argument('--history-mode', choices=('normal','empty','other_episode'), default='normal',
                    help='历史读取对照；other_episode要求至少两个并行环境')
parser.add_argument('--task40-diagnostics', action='store_true',
                    help='使用现有 Task40 轨迹记录协议，每个端口只绑定一个采集 worker')
parser.add_argument('--foreground', action='store_true', help='前台运行，日志直接显示在终端')
parser.add_argument('--dry-run', action='store_true', help='只打印命令，不启动服务或创建文件')
parser.add_argument('--check-only', action='store_true', help='检查权重、源码、环境和端口，不启动服务')
parser.add_argument('--json', action='store_true', help='输出供网页调用的JSON')
args = parser.parse_args()
if args.gpu_index < 0 or not 1 <= args.port <= 65535:
    parser.error('gpu-index必须非负，port必须在1至65535之间')
checkpoint = args.checkpoint.expanduser().resolve()
source = args.source_dir.expanduser().resolve()
python = args.python_path.expanduser().absolute()
if checkpoint.parent.name == 'checkpoints':
    default_logs = checkpoint.parents[2] / 'logs'
    checkpoint_label = f'{checkpoint.parents[2].name}_{checkpoint.name}'
else:
    default_logs = Path('/nvme3/data_wyh/outputs/memory_vla/policy_servers/logs')
    checkpoint_label = checkpoint.name
log_dir = (args.log_dir or default_logs).expanduser().resolve()
log = log_dir / f'policy_{checkpoint_label}_port{args.port}.log'
environment = dict(os.environ, CUDA_VISIBLE_DEVICES=str(args.gpu_index), PYTHONNOUSERSITE='1',
                   PYTHONDONTWRITEBYTECODE='1', NO_ALBUMENTATIONS_UPDATE='1',
                   HF_HUB_OFFLINE='1', TRANSFORMERS_OFFLINE='1', PYTHONPATH=str(source / 'src'))
command = [str(python), '-u', '-m', 'v1_groot_memoryvla.serve',
           '--model-path', str(checkpoint), '--embodiment-tag', 'BEHAVIOR_R1_PRO',
           '--device', 'cuda', '--host', args.host, '--port', str(args.port),
           '--use-sim-policy-wrapper', '--seed', str(args.seed)]
if args.history_mode != 'normal':
    command += ['--history-mode', args.history_mode]
diagnostic_script = source / 'Output/test_01/experiment_07/scripts/policy_server_diagnostic.py'
if args.task40_diagnostics:
    if args.history_mode != 'normal':
        parser.error('Task40 采集的记忆模式由评估作业控制')
    log_dir = (args.log_dir or PROJECT / 'Output/test_01/experiment_09/policy_services').expanduser().resolve()
    log = log_dir / f'policy_{checkpoint_label}_port{args.port}.log'
    output = log_dir / f'port{args.port}'
    instrumentation_dir = source / 'evaluation/scripts'
    if not (instrumentation_dir / 'task40_memory_instrumentation.py').is_file():
        instrumentation_dir = PROJECT / 'evaluation/scripts'
    bootstrap = ('import sys; sys.path.insert(0, sys.argv.pop(1)); '
                 'from task40_memory_instrumentation import run_legacy_server; run_legacy_server()')
    # Keep the archived script in argv for process discovery on the workbench.
    command = [str(python), '-u', '-c', bootstrap, str(instrumentation_dir), str(diagnostic_script), '--task40-trace',
               '--weights', str(checkpoint), '--output', str(output),
               '--anchor', str(source / 'Output/test_01/experiment_01/checkpoints'),
               '--host', args.host, '--port', str(args.port), '--remote-policy']
result = dict(gpu_index=args.gpu_index, host=args.host, port=args.port, checkpoint=str(checkpoint),
              source_dir=str(source), python_path=str(python), seed=args.seed, history_mode=args.history_mode,
              log=str(log), command=shlex.join(command), task40_diagnostics=args.task40_diagnostics,
              task40_trace=args.task40_diagnostics)
if not args.json:
    print(f'GPU: {args.gpu_index} | 地址: {args.host}:{args.port}')
    print(f'权重: {checkpoint}')
    print(f'源码: {source}')
    print(f'日志: {"当前终端" if args.foreground else log}')
    print(f'命令: {shlex.join(command)}')
if args.dry_run:
    print(json.dumps(dict(result, state='preview'), ensure_ascii=False) if args.json else 'dry-run：未启动服务。')
    raise SystemExit(0)
for path in (python, source / 'src/v1_groot_memoryvla/serve.py', checkpoint / 'config.json'):
    if not path.is_file():
        parser.error(f'文件不存在：{path}；请在A100执行或通过参数指定路径')
if args.task40_diagnostics and not diagnostic_script.is_file():
    parser.error(f'Task40 诊断入口缺失：{diagnostic_script}')
if args.task40_diagnostics and not (instrumentation_dir / 'task40_memory_instrumentation.py').is_file():
    parser.error(f'Task40 诊断模块缺失：{instrumentation_dir / "task40_memory_instrumentation.py"}')
# config.json is written before model shards during save_pretrained. A visible
# checkpoint directory alone does not mean it is ready to serve.
weights = []
for name in ('model.safetensors.index.json', 'pytorch_model.bin.index.json'):
    index = checkpoint / name
    if index.is_file():
        try:
            weights = [checkpoint / shard for shard in set(json.loads(index.read_text())['weight_map'].values())]
        except (OSError, ValueError, KeyError, TypeError):
            parser.error(f'权重尚未保存完整：{index}')
        break
else:
    weights = [path for path in (checkpoint / 'model.safetensors', checkpoint / 'pytorch_model.bin') if path.is_file()]
if not weights or any(not path.is_file() or path.stat().st_size == 0 for path in weights):
    parser.error(f'权重尚未保存完整：{checkpoint}；等待训练保存完成后再启动推理')
# A loading policy has not bound its port yet. Reject another launch for that
# port too, rather than letting two models race through loading on the GPU.
for proc in Path('/proc').iterdir():
    if not proc.name.isdigit():
        continue
    try:
        argv = (proc / 'cmdline').read_bytes().decode(errors='replace').rstrip('\0').split('\0')
        if ((('-m' in argv and argv[argv.index('-m') + 1] in
                ('v1_groot_memoryvla.serve', 'gr00t.eval.run_gr00t_server', 'baselines.groot_n1d6.serve'))
                or any(value.endswith('/policy_server_diagnostic.py') for value in argv))
                and '--port' in argv and argv[argv.index('--port') + 1] == str(args.port)):
            parser.error(f'端口{args.port}已有推理进程PID={proc.name}（可能正在加载），未启动新服务')
    except (OSError, IndexError):
        continue
try:
    with socket.socket() as probe:
        probe.bind((args.host, args.port))
except OSError as error:
    parser.error(f'端口不可用：{args.host}:{args.port} ({error})；未启动新进程，不改动现有服务')
if args.check_only:
    print(json.dumps(dict(result, state='checked'), ensure_ascii=False) if args.json else '检查通过，未启动服务。')
    raise SystemExit(0)
if source.name == 'web_live':
    manifest=json.loads((source/'web_source_manifest.json').read_text())
    frozen=source.parent/('inference_'+manifest['digest'])
    if not frozen.exists():
        shutil.copytree(source,frozen,symlinks=True)
    if args.task40_diagnostics:
        command = [str(frozen / Path(arg).relative_to(source))
                   if arg.startswith(str(source) + '/') else arg for arg in command]
    source=frozen
    environment['PYTHONPATH']=str(source/'src')
    result['source_dir']=str(source)
    result['command']=shlex.join(command)
if args.foreground:
    os.chdir(source)
    os.execvpe(str(python), command, environment)
if args.task40_diagnostics:
    (output / 'metrics').mkdir(parents=True, exist_ok=True)
    (output / 'checkpoints').mkdir(exist_ok=True)
log.parent.mkdir(parents=True, exist_ok=True)
with log.open('a') as stream:
    process = subprocess.Popen(command, cwd=source, env=environment, stdin=subprocess.DEVNULL,
                               stdout=stream, stderr=subprocess.STDOUT, start_new_session=True)
if args.json:
    print(json.dumps(dict(result, pid=process.pid, state='loading'), ensure_ascii=False))
else:
    print(f'已提交后台启动，PID={process.pid}；模型加载完成以日志中的服务就绪信息为准。')
    print(f'查看日志: tail -n 30 -F {shlex.quote(str(log))}')
PY
