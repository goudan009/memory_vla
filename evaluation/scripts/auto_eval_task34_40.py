#!/usr/bin/env python3
"""训练→推理→评估自动接力（task34 / task40 四个权重）。

做什么：
  1. 轮询 A100 上四个训练 run（task34 bas/mem、task40 bas/mem）是否跑完；
  2. 每跑完一个，就在 A100 空闲卡上按该 checkpoint 起一个策略服务；
  3. 在 4090 空闲卡上跑正式评估（single2，10 回合）；
  4. 记录得分、收服务，接着处理下一个。

刻意不做：不改训练队列、不动别人的进程、不删任何已有结果。

跑法（在 4090）：
  cd /nvme3/projects/memory_vla
  nohup python3 evaluation/scripts/auto_eval_task34_40.py >> evaluation/logs/auto_eval.log 2>&1 &

安全闸门：joint 队列（V_1_joint_task0_34_40_train01）在四个前置 run 跑完后会做一次
"新 GPU 进程"校验，只要那一刻 A100 上出现陌生 GPU 进程它就直接失败。所以
serve 只在校验不可能发生的窗口内启动。
"""

import json
import os
import re
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

PROJECT = Path('/nvme3/projects/memory_vla')
TRAIN_ROOT = Path('/nvme3/data_wyh/outputs/memory_vla/v1_groot_memoryvla/training')
QUEUE_STATUS = TRAIN_ROOT / 'queues/V_1_joint_task0_34_40_train01/status.json'
STATE_FILE = PROJECT / 'evaluation/auto_eval_task34_40_state.json'
SUMMARY = PROJECT / 'evaluation/auto_eval_task34_40_summary.md'
A100 = '1024201092WYH@100.100.2.10'
POLICY_HOST = '172.28.62.17'     # 4090 走内网直连 A100 的策略服务
PORT_BASE = 5570
EVAL_GPU_CANDIDATES = [1, 0]     # 4090 只有 0/1；0 常被别人占，优先 1
MIN_FREE_4090_MIB = 30000

RUNS = [
    dict(run='V_1_task34_bas', task=34, sub='baseline', group='baseline'),
    dict(run='V_1_task34_mem', task=34, sub='memory_v1', group='memory'),
    dict(run='V_1_task40_bas', task=40, sub='baseline', group='baseline'),
    dict(run='V_1_task40_mem', task=40, sub='memory_v1', group='memory'),
]
STEPS = 3000


def log(msg):
    print(f'[{datetime.now():%Y-%m-%d %H:%M:%S}] {msg}', flush=True)


def sh(cmd, timeout=120, check=False):
    p = subprocess.run(cmd, shell=isinstance(cmd, str), capture_output=True, text=True, timeout=timeout)
    if check and p.returncode != 0:
        raise RuntimeError(f'命令失败({p.returncode}): {cmd}\n{p.stdout}\n{p.stderr}')
    return p.stdout.strip() + (('\n' + p.stderr.strip()) if p.stderr.strip() else '')


def remote(py_code, timeout=120):
    """在 A100 上跑一段 python（走 stdin，避免引号转义），返回 stdout。"""
    p = subprocess.run(['ssh', '-o', 'BatchMode=yes', '-o', 'ConnectTimeout=10', A100, 'python3 -'],
                       input=py_code, capture_output=True, text=True, timeout=timeout)
    if p.returncode != 0:
        raise RuntimeError(f'A100 远程脚本失败: {p.stderr[:500]}')
    return p.stdout


def read_state():
    if STATE_FILE.exists():
        return json.loads(STATE_FILE.read_text())
    return {'runs': {}, 'updated': None}


def write_state(state):
    state['updated'] = datetime.now().isoformat(timespec='seconds')
    tmp = STATE_FILE.with_suffix('.tmp')
    tmp.write_text(json.dumps(state, indent=2, ensure_ascii=False))
    tmp.replace(STATE_FILE)


POLL_PY = f'''
import json, os
root = {str(TRAIN_ROOT)!r}
runs = {[r['run'] for r in RUNS]!r}
out = {{}}
for name in runs:
    d = os.path.join(root, name)
    if not os.path.isdir(d):
        out[name] = dict(exists=False)
        continue
    rec = dict(exists=True)
    try: rec['top'] = json.load(open(os.path.join(d, 'status.json')))
    except Exception as e: rec['top'] = dict(status='unreadable', error=str(e))
    rec['arms'] = {{}}
    for arm in ('baseline', 'memory_v1'):
        try: rec['arms'][arm] = json.load(open(os.path.join(d, arm, 'status.json')))
        except Exception: pass
        rec.setdefault('ckpts', {{}})[arm] = os.path.isdir(
            os.path.join(d, arm, 'checkpoints', 'step-%06d' % {STEPS}))
    if not rec.get('source'):
        for p in (os.path.join(d, 'config', 'web_process.json'),):
            try: rec['source'] = json.load(open(p)).get('source')
            except Exception: pass
joint = None
try:
    joint = json.load(open({str(QUEUE_STATUS)!r}))
except Exception as e:
    joint = dict(status='unreadable', error=str(e))
print(json.dumps({{'runs': out, 'joint': joint}}))
'''


def poll():
    raw = remote(POLL_PY, timeout=90)
    try:
        data = json.loads(raw)
        return data.get('runs', {}), data.get('joint') or {}
    except Exception:
        log(f'轮询解析失败: {raw[:300]}')
        return {}, {}


def run_complete(rec, sub):
    arm = rec.get('arms', {}).get(sub) or {}
    top = rec.get('top') or {}
    return (top.get('status') == 'completed'
            and arm.get('step') == STEPS
            and bool(arm.get('training_complete'))
            and step_ckpt_exists(rec, sub))


def step_ckpt_exists(rec, sub):
    return (rec.get('ckpts') or {}).get(sub, False)


def checkpoint_path(rec, sub):
    src = rec.get('source') or f'{PROJECT}/deployments'
    return rec, src


def serve_gate(joint):
    """True 表示现在可以在 A100 上起服务。"""
    if not joint:
        return True, 'no-queue-status'
    status = joint.get('status')
    if status == 'waiting_for_gpus':
        return False, 'joint 队列正在做 GPU 校验'
    if status == 'waiting_for_four_runs':
        # 四个依赖都跑完时校验随时会发生，必须让路
        deps = joint.get('dependencies') or []
        if deps and all(d.get('complete') for d in deps):
            return False, 'joint 队列即将做 GPU 校验'
    return True, f'joint={status}'


def a100_gpu_free():
    raw = sh(['ssh', '-o', 'BatchMode=yes', A100,
              'nvidia-smi --query-gpu=index,memory.used,memory.total --format=csv,noheader'], timeout=60)
    rows = []
    for line in raw.splitlines():
        parts = [p.strip() for p in line.split(',')]
        if len(parts) >= 3 and parts[0].isdigit():
            rows.append((int(parts[0]), int(parts[1].split()[0]), int(parts[2].split()[0])))
    rows.sort(key=lambda r: r[1])
    return rows


def gpu4090_free():
    raw = sh('nvidia-smi --query-gpu=index,memory.used,memory.total --format=csv,noheader', timeout=60)
    rows = []
    for line in raw.splitlines():
        parts = [p.strip() for p in line.split(',')]
        if len(parts) >= 3 and parts[0].isdigit():
            rows.append((int(parts[0]), int(parts[2].split()[0]) - int(parts[1].split()[0])))
    return dict(rows)


def start_serve(ckpt, source, gpu, port):
    cmd = (f'bash {PROJECT}/evaluation/scripts/run_groot_server.sh '
           f'--checkpoint {ckpt} --gpu-index {gpu} --port {port} '
           f'--source-dir {source} --host 0.0.0.0')
    out = sh(['ssh', '-o', 'BatchMode=yes', A100, cmd], timeout=180)
    log(f'起服务 GPU{gpu}:{port} -> {out.splitlines()[0] if out else ""}')
    return out


def wait_port(port, timeout=1200):
    end = time.time() + timeout
    while time.time() < end:
        r = sh(f'timeout 4 bash -c "</dev/tcp/{POLICY_HOST}/{port}" && echo UP || echo DOWN', timeout=20)
        if 'UP' in r:
            return True
        time.sleep(15)
    return False


def stop_serve(port):
    out = sh(['ssh', '-o', 'BatchMode=yes', A100,
              f'pkill -f "v1_groot_memoryvla.serve.*--port {port}" && echo KILLED || echo NONE'], timeout=60)
    log(f'收服务 {port}: {out}')


def run_eval(run, task, group, ckpt, port, gpu):
    cmd = [
        'bash', f'{PROJECT}/evaluation/scripts/run_groot_eval.sh',
        '--version', 'V_1', '--group', group,
        '--mode', 'single2', '--gpu-index', str(gpu),
        '--task', str(task), '--seed', '0',
        '--policy-host', POLICY_HOST, '--policy-port', str(port),
        '--policy-checkpoint', str(ckpt), '--training-id', run,
        '--wait-for-gpu',
    ]
    log(f'评估 {run} task{task} group={group} gpu4050={gpu} port={port}')
    p = subprocess.run(cmd, cwd=str(PROJECT), capture_output=True, text=True, timeout=12 * 3600)
    tail = (p.stdout or '')[-2000:]
    log(f'评估结束 rc={p.returncode}\n{tail}')
    return p.returncode


def read_result(run, task, group):
    base = PROJECT / 'evaluation/V_1' / group
    if not base.exists():
        return None
    cand = [d for d in base.iterdir() if d.is_dir() and f'task{task}' in d.name]
    cand.sort(key=lambda p: p.stat().st_mtime)
    for d in reversed(cand):
        f = d / 'results.json'
        if not f.exists():
            continue
        try:
            r = json.loads(f.read_text())
        except Exception:
            continue
        if r.get('training_id') == run:
            return r, d
    return None


def append_summary(rows):
    lines = ['# task34 / task40 自动评估结果', '', f'更新时间：{datetime.now():%Y-%m-%d %H:%M}', '',
             '| run | task | group | success | 有效回合 | checkpoint | 结果目录 |',
             '|---|---|---|---|---|---|---|']
    for r in rows:
        lines.append('| {training_id} | {task} | {group} | {success} | {valid} | {ckpt} | {dir} |'.format(
            training_id=r.get('training_id', '?'), task=r.get('task', '?'), group=r.get('group', '?'),
            success=r.get('success_rate', '?'), valid=r.get('valid_episodes', r.get('episodes_completed', '?')),
            ckpt=Path(str(r.get('policy_checkpoint', '?'))).name, dir=r.get('dir', '?')))
    SUMMARY.write_text('\n'.join(lines) + '\n')


def main():
    state = read_state()
    while True:
        pending = [r for r in RUNS if state['runs'].get(r['run'], {}).get('phase') not in ('done', 'failed')]
        if not pending:
            log('四个 run 全部结束，退出')
            return
        info, joint = poll()
        if not info:
            time.sleep(60)
            continue

        progressed = False
        for spec in pending:
            name = spec['run']
            entry = state['runs'].setdefault(name, {'phase': 'waiting'})
            try:
                rec = info.get(name) or {}
                if not rec.get('exists'):
                    entry['phase'] = 'waiting'
                    continue
                if not run_complete(rec, spec['sub']):
                    arm = (rec.get('arms') or {}).get(spec['sub']) or {}
                    entry['phase'] = arm.get('status') or (rec.get('top') or {}).get('status') or 'waiting'
                    entry['step'] = arm.get('step')
                    log(f'{name} 训练中 step={entry.get("step")} 全部={ {r["run"]: (state["runs"].get(r["run"]) or {}).get("step") for r in RUNS} }')
                    continue

                # 训练已完 —— 准备接力
                ok, why = serve_gate(joint)
                if not ok:
                    log(f'{name} 就绪，但暂不起服务：{why}')
                    continue
                gpus = a100_gpu_free()
                usable = [g for g in gpus if g[2] - g[1] > 20000]
                if not usable:
                    log(f'A100 无空闲卡（>20G 空闲）{gpus}，等待')
                    continue
                a_gpu = usable[0][0]
                free4090 = gpu4090_free()
                e_gpu = next((g for g in EVAL_GPU_CANDIDATES if free4090.get(g, 0) > MIN_FREE_4090_MIB), None)
                if e_gpu is None:
                    log(f'4090 无空闲卡可用 {free4090}，等待')
                    continue

                port = PORT_BASE + len([r for r in RUNS if state['runs'].get(r['run'], {}).get('phase') == 'done'])
                ckpt = TRAIN_ROOT / name / spec['sub'] / 'checkpoints' / f'step-{STEPS:06d}'
                source = rec.get('source') or f'{PROJECT}/deployments/{name}'
                entry.update(phase='serving', a100_gpu=a_gpu, eval_gpu=e_gpu, port=port, checkpoint=str(ckpt))
                write_state(state)
                log(f'{name} 训练完成，起服务 checkpoint={ckpt} A100-GPU{a_gpu} port={port}')
                start_serve(ckpt, source, a_gpu, port)
                if not wait_port(port):
                    log(f'{name} 服务 20 分钟未就绪，跳过')
                    stop_serve(port)
                    entry.update(phase='failed', error='serve timeout')
                    write_state(state)
                    continue
                entry['phase'] = 'evaluating'
                write_state(state)
                rc = run_eval(name, spec['task'], spec['group'], ckpt, port, e_gpu)
                stop_serve(port)
                got = read_result(name, spec['task'], spec['group'])
                if got:
                    result, d = got
                    result['dir'] = str(d.relative_to(PROJECT))
                    entry.update(phase='done', rc=rc, success_rate=result.get('success_rate'),
                                 valid_episodes=result.get('valid_episodes'),
                                 episodes_completed=result.get('episodes_completed'),
                                 result_dir=result['dir'])
                    log(f'{name} 完成：success={result.get("success_rate")} 有效={result.get("valid_episodes")}')
                else:
                    entry.update(phase='failed', rc=rc, error='no results.json')
                    log(f'{name} 评估结束但没找到 results.json (rc={rc})')
                write_state(state)
                rows = [dict(state['runs'][r['run']], **{'training_id': r['run']})
                        for r in RUNS if state['runs'].get(r['run'], {}).get('phase') == 'done']
                append_summary(rows)
                progressed = True
                break   # 一次只处理一个，重新轮询
            except subprocess.TimeoutExpired as e:
                log(f'{name} 子命令超时：{e}')
                continue
            except Exception as e:  # 单次失败不能拖死整条流水线
                log(f'{name} 处理异常：{type(e).__name__}: {e}')
                entry.update(phase='waiting', last_error=f'{type(e).__name__}: {e}')
                write_state(state)
                time.sleep(30)
                continue
        write_state(state)
        if not progressed:
            time.sleep(60)


if __name__ == '__main__':
    os.makedirs(PROJECT / 'evaluation/logs', exist_ok=True)
    main()
