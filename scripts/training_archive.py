"""Deterministic, read-only training evidence archive and optional Git publishing.

Run --once from a systemd timer. The initial scan baselines historical terminal
jobs; --backfill RUN explicitly imports an old run. New jobs and subsequent
stop/resume/finish events are captured automatically. No GPU actions or LLM calls.
"""
import argparse
import base64
from collections import defaultdict
from datetime import datetime, timedelta, timezone
import fcntl
import gzip
import hashlib
import json
import math
from pathlib import Path
import re
import subprocess
from urllib.error import HTTPError
from urllib.parse import urlencode
from urllib.request import ProxyHandler, build_opener

ROOT = Path(__file__).resolve().parents[1]
REPORT = Path('reports/training')
STATE = Path('outputs/shared/training_archive/state.json')
TERMINAL = {'complete', 'completed', 'stopped', 'paused', 'failed', 'cancelled'}
EVALUATION_TERMINAL = TERMINAL | {'finished', 'interrupted', 'aborted', 'canceled'}


def sha(data):
    return hashlib.sha256(data).hexdigest()


def encoded(value):
    return (json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + '\n').encode()


def write(path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists() and path.read_bytes() == data:
        return
    pending = path.with_name(path.name+'.pending')
    pending.write_bytes(data)
    pending.replace(path)


def load(path, default=None):
    return json.loads(path.read_text()) if path.is_file() else default


def safe(value):
    if not isinstance(value, str) or not re.fullmatch(r'[A-Za-z0-9_.-]+', value) or value in {'.', '..'}:
        raise ValueError('Invalid recorded path component: '+repr(value))
    return value


def api(base, action, **query):
    url = base.rstrip('/')+'/api/'+action
    if query:
        url += '?'+urlencode(query)
    try:
        with build_opener(ProxyHandler({})).open(url, timeout=90) as response:
            value = json.load(response)
    except HTTPError as error:
        body = error.read().decode(errors='replace')
        try:
            body = json.loads(body).get('error', body)
        except ValueError:
            pass
        raise RuntimeError(f'{action}: {body}') from error
    if value.get('error'):
        raise ValueError(value['error'])
    return value


def fingerprint(job):
    # Exclude ETA, process liveness and display fields that change each poll.
    return sha(encoded({k: job.get(k) for k in ('run', 'arm', 'status', 'step', 'max_steps',
        'source_digest', 'training_complete', 'checkpoint_interval', 'stop_after_step', 'resume')}))[:20]


def remote_snapshot(device, job):
    request = dict(output=job['output'], arm=safe(job['arm']), source=job.get('source'))
    script = 'REQUEST='+repr(request)+'\n'+(ROOT/'scripts/training_archive_remote.py').read_text()
    data = subprocess.check_output(['ssh', '-o', 'BatchMode=yes', '-o', 'ConnectTimeout=5',
        device['host'], 'python3 -'], input=script.encode(), timeout=240)
    packet = json.loads(gzip.decompress(data))
    files = {name: base64.b64decode(content) for name, content in packet['files'].items()}
    for name, data in files.items():
        path = Path(name)
        if path.is_absolute() or '..' in path.parts:
            raise ValueError('Invalid collector path: '+name)
        if sha(data) != packet['origins'][name]['sha256']:
            raise ValueError('Collector checksum mismatch: '+name)
    manifest = json.loads(files.get('source/web_source_manifest.json', b'{}'))
    if manifest.get('digest') and job.get('source_digest') and manifest['digest'] != job['source_digest']:
        raise ValueError('Source snapshot does not match workbench source_digest')
    for name, expected in manifest.get('files', {}).items():
        if 'source/'+name in files and sha(files['source/'+name]) != expected:
            raise ValueError('Source file does not match frozen manifest: '+name)
    return files, packet['origins']


def numeric(value):
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value)


def statistics(data, parameters):
    rows, incomplete = [], 0
    lines = data.splitlines()
    for i, line in enumerate(lines):
        if not line.strip():
            continue
        try:
            row = json.loads(line)
        except ValueError:
            if i == len(lines)-1 and not data.endswith(b'\n'):
                incomplete += 1
                continue
            raise ValueError(f'Malformed metric record at line {i+1}')
        if row.get('kind') in (None, 'train') and isinstance(row.get('step'), int):
            rows.append(row)
    def loss(row):
        if 'loss' in row:
            return row['loss']
        if parameters.get('recipe') == 'native_random_v2' and numeric(row.get('fm')):
            return row['fm']
        weight = parameters.get('state_weight')
        if numeric(row.get('fm')) and numeric(row.get('state')) and numeric(weight):
            return row['fm']+weight*row['state']
        return None
    def mean(values):
        finite = [v for v in values if numeric(v)]
        return sum(finite)/len(finite) if finite else None
    steps = [r['step'] for r in rows]
    unique = sorted(set(steps))
    values = [loss(r) for r in rows]
    fields = defaultdict(list)
    epochs = defaultdict(list)
    for row in rows:
        epochs[str(row.get('epoch', 'unrecorded'))].append(loss(row))
        for key, value in row.items():
            if key in {'loss', 'fm', 'state', 'seconds', 'learning_rate', 'lr', 'gradient_norm',
                       'gradient_norm_after_clip', 'grad_norm', 'peak_allocated_gib', 'decisions_per_second_end_to_end'}:
                fields[key].append(value)
            elif isinstance(value, dict) and ('grad' in key or key in ('body_flow_mse', 'part_loss')):
                for sub, v in value.items():
                    fields[key+'.'+sub].append(v)
    coverage = {}
    for key, field in sorted(fields.items()):
        finite = [v for v in field if numeric(v)]
        coverage[key] = dict(finite=len(finite), zero=sum(v == 0 for v in finite),
            missing_or_nonfinite=len(rows)-len(finite), mean=mean(finite),
            minimum=min(finite) if finite else None, maximum=max(finite) if finite else None)
    return dict(records=len(rows), unique_steps=len(unique), first_step=min(steps) if steps else None,
        last_step=max(steps) if steps else None, duplicate_steps=len(steps)-len(unique),
        nonincreasing_transitions=sum(b <= a for a,b in zip(steps, steps[1:])),
        interior_missing_steps=sum(max(0,b-a-1) for a,b in zip(unique, unique[1:])),
        incomplete_tail_records=incomplete, finite_loss=sum(numeric(v) for v in values),
        loss_first100=mean(values[:100]), loss_last100=mean(values[-100:]),
        loss_min=min((v for v in values if numeric(v)), default=None),
        epochs={key:dict(records=len(v), mean_loss=mean(v)) for key,v in epochs.items()}, fields=coverage)


def archive(root, device_id, device, job, detail):
    run, arm = safe(job['run']), safe(job['arm'])
    directory = root/REPORT/'runs'/safe(device_id)/run/arm
    files, origins = remote_snapshot(device, job)
    parameters = (detail.get('training_parameters') or {}).get('values', {})
    metric = files.pop('metrics.jsonl', b'')
    metric_origin = origins.pop('metrics.jsonl', None)
    stats = statistics(metric, parameters)
    event_id = fingerprint(job)+'-'+sha(encoded(origins))[:12]
    event = directory/'events'/event_id
    payload = dict(job=job, detail=detail, statistics=stats)
    for name in ('curve', 'loss_series', 'metrics','diagnostic_series','analysis_report'):
        payload['detail'] = {k:v for k,v in payload['detail'].items() if k != name}
    files['workbench.json'] = encoded(payload)
    origins['workbench.json'] = dict(path='workbench training/jobs and training/detail', selection='JSON snapshot')
    parts = []
    lines = metric.splitlines(keepends=True)
    for offset in range(0, len(lines), 1000):
        data = b''.join(lines[offset:offset+1000])
        name = 'raw_metrics/'+sha(data)+'.jsonl'
        write(directory/name, data)
        parts.append(dict(path=name, bytes=len(data), sha256=sha(data), first_line=offset+1,
                          lines=min(1000, len(lines)-offset)))
    existing = load(event/'MANIFEST.json', {})
    manifest = dict(schema=1, device=device_id, run=run, arm=arm, event=event_id,
        captured_at=existing.get('captured_at', datetime.now(timezone.utc).isoformat()), metrics=dict(origin=metric_origin,
        bytes=len(metric), sha256=sha(metric), parts=parts), files={})
    for name, data in sorted(files.items()):
        write(event/name, data)
        manifest['files'][name] = dict(origin=origins[name], bytes=len(data), sha256=sha(data))
    write(event/'MANIFEST.json', encoded(manifest))
    latest = dict(event=event_id, run=run, arm=arm, device=device_id, job=job,
        parameters=parameters, parameter_sources=(detail.get('training_parameters') or {}).get('sources', []),
        initial_weights=detail.get('initial_weights'), statistics=stats,
        metrics_recorded=metric_origin is not None, checkpoints=detail.get('checkpoints', []))
    write(directory/'latest.json', encoded(latest))


def evaluations(root, jobs=()):
    """Archive terminal evaluations independently of training archive coverage."""
    known = defaultdict(list)
    for path in (root/REPORT/'runs').glob('*/*/*/latest.json'):
        known[load(path)['run']].append(path.parent)
    for path in sorted((root/'evaluation').glob('V_*/**/results.json')):
        if '.runtime' in path.parts:
            continue
        result_data = path.read_bytes()
        value = json.loads(result_data)
        job = next((j for j in jobs if j.get('output') == str(path.parent)), {})
        # A resumed evaluation may still have its previous interrupted summary.
        # Defer a new snapshot when the live workbench says it is active.
        if job.get('state') and job['state'] not in EVALUATION_TERMINAL:
            continue
        if value.get('status') not in EVALUATION_TERMINAL and job.get('state') not in EVALUATION_TERMINAL:
            continue
        identity = value.get('policy_identity') or value.get('configuration', {}).get('policy_identity') or {}
        checkpoint = identity.get('checkpoint')
        key = sha(str(path.relative_to(root)).encode())[:16]
        episode = path.parent/'episodes.jsonl'
        episode_data = episode.read_bytes() if episode.is_file() else None
        origin = dict(path=str(path.relative_to(root)), checkpoint=checkpoint,
            training_id=identity.get('training_id'),
            workbench_state=job.get('state'), workbench_job_id=job.get('id'),
            results_sha256=sha(result_data), episodes_sha256=sha(episode_data) if episode_data is not None else None)
        # Keep existing per-run links, and always publish an independent entry.
        destinations = [root/REPORT/'evaluations'/key] + [
            directory/'evaluations'/key for directory in known.get(identity.get('training_id'), [])]
        for dest in destinations:
            write(dest/'results.json', result_data)
            if episode_data is not None:
                write(dest/'episodes.jsonl', episode_data)
            write(dest/'origin.json', encoded(origin))


def fmt(value):
    if value is None:
        return '未记录'
    if isinstance(value, float):
        return f'{value:.8g}'
    if isinstance(value, (list, dict)):
        value = json.dumps(value, ensure_ascii=False)
    return str(value).replace('|', '\\|').replace('\n', ' ')



def render_run_report(root, directory, record):
    """Self-contained deterministic report. Never infer missing metrics."""
    event = directory/'events'/record['event']
    manifest = load(event/'MANIFEST.json', {})
    payload = load(event/'workbench.json', {})
    detail = payload.get('detail', {})
    job, params, stats = record['job'], record['parameters'], record['statistics']
    rec = event/'recorded/run'
    runtime = load(rec/'native_runtime.json', {})
    prepared = load(rec/'prepared/PREPARED.json', {})
    contract = load(rec/'native_contract.json', {})
    raw = b''.join((directory/p['path']).read_bytes()
                   for p in manifest.get('metrics', {}).get('parts', []))
    rows = []
    for line in raw.splitlines():
        try:
            row = json.loads(line)
        except ValueError:
            continue  # collector already checks interior corruption
        if row.get('kind') in (None, 'train') and isinstance(row.get('step'), int):
            rows.append(row)
    def val(row):
        if numeric(row.get('loss')):
            return row['loss']
        if params.get('recipe') == 'native_random_v2':
            return row.get('fm')
        if numeric(row.get('fm')) and numeric(row.get('state')) and numeric(params.get('state_weight')):
            return row['fm'] + params['state_weight']*row['state']
        return None
    def avg(values):
        values = [v for v in values if numeric(v)]
        return sum(values)/len(values) if values else None
    lines = ['# 训练分析报告', '', '由工作台归档程序生成；不含 AI 推断。缺失字段为未记录，不以零替代。', '',
             '## 1. 实验身份与权重', '',
             '| 字段 | 值 |', '| --- | --- |']
    def item(k,v):
        lines.append(f'| {fmt(k)} | {fmt(v)} |')
    for k,v in [('run',record['run']),('device',record['device']),('arm',record['arm']),
                ('status',job.get('status')),('step',job.get('step')),('目标步数',job.get('max_steps')),
                ('归档事件',record['event']),('归档时间',manifest.get('captured_at')),
                ('训练备注',job.get('notes')),('源码摘要',job.get('source_digest')),
                ('初始权重/续训',record.get('initial_weights'))]:
        item(k,v)
    lines += ['', '## 2. 实际训练配置', '', '配置来源：'+fmt(record.get('parameter_sources')), '',
              '| 参数 | 值 |','| --- | --- |']
    for k,v in sorted(params.items()):
        if k not in {'argv','env','source_manifest'}:
            item(k,v)
    lines += ['', '## 3. 实际运行与数据', '', '| 字段 | 值 |','| --- | --- |']
    for k,v in sorted(runtime.items()): item(k,v)
    for k in ('mode','train_frames','val_frames','train_valid_anchors','accepted_cells','hashes'):
        item('prepared.'+k, prepared.get(k))
    item('split',contract.get('split'))
    item('初始权重分片 SHA-256',contract.get('base_weights'))
    item('统计量来源',contract.get('statistics'))
    lines += ['', '## 4. 训练指标', '',
              '训练 Loss 为记录点/日志区间值；验证 Loss 单独列出。不同动作归一化、mask、噪声采样及数据划分下的 Loss 不宜直接比较。',
              '首尾统计为最多100条日志记录；日志间隔可能大于1步，步号间隔不等于丢失训练更新。', '',
              '| 字段 | 值 |','| --- | --- |']
    for k in ('records','first_step','last_step','duplicate_steps','nonincreasing_transitions',
              'interior_missing_steps','incomplete_tail_records','finite_loss','loss_first100','loss_last100','loss_min'):
        item(k,stats.get(k))
    lines += ['', '### 每1000 optimizer steps 的训练记录均值', '',
              '| 步数范围 | 记录数 | Loss | LR首值 | LR末值 | 梯度范数 | 底盘 | 躯干 | 左臂 | 右臂 | 左夹爪 | 右夹爪 |',
              '| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |']
    groups = defaultdict(list)
    for row in rows:
        groups[max(0,row['step']-1)//1000].append(row)
    for k,group in sorted(groups.items()):
        lr = [r.get('lr',r.get('learning_rate')) for r in group if numeric(r.get('lr',r.get('learning_rate')))]
        values = [f'{k*1000+1}–{(k+1)*1000}',len(group),avg(val(r) for r in group),
                  lr[0] if lr else None,lr[-1] if lr else None,
                  avg(r.get('grad_norm',r.get('gradient_norm')) for r in group)]
        for name in ('base','torso','left_arm','right_arm','left_gripper','right_gripper'):
            values.append(avg((r.get('part_loss') or r.get('body_flow_mse') or {}).get(name) for r in group))
        lines.append('| '+' | '.join(fmt(v) for v in values)+' |')
    lines += ['', '### 按实际 epoch 字段汇总', '', '| epoch | 记录数 | Loss |','| --- | ---: | ---: |']
    for k,v in stats.get('epochs',{}).items():
        lines.append('| '+' | '.join(fmt(x) for x in (k,v.get('records'),v.get('mean_loss')))+' |')
    lines += ['', '### 验证指标（独立于训练记录）', '', '| step | FM Loss | 样本数 |','| ---: | ---: | ---: |']
    validation = {r.get('step'):r for r in (detail.get('validation') or []) if isinstance(r,dict)}
    for p in sorted(rec.glob('validation-*.json')):
        v = load(p,{})
        validation[v.get('step')] = v
    for step,v in sorted(validation.items(),key=lambda kv:str(kv[0]).zfill(12)):
        lines.append('| '+' | '.join(fmt(x) for x in (step,v.get('fm',v.get('validation_loss')),v.get('samples')))+' |')
    if not validation: lines.append('| 未记录 | 未记录 | 未记录 |')
    lines += ['', '### 指标覆盖', '', '| 字段 | 有限记录 | 缺失/非有限 | 均值 | 最小 | 最大 |',
              '| --- | ---: | ---: | ---: | ---: | ---: |']
    for k,v in stats.get('fields',{}).items():
        lines.append('| '+' | '.join(fmt(x) for x in (k,v['finite'],v['missing_or_nonfinite'],v['mean'],v['minimum'],v['maximum']))+' |')
    lines += ['', '## 5. Checkpoint', '', '| step | 路径 |','| ---: | --- |']
    for c in record.get('checkpoints',[]):
        lines.append('| '+' | '.join(fmt(x) for x in (c.get('step'),c.get('path')))+' |')
    lines += ['', '## 6. 闭环评估（仅明确绑定本 run 的终态记录）', '']
    found = False
    for p in sorted((directory/'evaluations').glob('*/results.json')):
        v = load(p,{}); identity = v.get('policy_identity') or v.get('configuration',{}).get('policy_identity') or {}
        if identity.get('training_id') != record['run'] or not identity.get('checkpoint'):
            continue
        found = True
        origin = load(p.parent/'origin.json',{})
        lines += ['### '+fmt(origin.get('path')), '', '| 字段 | 值 |','| --- | --- |']
        for k,x in [('checkpoint',identity['checkpoint']),('状态',v.get('status')),
                    ('完成/计划',[v.get('completed_episodes'),v.get('requested_episodes')]),
                    ('成功/有效',[v.get('successful_episodes'),v.get('official_episodes')]),
                    ('Q Score',v.get('mean_q_score')),('Task Progress',v.get('mean_task_progress')),
                    ('实际评估配置',v.get('configuration'))]: item(k,x)
        ep = p.parent/'episodes.jsonl'
        if ep.is_file():
            lines += ['', '| 回合 | instance | seed | 成功 | Q Score | Task Progress | 终止 |',
                      '| --- | --- | --- | --- | --- | --- | --- |']
            for line in ep.read_text().splitlines():
                try: r=json.loads(line)
                except ValueError: continue
                lines.append('| '+' | '.join(fmt(r.get(k)) for k in
                    ('episode_index','instance_id','seed','success','q_score','task_progress','native_termination'))+' |')
        lines += ['']
    if not found: lines.append('尚未归档匹配的终态评估。训练结束不等于评估完成。')
    lines += ['', '## 7. 证据与局限', '',
              f'- [归档清单](events/{record["event"]}/MANIFEST.json)',
              f'- [工作台原始快照](events/{record["event"]}/workbench.json)',
              '- 原始指标按清单中的 SHA-256 分片保存；日志只保存已声明范围。未记录的指标不作推断。',
              '- 归一化审计、实际 processor 配置、源码、终态评估及逐回合数据以归档证据为准。']
    for name in sorted(manifest.get('files',{})):
        if any(x in name for x in ('normalization_audit','relative_stats','native_runtime','native_contract','validation-','logs/')):
            lines.append(f'- [{name}](events/{record["event"]}/{name})')
    content = ('\n'.join(lines)+'\n').encode()
    write(directory/'training_report.md',content)
    # Immutable report revisions are content-addressed; no clock-driven churn.
    write(directory/'report_revisions'/(sha(content)+'.md'),content)


def render(root):
    lines = ['# 训练与评估自动汇总', '',
        '由 `scripts/training_archive.py` 按固定规则生成，无 LLM 参与。原始数据及 SHA-256 清单见各事件链接。', '',
        '首次安装只补录指定历史 run；随后自动记录工作台新训练及已有训练的完成、暂停、失败、取消事件。',
        '训练显示最近一次归档终态；训练恢复后，等下一个终态再更新。评估独立收录完成、中断、失败和取消的结果，不要求训练已归档。运行中的评估不计作终态。', '',
        'Loss 为训练记录中的 loss；V2.1 为 fm + 配方 state_weight × state。首尾均值各取最多100条记录，'
        '保留重复 step 并报告数量；缺失/非有限值不当作零。epoch 均值不代表完整 epoch。', '',
        '| 设备 / run / arm | 归档状态 | optimizer step / 当前停止目标 | 首100 Loss | 末100 Loss | 原始记录数 |',
        '| --- | --- | --- | --- | --- | --- |']
    records = [(p,load(p)) for p in sorted((root/REPORT/'runs').glob('*/*/*/latest.json'))]
    for path, record in records:
        j,s = record['job'],record['statistics']
        render_run_report(root, path.parent, record)
        rel = path.parent.relative_to(root/REPORT).as_posix()
        lines.append(f"| [{fmt(record['device']+'/'+record['run']+'/'+record['arm'])}]({rel}/training_report.md) | {fmt(j.get('status'))} | {fmt(j.get('step'))} / {fmt(j.get('max_steps'))} | {fmt(s['loss_first100'])} | {fmt(s['loss_last100'])} | {s['records']} |")
    lines += ['', '## 闭环评估总表', '',
        '覆盖本机 `evaluation/V_*/` 下的终态结果，包括历史训练和身份未完整记录的评估。未记录字段保留为空缺，不推断归属。中断作业的已完成回合与计划回合分别列出。', '',
        '| 评估 / 原始数据 | 训练 run | checkpoint | 结果状态 / 工作台状态 | job ID | 完成 / 计划回合 | 成功 / 有效回合 | 平均任务进度（原值） |',
        '| --- | --- | --- | --- | --- | --- | --- | --- |']
    for result in sorted((root/REPORT/'evaluations').glob('*/results.json')):
        value = load(result)
        origin = load(result.parent/'origin.json', {})
        identity = value.get('policy_identity') or value.get('configuration', {}).get('policy_identity') or {}
        link = result.relative_to(root/REPORT).as_posix()
        lines.append(f"| [{fmt(origin.get('path'))}]({link}) | {fmt(identity.get('training_id'))} | {fmt(identity.get('checkpoint'))} | {fmt(value.get('status'))} / {fmt(origin.get('workbench_state'))} | {fmt(origin.get('workbench_job_id'))} | {fmt(value.get('completed_episodes'))} / {fmt(value.get('requested_episodes'))} | {fmt(value.get('successful_episodes'))} / {fmt(value.get('official_episodes'))} | {fmt(value.get('mean_task_progress'))} |")
    for path, record in records:
        j,s = record['job'],record['statistics']
        rel = path.parent.relative_to(root/REPORT).as_posix()
        event = rel+'/events/'+record['event']
        lines += ['', '## '+record['run']+' / '+record['arm']+' / '+record['device'], '',
            f"证据：[清单]({event}/MANIFEST.json)、[实际配方与统计]({event}/workbench.json)、[最近终态]({rel}/latest.json)。",
            '', f"原始输出：`{j.get('output')}`；源码：`{j.get('source')}`；源码 digest：`{j.get('source_digest')}`。", '',
            '身份字段（直接沿用工作台，不补猜历史编号）：'+fmt({k:j.get(k) for k in ('model_version','test_id','experiment_id','training_id','run')}), '',
            '初始权重 / 恢复：'+fmt(record['initial_weights']), '',
            f"原始指标：{'已归档' if record['metrics_recorded'] else '未找到'}；step 范围 {s['first_step']}–{s['last_step']}；重复 step {s['duplicate_steps']}；非递增转换 {s['nonincreasing_transitions']}；范围内缺步 {s['interior_missing_steps']}；不完整尾行 {s['incomplete_tail_records']}。", '',
            '预算记录：当前停止目标 '+fmt(j.get('max_steps'))+'；原始配置 steps '+fmt(record['parameters'].get('original_steps',record['parameters'].get('steps')))+'。终态 completed 只描述本次工作台目标。', '',
            '### 实际记录配方', '', '字段来源：'+fmt(record['parameter_sources']), '', '| 参数 | 值 |', '| --- | --- |']
        for key,value in sorted(record['parameters'].items()):
            if key not in {'source_manifest', 'argv', 'env'}:
                lines.append(f'| {fmt(key)} | {fmt(value)} |')
        lines += ['', '### 指标覆盖', '', '| 字段 | 有限记录 | 零值记录 | 缺失或非有限 | 均值 | 最小 / 最大 |', '| --- | --- | --- | --- | --- | --- |']
        for key,value in s['fields'].items():
            lines.append(f"| {fmt(key)} | {value['finite']} | {value['zero']} | {value['missing_or_nonfinite']} | {fmt(value['mean'])} | {fmt(value['minimum'])} / {fmt(value['maximum'])} |")
        lines += ['', '### 按原始 epoch 字段汇总', '', '| epoch 原值 | 记录数 | 有限 Loss 均值 |', '| --- | --- | --- |']
        for key,value in sorted(s['epochs'].items(), key=lambda item: (not item[0].isdigit(), int(item[0]) if item[0].isdigit() else item[0])):
            lines.append(f"| {fmt(key)} | {value['records']} | {fmt(value['mean_loss'])} |")
        lines += ['', '### 工作台列出的 checkpoint', '',
                  '仅列元数据；权重与 optimizer tensor 留在训练设备。完整性标记见事件清单中的原始文件。', '',
                  '| optimizer step | 原始路径 |', '| --- | --- |']
        for checkpoint in record['checkpoints']:
            lines.append(f"| {fmt(checkpoint.get('step'))} | {fmt(checkpoint.get('path'))} |")
        lines += ['', '### 闭环评估', '']
        evals = sorted((path.parent/'evaluations').glob('*/results.json'))
        if not evals:
            lines += ['尚未归档已结束且明确绑定本 run / checkpoint 的评估。']
        for result in evals:
            value = load(result)
            origin = load(result.parent/'origin.json', {})
            link = result.relative_to(root/REPORT).as_posix()
            identity = value.get('policy_identity') or value.get('configuration', {}).get('policy_identity') or {}
            lines += [f"- [原始结果]({link})：结果文件状态 {fmt(value.get('status'))}，工作台状态 {fmt(origin.get('workbench_state'))}；完成 {fmt(value.get('completed_episodes'))}/{fmt(value.get('requested_episodes'))}；成功 {fmt(value.get('successful_episodes'))}/{fmt(value.get('official_episodes'))}（分母为原始 official_episodes 字段）；checkpoint `{identity.get('checkpoint')}`。"]
    write(root/REPORT/'README.md', ('\n'.join(lines)+'\n').encode())


def git(root, *args):
    try:
        return subprocess.check_output(['git', '-C', str(root), *args], stderr=subprocess.STDOUT, timeout=120).decode().strip()
    except subprocess.CalledProcessError as error:
        raise RuntimeError(error.output.decode(errors='replace').strip()) from error


def publish(root):
    if git(root, 'branch', '--show-current') != 'main':
        raise ValueError('Automatic archive publishing requires main')
    # commit --only preserves unrelated staged and working-tree changes.
    git(root, 'add', '--', str(REPORT))
    if git(root, 'diff', '--cached', '--name-only', '--', str(REPORT)):
        git(root, 'commit', '--only', '-m', 'reports: automatically archive training evidence', '--', str(REPORT))
    git(root, 'push', 'origin', 'HEAD:main')  # No force, pull, reset or rebase.


def verify(root):
    count = 0
    for path in sorted((root/REPORT).glob('runs/*/*/*/events/*/MANIFEST.json')):
        value = load(path)
        for name, expected in value['files'].items():
            data = (path.parent/name).read_bytes()
            if len(data) != expected['bytes'] or sha(data) != expected['sha256']:
                raise ValueError('Snapshot checksum mismatch: '+str(path.parent/name))
            count += 1
        metric = b''
        for part in value['metrics']['parts']:
            data = (path.parents[2]/part['path']).read_bytes()
            if sha(data) != part['sha256'] or len(data) != part['bytes']:
                raise ValueError('Metric part checksum mismatch: '+part['path'])
            metric += data
        if sha(metric) != value['metrics']['sha256'] or len(metric) != value['metrics']['bytes']:
            raise ValueError('Original metric reconstruction mismatch: '+str(path))
        origin = value['metrics']['origin']
        if origin and sha(metric) != origin['sha256']:
            raise ValueError('Original remote metric checksum mismatch: '+str(path))
        source_manifest = load(path.parent/'source/web_source_manifest.json', {})
        for name, expected in source_manifest.get('files', {}).items():
            source = path.parent/'source'/name
            if source.is_file() and sha(source.read_bytes()) != expected:
                raise ValueError('Recorded source digest mismatch: '+str(source))
    eval_origins = list((root/REPORT).glob('runs/*/*/*/evaluations/*/origin.json'))
    eval_origins += list((root/REPORT/'evaluations').glob('*/origin.json'))
    for path in eval_origins:
        expected = load(path)
        for name in ('results', 'episodes'):
            digest = expected.get(name+'_sha256')
            if digest and sha((path.parent/(name+('.json' if name == 'results' else '.jsonl'))).read_bytes()) != digest:
                raise ValueError('Evaluation checksum mismatch: '+str(path))
    print(json.dumps(dict(verified_payload_files=count, verified_evaluations=len(eval_origins))))


def scan(args):
    root = args.root.resolve()
    state_path = root/STATE
    state_path.parent.mkdir(parents=True, exist_ok=True)
    with state_path.with_suffix('.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        state = load(state_path, dict(jobs={}, initialized=False))
        devices = load(root/'configs/workbench_devices.json', {})
        errors, archived, scanned_devices = {}, [], []
        for device_id in args.devices:
            try:
                jobs = api(args.workbench, 'training/jobs', device=device_id)['jobs']
            except Exception as error:
                errors[device_id] = str(error)
                continue
            scanned_devices.append(device_id)
            for job in jobs:
                key = device_id+'/'+safe(job['run'])+'/'+safe(job['arm'])
                previous = state['jobs'].get(key)
                current = fingerprint(job)
                terminal = job.get('status') in TERMINAL
                # Only the installation scan baselines history. A device that
                # reconnects later must backfill, so jobs finished offline count.
                first_device_scan = not state['initialized']
                requested_backfill = job['run'] in args.backfill and not (root/REPORT/'runs'/device_id/job['run']/job['arm']/'latest.json').exists()
                requested_refresh = job['run'] in getattr(args,'refresh_run',[])
                if terminal and (previous != current or requested_backfill or requested_refresh) and (not first_device_scan or requested_backfill or requested_refresh):
                    try:
                        detail = api(args.workbench, 'training/detail', device=device_id, run=job['run'], arm=job['arm'])
                        archive(root, device_id, devices[device_id], job, detail)
                        archived.append(key)
                    except Exception as error:
                        errors[key] = repr(error)
                        continue  # retry; never acknowledge a failed export
                state['jobs'][key] = current
            state.setdefault('devices_initialized', [])
            if device_id not in state['devices_initialized']:
                state['devices_initialized'].append(device_id)
        try:
            eval_jobs = []
            try:
                eval_jobs = api(args.workbench, 'launch/jobs')['jobs']
            except Exception as error:
                errors['evaluation_jobs'] = repr(error)
            evaluations(root, eval_jobs)
            render(root)
            if args.publish:
                publish(root)
        except Exception as error:
            errors['report_or_publish'] = repr(error)
        now = datetime.now(timezone.utc)
        state.update(initialized=True, checked_at=now.isoformat(), errors=errors,
                     status='degraded' if errors else 'ok', scanned_devices=scanned_devices,
                     next_check_at=(now+timedelta(seconds=300)).isoformat(),
                     next_check_reason='systemd 300s: training terminal events, finished evaluations, Git publishing retry',
                     archived_last_scan=archived, publish_enabled=args.publish)
        write(state_path, encoded(state))
        print(json.dumps(dict(archived=archived, errors=errors), ensure_ascii=False))
        # An unavailable secondary device is recorded as degraded; it must not
        # mark a successful archive/push for the reachable device as failed.
        critical = any(key not in args.devices for key in errors)
        return 1 if critical or not scanned_devices else 0


def main():
    parser = argparse.ArgumentParser(__doc__)
    action = parser.add_mutually_exclusive_group(required=True)
    action.add_argument('--once', action='store_true')
    action.add_argument('--verify-only', action='store_true')
    parser.add_argument('--root', type=Path, default=ROOT)
    parser.add_argument('--workbench', default='http://127.0.0.1:8765')
    parser.add_argument('--devices', nargs='+', default=['a100', 'a6000'])
    parser.add_argument('--backfill', action='append', default=[])
    parser.add_argument('--refresh-run', action='append', default=[], help='Recollect an explicitly selected terminal run with new evidence fields')
    parser.add_argument('--publish', action='store_true')
    args = parser.parse_args()
    if args.verify_only:
        verify(args.root.resolve())
        return 0
    return scan(args)


if __name__ == '__main__':
    raise SystemExit(main())
