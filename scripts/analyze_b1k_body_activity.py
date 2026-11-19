"""Measure observed body activity in the 600 local Task 0/34/40 demonstrations.

Run with env/b1k2025-policy/bin/python scripts/analyze_b1k_body_activity.py.
Only analysis products are written; source demonstrations are read unchanged.
"""

import argparse
import csv
import json
from pathlib import Path
import xml.etree.ElementTree as ET

import numpy as np
import pyarrow.parquet as pq
from scipy.spatial.transform import Rotation


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_URDF = Path('/nvme3/projects/SERF-VLA/datasets/omnigibson-robot-assets/models/r1pro/urdf/r1pro.urdf')
LABELS = {
    'base': '底盘（平移或转向）',
    'base_translation': '底盘平移',
    'base_rotation': '底盘转向',
    'torso': '躯干机构（任一关节）',
    'lift': '有效升降（躯干顶端高度）',
    'left_arm': '左臂',
    'right_arm': '右臂',
    'head_relative': '头部相机相对底盘转向（躯干带动）',
    'head_world': '头部相机世界系转向（身体带动）',
}
THRESHOLDS = {
    'base_translation': 0.01,  # m/s
    'base_rotation': 0.02,  # rad/s
    'torso': 0.05,  # maximum individual joint speed, rad/s
    'lift': 0.005,  # m/s
    'left_arm': 0.05,  # maximum individual joint speed, rad/s
    'right_arm': 0.05,
    'head_relative': 0.02,  # rad/s, SO(3) angular distance
    'head_world': 0.02,
}


def runs(mask):
    edges = np.flatnonzero(np.diff(np.r_[False, mask, False]))
    return zip(edges[::2], edges[1::2])


def sustained(mask, dt, minimum=0.1):
    result = mask.copy()
    for start, stop in runs(mask):
        if dt[start:stop].sum() < minimum - 1e-8:
            result[start:stop] = False
    return result


def column_matrix(table, name, width):
    column = table[name].combine_chunks()
    lengths = np.diff(column.offsets.to_numpy())
    assert np.all(lengths == width), (name, np.unique(lengths))
    return column.values.to_numpy().reshape(len(column), width)


def torso_chain(urdf):
    model = ET.parse(urdf).getroot()
    joints = {j.attrib['name']: j for j in model.findall('joint')}
    assert joints['zed_joint'].attrib['type'] == 'fixed'
    assert joints['zed_joint'].find('parent').attrib['link'] == 'torso_link4'
    chain = []
    for index in range(1, 5):
        j = joints[f'torso_joint{index}']
        origin = j.find('origin')
        xyz = np.fromstring(origin.attrib['xyz'], sep=' ')
        rpy = np.fromstring(origin.attrib.get('rpy', '0 0 0'), sep=' ')
        axis = np.fromstring(j.find('axis').attrib['xyz'], sep=' ')
        chain.append((xyz, Rotation.from_euler('xyz', rpy).as_matrix(), axis))
    return chain


def torso_top_height(qpos, chain):
    rotation = np.broadcast_to(np.eye(3), (len(qpos), 3, 3)).copy()
    position = np.zeros((len(qpos), 3))
    for index, (xyz, origin_rotation, axis) in enumerate(chain):
        position += np.einsum('nij,j->ni', rotation, xyz)
        joint_rotation = Rotation.from_rotvec(qpos[:, index, None] * axis).as_matrix()
        rotation = rotation @ origin_rotation @ joint_rotation
    return position[:, 2]


def speed_signals(state, camera, timestamp, chain):
    # A centered six-interval (0.2 s) secant suppresses simulation jitter.
    # Endpoints use the available shorter window; no episode boundaries are crossed.
    index = np.arange(len(state))
    left = np.maximum(index - 3, 0)
    right = np.minimum(index + 3, len(state) - 1)
    elapsed = timestamp[right] - timestamp[left]

    def derivative(value):
        return (value[right] - value[left]) / elapsed.reshape((-1,) + (1,) * (value.ndim - 1))

    def angular_speed(rotation):
        return (rotation[left].inv() * rotation[right]).magnitude() / elapsed

    head = Rotation.from_quat(camera[:, 17:21])
    base_rotation = Rotation.from_euler('xyz', np.arctan2(state[:, 146:149], state[:, 143:146]))
    signals = {
        'base_translation': np.linalg.norm(derivative(state[:, 140:142]), axis=1),
        'base_rotation': np.abs(derivative(np.unwrap(state[:, 149]))),
        'torso': np.max(np.abs(derivative(np.unwrap(state[:, 236:240], axis=0))), axis=1),
        'lift': np.abs(derivative(torso_top_height(state[:, 236:240], chain))),
        'left_arm': np.max(np.abs(derivative(np.unwrap(state[:, 158:165], axis=0))), axis=1),
        'right_arm': np.max(np.abs(derivative(np.unwrap(state[:, 197:204], axis=0))), axis=1),
        'head_relative': angular_speed(head),
        'head_world': angular_speed(base_rotation * head),
    }
    # Allocate each estimate to the interval between two observed timestamps.
    return {key: (value[:-1] + value[1:]) / 2 for key, value in signals.items()}


def gripper_events(action, state, timestamp, task, episode):
    records = []
    for side, action_index, state_start in [('left', 14, 193), ('right', 22, 232)]:
        command = action[:, action_index] > 0
        changes = np.r_[0, np.flatnonzero(command[1:] != command[:-1]) + 1, len(command)]
        accepted = [(0, bool(command[0]))]
        for start, stop in zip(changes[1:-1], changes[2:]):
            value = bool(command[start])
            if stop - start >= 3 and value != accepted[-1][1]:
                accepted.append((int(start), value))
        aperture = state[:, state_start:state_start + 2].mean(axis=1)
        for event_index in range(1, len(accepted)):
            start, opening = accepted[event_index]
            stop = accepted[event_index + 1][0] if event_index + 1 < len(accepted) else len(command)
            baseline = float(np.median(aperture[max(0, start - 3):start]))
            displacement = (aperture[start:stop] - baseline) * (1 if opening else -1)
            enough = displacement >= 0.005
            effective = any(b - a >= 3 for a, b in runs(enough))
            records.append(dict(task=task, episode_index=episode, side=side,
                                direction='open' if opening else 'close',
                                timestamp_s=float(timestamp[start]), effective=effective,
                                peak_mean_finger_displacement_m=float(displacement.max())))
    return records


def summarize(rows, events, task):
    selected = rows if task == 'all' else [r for r in rows if r['task'] == task]
    selected_events = events if task == 'all' else [e for e in events if e['task'] == task]
    duration = sum(r['duration_s'] for r in selected)
    result = dict(task=task, episodes=len(selected), frames=sum(r['frames'] for r in selected),
                  duration_s=duration, sampled_duration_s=sum(r['frames'] / 30 for r in selected),
                  activity={}, grippers={})
    for part in LABELS:
        seconds = sum(r[f'{part}_active_s'] for r in selected)
        ratios = [r[f'{part}_active_s'] / r['duration_s'] * 100 for r in selected]
        result['activity'][part] = dict(active_s=seconds, active_percent=seconds / duration * 100,
                                        per_episode_mean_percent=float(np.mean(ratios)))
    for side in ('left', 'right'):
        e = [e for e in selected_events if e['side'] == side]
        result['grippers'][side] = dict(
            command_events=len(e), effective_events=sum(x['effective'] for x in e),
            effective_open=sum(x['effective'] and x['direction'] == 'open' for x in e),
            effective_close=sum(x['effective'] and x['direction'] == 'close' for x in e))
    result['threshold_sensitivity'] = {}
    for scale in (0.5, 1.0, 2.0):
        result['threshold_sensitivity'][str(scale)] = {
            part: sum(r[f'{part}_scale{scale}_active_s'] for r in selected) / duration * 100
            for part in LABELS}
    return result


def write_csv(path, rows):
    with path.open('w', newline='') as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def report_text(summaries, metadata):
    lines = ['# B1K Task 0 / 34 / 40 身体部件活动统计', '',
             '范围：三个本地 GR00T 数据集清单对应的全部 600 条原始示教，包含训练集和验证集；无抽样。', '',
             '| 范围 | 示教数 | 帧数 | 可观测时长（小时） |',
             '|---|---:|---:|---:|']
    for s in summaries:
        name = '总体' if s['task'] == 'all' else f"Task {s['task']}"
        lines.append(f"| {name} | {s['episodes']} | {s['frames']:,} | {s['duration_s']/3600:.4f} |")
    lines += ['', '## 有效运动时间与占比', '',
              '每格为运动分钟数 / 占该范围可观测时长的百分比。总体按时长加权；部件可同时运动，比例不能相加。', '',
              '| 部件 / 指标 | Task 0 | Task 34 | Task 40 | 总体 |', '|---|---:|---:|---:|---:|']
    for part, label in LABELS.items():
        cells = [f"{s['activity'][part]['active_s']/60:.2f} 分钟 / {s['activity'][part]['active_percent']:.2f}%" for s in summaries]
        lines.append('| ' + ' | '.join([label] + cells) + ' |')
    lines += ['| 头部独立主动转动 | 不适用（无独立关节） | 不适用（无独立关节） | 不适用（无独立关节） | 不适用（无独立关节） |', '',
              '## 夹爪开合事件', '',
              '开和合各算一次事件；不把初始开合状态算作事件，也不把持续夹持帧重复计数。实际有效事件需要指令切换后，双指平均位置沿指令方向变化至少 5 mm，且达到该行程至少连续 3 帧。', '',
              '| 范围 | 左爪有效开 / 合 | 左爪有效合计 / 指令次数 | 右爪有效开 / 合 | 右爪有效合计 / 指令次数 |',
              '|---|---:|---:|---:|---:|']
    for s in summaries:
        name = '总体' if s['task'] == 'all' else f"Task {s['task']}"
        l, r = s['grippers']['left'], s['grippers']['right']
        lines.append(f"| {name} | {l['effective_open']} / {l['effective_close']} | {l['effective_events']} / {l['command_events']} | {r['effective_open']} / {r['effective_close']} | {r['effective_events']} / {r['command_events']} |")
    lines += ['', '## 统计口径', '',
              '- 使用原始 256 维 observation.state、21 维 observation.cam_rel_poses 以及 23 维 action。实际运动来自观测；action 只用于夹爪事件分段。',
              '- 30 Hz。时长使用每条示教最后一个时间戳减第一个时间戳，即 (帧数−1)/30；不推测最后一帧以后的运动。',
              '- 位姿采用 0.2 秒中心差分估速，边界缩短窗口；相邻帧估速平均分配到时间间隔。过滤持续不足 0.1 秒的超阈值片段，不填补静止间隙。',
              '- 底盘：世界系 XY 平移速度 > 0.01 m/s，或偏航角速度 > 0.02 rad/s。',
              '- 左右臂：各自 7 个关节中任一关节速度 > 0.05 rad/s（约 2.86°/s）。关节坐标不计底盘或躯干带动整条手臂的刚体运动。',
              '- 躯干机构：4 个躯干关节中任一速度 > 0.05 rad/s。纯升降另以 URDF 正运动学得到 torso_link4 原点相对底盘的高度，速度 > 0.005 m/s；不是把四个旋转关节误当成直线升降轴。',
              '- 头部：URDF 中 zed_joint 为 fixed，父节点为 torso_link4，没有独立主动自由度。相机相对底盘和世界系的转向分别按 SO(3) 角速度 > 0.02 rad/s 计算，含躯干/底盘带动，不能解释为独立头部动作。',
              '- 相机顺序经本地 OmniGibson eval_utils.py 核对为 left_wrist、right_wrist、head；使用最后 7 维头部相机位姿，四元数为 XYZW。',
              '- 夹爪指令需新状态连续至少 3 帧才接受。有效行程指双指平均位置变化 5 mm，即总开口宽度约变化 10 mm；不足行程的切换仍保留在指令事件数中。',
              '- “有效”是明确阈值下的运动检测口径，不代表任务成功、有效接触或有用动作。', '',
              '## 阈值敏感性', '',
              '阈值同时乘 0.5 / 1 / 2；窗口与 0.1 秒持续条件不变。每格为三档总体运动占比，用于展示慢动作与抖动界限对结果的影响。', '',
              '| 指标 | 0.5× | 1×（主结果） | 2× |', '|---|---:|---:|---:|']
    for part, label in LABELS.items():
        values = [f"{summaries[-1]['threshold_sensitivity'][str(scale)][part]:.2f}%" for scale in (0.5, 1.0, 2.0)]
        lines.append('| ' + ' | '.join([label] + values) + ' |')
    lines += ['', '## 复现与明细', '',
              '- `per_episode.csv`：600 条示教逐条的时长、运动秒数与三档阈值结果。',
              '- `gripper_events.csv`：每个开合指令事件、方向、时间、实际行程和是否有效。',
              '- `summary.json`：完整汇总、阈值与来源。',
              '- `summary.csv`：分任务和总体的部件秒数及占比。',
              '- 复现：`env/b1k2025-policy/bin/python scripts/analyze_b1k_body_activity.py`。', '',
              f"原始数据：`{metadata['raw_root']}`。", '',
              f"机构模型：`{metadata['urdf']}`。", '']
    return '\n'.join(lines)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, default=ROOT / 'outputs/b1k_body_activity_0_34_40')
    parser.add_argument('--urdf', type=Path, default=DEFAULT_URDF)
    args = parser.parse_args()
    raw = ROOT / 'assets/datasets/b1k2025'
    info = json.loads((raw / 'meta/info.json').read_text())
    assert info['fps'] == 30
    chain = torso_chain(args.urdf)
    rows, events, sources = [], [], []
    for task in (0, 34, 40):
        dataset = ROOT / f'assets/datasets/b1k2025_groot_task{task}_v1_200'
        episodes = [json.loads(line) for line in (dataset / 'meta/episodes.jsonl').read_text().splitlines()]
        assert len(episodes) == len({e['episode_index'] for e in episodes}) == 200
        source_ids = sorted(int(p.stem.split('_')[-1]) for p in (raw / 'data' / f'task-{task:04d}').glob('*.parquet'))
        assert sorted(e['episode_index'] for e in episodes) == source_ids
        sources.append(dict(task=task, episode_manifest=str(dataset / 'meta/episodes.jsonl'),
                            episode_ids=source_ids))
        for number, episode in enumerate(episodes):
            eid = episode['episode_index']
            path = raw / info['data_path'].format(episode_chunk=eid // info['chunks_size'], episode_index=eid)
            table = pq.read_table(path, columns=['episode_index', 'task_index', 'timestamp',
                                                'observation.state', 'observation.cam_rel_poses', 'action'])
            state = column_matrix(table, 'observation.state', 256)
            camera = column_matrix(table, 'observation.cam_rel_poses', 21)
            action = column_matrix(table, 'action', 23)
            timestamp = table['timestamp'].to_numpy()
            assert len(state) == episode['length']
            assert np.all(table['episode_index'].to_numpy() == eid)
            assert np.all(table['task_index'].to_numpy() == task)
            assert all(np.isfinite(v).all() for v in (state, camera, action, timestamp))
            np.testing.assert_allclose(timestamp, np.arange(len(state)) / 30, atol=2e-5)
            np.testing.assert_allclose(state[:, 6:10], state[:, 236:240], atol=1e-6)
            np.testing.assert_allclose(state[:, 10:24:2], state[:, 158:165], atol=1e-6)
            np.testing.assert_allclose(state[:, 11:24:2], state[:, 197:204], atol=1e-6)
            np.testing.assert_allclose(np.linalg.norm(camera[:, 17:21], axis=1), 1, atol=2e-5)
            assert np.isin(action[:, [14, 22]], [-1, 1]).all()
            dt = np.diff(timestamp)
            signals = speed_signals(state, camera, timestamp, chain)
            row = dict(task=task, episode_index=eid, frames=len(state), duration_s=float(dt.sum()))
            for scale in (0.5, 1.0, 2.0):
                masks = {key: sustained(value > THRESHOLDS[key] * scale, dt)
                         for key, value in signals.items()}
                masks['base'] = sustained((signals['base_translation'] > THRESHOLDS['base_translation'] * scale)
                                         | (signals['base_rotation'] > THRESHOLDS['base_rotation'] * scale), dt)
                for key, mask in masks.items():
                    seconds = float(dt[mask].sum())
                    row[f'{key}_scale{scale}_active_s'] = seconds
                    if scale == 1:
                        row[f'{key}_active_s'] = seconds
            rows.append(row)
            events.extend(gripper_events(action, state, timestamp, task, eid))
            if (number + 1) % 50 == 0:
                print(f'Task {task}: {number + 1}/200 episodes', flush=True)
    summaries = [summarize(rows, events, task) for task in (0, 34, 40, 'all')]
    assert len(rows) == 600 and len({r['episode_index'] for r in rows}) == 600
    for row in rows:
        assert all(0 <= row[f'{part}_active_s'] <= row['duration_s'] + 1e-8 for part in LABELS)
    metadata = dict(raw_root=str((raw / 'data').resolve()), urdf=str(args.urdf.resolve()),
                    fps=30, velocity_window_s=0.2, minimum_active_duration_s=0.1,
                    thresholds=THRESHOLDS, threshold_scales=[0.5, 1.0, 2.0],
                    gripper_minimum_mean_finger_travel_m=0.005,
                    gripper_minimum_command_frames=3, head_independent_dof=0, sources=sources)
    args.output.mkdir(parents=True, exist_ok=True)
    write_csv(args.output / 'per_episode.csv', rows)
    write_csv(args.output / 'gripper_events.csv', events)
    flat = []
    for summary in summaries:
        for part in LABELS:
            flat.append(dict(task=summary['task'], part=part, label=LABELS[part],
                             episodes=summary['episodes'], duration_s=summary['duration_s'],
                             **summary['activity'][part]))
    write_csv(args.output / 'summary.csv', flat)
    (args.output / 'summary.json').write_text(json.dumps(dict(metadata=metadata, summaries=summaries), ensure_ascii=False, indent=2) + '\n')
    (args.output / 'report.md').write_text(report_text(summaries, metadata))
    print(json.dumps(summaries, ensure_ascii=False, indent=2))
    print(f'Report: {args.output / "report.md"}', flush=True)


if __name__ == '__main__':
    main()
