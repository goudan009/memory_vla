"""Configuration driven Task40 diagnostics over existing, immutable rollouts.

The action step is the fact axis. Old videos have an approximate mapping only.
Missing observations are unknown; a physical outcome does not invent an attempt.
"""
from __future__ import annotations

import csv
from datetime import datetime, timezone
import hashlib
import io
import json
import math
from pathlib import Path
import re

import b1k_metrics

CONFIG = Path(__file__).resolve().parents[2] / 'configs/evaluation/task40_stages.json'
ERRORS = {'navigation_error', 'task_order_error', 'state_order_error',
          'history_state_error', 'execution_failure', 'grasp_failure',
          'contact_failure', 'repetitive_stuck', 'unknown', 'none'}
BEHAVIORS = {'in_kitchen', 'navigate_kitchen', 'choose_microwave', 'pull_door',
             'leave_for_bag', 'grasp_bag', 'place_bag', 'close_door', 'press_start', 'unknown'}


def read_jsonl(path):
    rows = []
    if path.is_file():
        for line in path.read_text(encoding='utf-8').splitlines():
            try:
                row = json.loads(line)
                if isinstance(row, dict):
                    rows.append(row)
            except ValueError:
                # A running writer may not have finished the last line.
                continue
    return rows


def configuration():
    return json.loads(CONFIG.read_text(encoding='utf-8'))


def predicate(facts, conditions):
    """Conjunction in three valued logic. No conditions means no evidence."""
    if not conditions:
        return None
    values = [None if facts.get(key) is None else facts[key] == value
              for key, value in conditions.items()]
    if False in values:
        return False
    return None if None in values else True


def _single(items):
    return next(iter(items.items())) if len(items) == 1 else (None, {})


def physics(row):
    # Do not combine Open from one microwave and Inside from another.
    microwave, oven = _single(row.get('microwaves', {}))
    bag_name, bag = _single(row.get('popcorn', {}))
    held = None
    grasped = row.get('grasped', {})
    grasp_states = {key: value for key, value in row.get('grasp_states', {}).items()
                    if bag_name and key.endswith('/' + bag_name)}
    if bag_name and (bag_name in grasped.values() or 'TRUE' in grasp_states.values()):
        held = True
    elif grasp_states and all(value == 'FALSE' for value in grasp_states.values()):
        held = False
    return dict(door_open=oven.get('Open'), microwave_on=oven.get('ToggledOn'),
                bag_inside=bag.get('inside', {}).get(microwave), bag_held=held,
                bag_cooked=bag.get('cooked'))


def episode_key(episode):
    return 'episode:' + str(episode['episode_index'])


def _annotation_path(run, episode):
    key = hashlib.sha256(episode_key(episode).encode()).hexdigest()
    return run / 'stage_diagnostics' / 'reviews' / (key + '.json')


def review(run, episode):
    path = _annotation_path(run, episode)
    return json.loads(path.read_text(encoding='utf-8')) if path.is_file() else None


def validate_events(events, episode):
    if not isinstance(events, list) or len(events) > 500:
        raise ValueError('events 必须为不超过500条的事件列表')
    result = []
    limit = episode.get('sim_steps') or episode.get('episode_length')
    for event in events:
        if not isinstance(event, dict):
            raise ValueError('事件必须为JSON对象')
        item = dict(event)
        if item.get('stage_id') not in {s['id'] for s in configuration()['stages']}:
            raise ValueError('无效的阶段编号')
        for name in ('action_start', 'action_end', 'confidence'):
            value = item.get(name)
            if type(value) not in (int, float) or not math.isfinite(value):
                raise ValueError(name + ' 必须为有限数值')
        if not (0 <= item['action_start'] <= item['action_end']
                and (limit is None or item['action_end'] <= limit)
                and 0 <= item['confidence'] <= 1):
            raise ValueError('事件时间或置信度超出范围')
        if item.get('observed_behavior') not in BEHAVIORS or item.get('error_type') not in ERRORS:
            raise ValueError('不支持的行为或错误分类')
        if item.get('outcome', 'attempted') not in ('attempted', 'completed', 'failed', 'unknown', 'not_reached'):
            raise ValueError('无效的事件结果')
        item['outcome'] = item.get('outcome', 'attempted')
        if item['outcome'] == 'not_reached' and (item['action_start'] != 0 or limit is None
                or item['action_end'] != limit or item['observed_behavior'] != 'unknown'):
            raise ValueError('未到达判断需覆盖整集，行为应为unknown')
        if not isinstance(item.get('explanation'), str) or len(item['explanation']) > 10000:
            raise ValueError('事件需要文字证据说明')
        frames = item.get('evidence_frames', [])
        if not isinstance(frames, list) or any(type(x) is not int or x < 0 for x in frames):
            raise ValueError('evidence_frames 必须为非负帧编号列表')
        item['evidence_frames'] = frames
        item['episode_id'] = episode_key(episode)
        result.append(item)
    return result


def save_review(run, episode, request):
    if request.get('decision') not in ('confirmed', 'modified', 'unknown'):
        raise ValueError('复核状态应为 confirmed、modified 或 unknown')
    events = validate_events(request.get('events', []), episode)
    notes = request.get('notes', '')
    if not isinstance(notes, str) or len(notes) > 10000:
        raise ValueError('复核备注过长或格式不正确')
    record = dict(schema_version=1, episode_id=episode_key(episode),
                  source='human', decision=request['decision'], events=events,
                  notes=notes, updated_at=datetime.now(timezone.utc).isoformat())
    path = _annotation_path(run, episode)
    path.parent.mkdir(parents=True, exist_ok=True)
    # Separate annotation history; raw trace and VL output are never overwritten.
    history = path.with_suffix('.jsonl')
    with history.open('a', encoding='utf-8') as stream:
        stream.write(json.dumps(record, ensure_ascii=False) + '\n')
    path.write_text(json.dumps(record, ensure_ascii=False, indent=2), encoding='utf-8')
    return record


def _select_traces(run, episode):
    rows = []
    for path in sorted((run / 'stages').glob('task40_gpu*_env*.jsonl')):
        gpu = re.fullmatch(r'task40_gpu(\d+)_env(\d+)\.jsonl', path.name)
        if not gpu or int(gpu[1]) != episode.get('gpu_index') or int(gpu[2]) != episode.get('env_idx'):
            continue
        matching = [row for row in read_jsonl(path)
                    if all(row.get(key) == episode.get(key) for key in ('instance_id', 'seed', 'env_idx'))]
        # A repeated identity/reset is ambiguous; never silently mix episodes.
        if len({row.get('episode_number') for row in matching}) > 1:
            return [], 'ambiguous_trace_identity'
        for row in matching:
            if type(row.get('action_steps')) in (int, float):
                rows.append(dict(row, trace_file=path.relative_to(run).as_posix()))
    return sorted(rows, key=lambda row: row['action_steps']), None


def _rate(numerator, denominator, valid_n, unknown_n=0):
    return dict(numerator=numerator, denominator=denominator, valid_n=valid_n,
                unknown_n=unknown_n, value=numerator / denominator if denominator else None)


def memory_diagnostics(run, episode):
    path = run / 'stages' / f"memory_gpu{episode.get('gpu_index')}_env{episode.get('env_idx')}.jsonl"
    rows = [row for row in read_jsonl(path)
            if all(row.get(key) == episode.get(key) for key in ('instance_id','seed','env_idx'))]
    warning = None
    if len({(row.get('session_id'),row.get('episode_id')) for row in rows}) > 1:
        rows, warning = [], 'ambiguous_memory_identity'
    return dict(sample_count=len(rows),samples=rows,warning=warning,
                evidence_file=path.relative_to(run).as_posix() if path.is_file() else None,
                model_module_cause='unknown',
                interpretation='槽位、门控、残差与注意力是运行观测，不能单独证明错误根因或历史利用正确。',
                time_axis='decision_index；不是仿真动作步或精确视频时间')


def video_mapping(run, episode, scale):
    path = run/'stages'/f"video_gpu{episode.get('gpu_index')}_env{episode.get('env_idx')}.jsonl"
    rows = [row for row in read_jsonl(path)
            if all(row.get(key) == episode.get(key) for key in ('instance_id','seed','env_idx'))]
    result = dict(axis='action_steps', video_seconds_per_action_step=1/scale,
                  approximate=True, reason='旧录像无逐帧仿真索引及起录偏移')
    if not rows:
        return result
    if len({(r.get('episode_number'), r.get('recording_id')) for r in rows}) != 1:
        return dict(result, warning='ambiguous_video_identity')
    if (any(type(r.get(key)) not in (int, float) or not math.isfinite(r[key])
            for r in rows for key in ('frame_index','action_steps','sim_time_s','video_fps'))
            or [r['frame_index'] for r in rows] != list(range(len(rows)))
            or any(r['video_fps'] <= 0 for r in rows)
            or len({r['video_fps'] for r in rows}) != 1
            or any(b['action_steps'] <= a['action_steps'] or b['sim_time_s'] < a['sim_time_s']
                   for a,b in zip(rows, rows[1:]))):
        return dict(result, warning='invalid_video_frame_index')
    return dict(result, approximate=False, reason='逐帧动作步与仿真时间实录',
                evidence_file=path.relative_to(run).as_posix(), frames=rows)


def video_position(mapping, step):
    """Seek the closest recorded frame; unsampled physical states stay approximate."""
    rows = mapping.get('frames', [])
    if rows:
        closest = min(rows, key=lambda row: abs(row['action_steps']-step))
        return closest['frame_index']/closest['video_fps'], closest['action_steps'] != step
    return step*mapping['video_seconds_per_action_step'], True


def analyze_episode(run, episode, config=None, traces=None, events=None):
    config = config or configuration()
    results = json.loads((run / 'results.json').read_text(encoding='utf-8'))
    video = results.get('configuration', {})
    scale = (video.get('video_steps_per_render') or 24) * (video.get('video_fps') or 20)
    mapping = video_mapping(run, episode, scale)
    rows, trace_warning = _select_traces(run, episode) if traces is None else (traces, None)
    annotation = review(run, episode)
    source = 'supplied'
    if events is None:
        raw = run / 'stage_diagnostics' / 'vl_events' / (str(episode['episode_index']) + '.json')
        document = json.loads(raw.read_text()) if raw.is_file() else {}
        events = validate_events(document.get('events', []), episode)
        source = 'vl'
        if annotation:
            # An uncertain human review suppresses unconfirmed model judgements.
            events = annotation['events'] if annotation['decision'] != 'unknown' else []
            source = 'human'
    else:
        events = validate_events(events, episode)
    candidate_events = events
    # Model confidence is not calibration. Keep unreviewed VL events visible,
    # but require the independent review before they affect research metrics.
    if source == 'vl':
        events = []
    stages = {s['id']: dict(id=s['id'], name=s['name'], reached=None, attempted=None,
                           completed=None, failed=None, first_reached_step=None,
                           first_attempt_step=None, first_completed_step=None,
                           attempt_outcome=None,
                           evidence=[], status='unknown') for s in config['stages']}
    timeline, transitions, failures = [], [], []
    previous = {}
    for row in rows:
        step = row['action_steps']
        facts = physics(row)
        evidence = dict(source='physics', action_steps=step, trace_file=row.get('trace_file'),
                        facts=facts, sim_time_s=row.get('sim_time_s'), robots=row.get('robots'),
                        joint_positions={k:v.get('joint_positions') for k,v in row.get('microwaves', {}).items()})
        seconds, approximate = video_position(mapping, step)
        timeline.append(dict(action_steps=step, video_seconds=seconds,
                             video_time_approximate=approximate, facts=facts, evidence=evidence))
    # Visual behavior is transient, attached only to samples within its evidence window.
    points = [(point['action_steps'], point, None) for point in timeline]
    points += [(event['action_start'], None, event) for event in events]
    sampled_facts = {point['action_steps']: point['facts'] for point in timeline}
    for step, point, event in sorted(points, key=lambda x: (x[0], x[1] is None)):
        # No zero-order hold: a closed door at step 80 does not prove its state
        # at a visual action at step 120. Only coincident samples are current facts.
        facts = dict(point['facts'] if point else {} if event.get('action_time_approximate')
                     else sampled_facts.get(step, {}))
        active = [ev for ev in events if ev['action_start'] <= step <= ev['action_end']
                  and ev['outcome'] not in ('unknown','not_reached')
                  and (point is None or not ev.get('action_time_approximate'))]
        facts.update({ev['observed_behavior']: True for ev in active})
        for definition in config['stages']:
            stage = stages[definition['id']]
            entry = predicate(facts, definition['entry_condition'])
            attempt = predicate(facts, definition['attempt_condition'])
            complete = predicate(facts, definition['success_predicate'])
            if attempt is True and stage['first_attempt_step'] is None:
                stage['first_attempt_step'] = step
            if (stage['first_attempt_step'] is not None and step >= stage['first_attempt_step']
                    and complete is not None):
                stage['attempt_outcome'] = stage['attempt_outcome'] is True or complete
            for key, value, timestamp in [('reached', entry, 'first_reached_step'),
                                          ('attempted', attempt, 'first_attempt_step'),
                                          ('completed', complete, 'first_completed_step')]:
                if value is True:
                    if stage[key] is not True:
                        stage[timestamp] = step
                        stage['evidence'].append(point['evidence'] if point else dict(source=source, **event))
                    stage[key] = True
                elif value is False and stage[key] is None and key == 'completed':
                    # This is an observed false outcome, not proof the stage was never reached.
                    stage[key] = False
            if complete is True:
                if stage['reached'] is not True:
                    stage['first_reached_step'] = step
                stage['reached'] = True
        for rule in config['transition_rules']:
            trigger = rule['trigger']
            triggered = (point is not None and facts.get(trigger) is True and previous.get(trigger) is False)
            if event and event['outcome'] not in ('unknown','not_reached') and event['observed_behavior'] == trigger:
                triggered = True
            if not triggered:
                continue
            precondition = predicate(facts, rule['precondition'])
            transition = dict(id=f'transition-{len(transitions)+1}', stage_id=rule['stage_id'],
                              action_steps=step, valid=precondition,
                              error_type=rule['error_type'] if precondition is False else 'none',
                              explanation=rule['explanation'] if precondition is False else '阶段前置条件检查',
                              evidence=point['evidence'] if point else dict(source=source, **event))
            transitions.append(transition)
            if precondition is False:
                failures.append(dict(transition, kind='invalid_transition'))
        if event and event['outcome'] == 'failed':
            # A failed result requires an observed attempt, never simply a closed door.
            if event['observed_behavior'] != 'unknown':
                stage = stages[event['stage_id']]
                stage['failed'] = True
                if stage['attempt_outcome'] is None:
                    stage['attempt_outcome'] = False
                failures.append(dict(id=f'event-{len(failures)+1}', stage_id=event['stage_id'],
                                     action_steps=event['action_end'], kind='observed_failure',
                                     error_type=event['error_type'], explanation=event['explanation'],
                                     evidence=dict(source=source, **event)))
        if point:
            previous = point['facts']
    for stage in stages.values():
        negative = [ev for ev in events if ev['stage_id'] == stage['id'] and ev['outcome'] == 'not_reached']
        if negative and stage['reached'] is None and stage['attempted'] is None and stage['completed'] is not True:
            stage.update(reached=False, attempted=False, completed=False)
            stage['evidence'].extend(dict(source=source, **ev) for ev in negative)
        stage['status'] = ('completed' if stage['completed'] is True else 'failed' if stage['failed'] is True
                           else 'attempted' if stage['attempted'] is True else 'reached' if stage['reached'] is True
                           else 'not_reached' if stage['reached'] is False
                           else 'unknown')
    failures.sort(key=lambda item: item['action_steps'])
    for index, failure in enumerate(failures):
        failure['role'] = 'first_observed_failure' if index == 0 else 'subsequent_error'
        failure['preceding_failure_id'] = failures[index-1]['id'] if index else None
        failure['causal_relationship'] = 'unknown'  # temporal order is not causality
    completed = sum(s['completed'] is True for s in stages.values())
    known = sum(s['completed'] is not None for s in stages.values())
    return dict(schema_version=1, task_id=40, configuration_status=config['status'],
                episode_id=episode_key(episode), identity={k:episode.get(k) for k in
                    ('episode_index','instance_id','seed','gpu_index','env_idx','video_file','valid')},
                official=dict({k:episode.get(k) for k in ('success','q_score','task_progress','native_termination')},
                              b1k_success=b1k_metrics.terminal_success(episode)),
                policy_identity=video.get('policy_identity'), protocol=video.get('protocol'),
                time_mapping=mapping,
                trace_samples=len(rows), trace_warning=trace_warning,
                memory_diagnostics=memory_diagnostics(run,episode),
                current_state=timeline[-1]['facts'] if timeline else {},
                stages=list(stages.values()), timeline=timeline, events=candidate_events, event_source=source,
                events_used_for_metrics=len(events),
                event_review_required=source=='vl' and bool(candidate_events),
                transitions=transitions, failures=failures,
                first_observed_failure=failures[0] if failures else None,
                attribution='unknown' if not failures else 'behavioral_evidence_only',
                custom_stage_completion=dict(name='Custom Stage Completion', completed=completed,
                    total=len(stages), observed_stages=known, unknown_stages=len(stages)-known,
                    observed_fraction=completed/known if known else None,
                    confirmed_lower_bound=completed/len(stages)), human_review=annotation)


def analyze_run(run):
    result = json.loads((run/'results.json').read_text(encoding='utf-8'))
    if str(result.get('task_id', result.get('configuration', {}).get('task_id'))) != '40':
        raise ValueError('首版仅支持Task40')
    episodes = read_jsonl(run/'episodes.jsonl')
    reports = [analyze_episode(run, ep) for ep in episodes]
    return dict(schema_version=1, configuration=configuration(), run=str(run),
                evaluation_mode=result.get('evaluation_mode'), episodes=reports,
                summary=summarize(reports,result.get('evaluation_mode')))


def summarize(reports, evaluation_mode):
    valid = [r for r in reports if r['identity'].get('valid') is True]
    stages = {}
    for definition in configuration()['stages']:
        samples = [next(s for s in r['stages'] if s['id'] == definition['id']) for r in valid]
        attempts = [s for s in samples if s['attempted'] is True]
        known_attempts = [s for s in attempts if s['attempt_outcome'] is not None]
        observed = [s for s in samples if s['reached'] is not None]
        reach = _rate(sum(s['reached'] is True for s in observed),len(valid),len(valid),len(valid)-len(observed))
        reach['interpretation'] = 'confirmed_lower_bound_with_unknown_coverage'
        stages[definition['id']] = dict(name=definition['name'],
            reach_rate=reach,
            execution_success_rate=_rate(sum(s['attempt_outcome'] is True for s in known_attempts),len(known_attempts),len(valid),len(valid)-len(known_attempts)),
            observed_attempts=len(attempts), unknown_attempts=sum(s['attempted'] is None for s in samples),
            completed=sum(s['completed'] is True for s in samples))
    transitions = [t for r in valid for t in r['transitions']]
    decisions = [t for t in transitions if t['valid'] is not None]
    distribution = {}
    for r in valid:
        f = r['first_observed_failure']
        key = f['stage_id']+':'+f['error_type'] if f else 'unknown' if not r['official'].get('success') else 'official_success'
        distribution[key] = distribution.get(key,0)+1
    q = [r['official']['q_score'] for r in valid if r['official'].get('q_score') is not None]
    successes = [r['official']['b1k_success'] for r in valid if r['official'].get('b1k_success') is not None]
    return dict(episodes=len(reports), valid_n=len(valid), stages=stages,
                    official_success_rate=_rate(sum(successes),len(successes),len(valid),len(valid)-len(successes)),
                    official_mean_q=sum(q)/len(q) if q and evaluation_mode=='official-eval' else None,
                    q_valid_n=len(q), invalid_transition_rate=_rate(sum(t['valid'] is False for t in decisions),len(decisions),len(valid),len(transitions)-len(decisions)),
                    decision_accuracy=_rate(sum(t['valid'] is True for t in decisions),len(decisions),len(valid),len(transitions)-len(decisions)),
                    first_observed_failure_distribution=distribution,
                    metric_scope='诊断草案；仅统计有证据的阶段与选择机会，unknown不按失败计算')


def compare(left, right):
    """Match actual scene identities, reject protocol mismatches, expose coverage."""
    def keyed(report):
        output = {}
        duplicates = set()
        for ep in report['episodes']:
            key = tuple(ep['identity'].get(k) for k in ('instance_id','seed'))
            if ep['identity'].get('valid') is not True or None in key:
                continue
            if key in output:
                duplicates.add(key)
            output[key] = ep
        return {key:value for key,value in output.items() if key not in duplicates}
    l, r = keyed(left), keyed(right)
    matched, rejected = [], []
    paired_episodes={'left':[],'right':[]}
    for key in sorted(l.keys() & r.keys()):
        a, b = l[key], r[key]
        if not a.get('protocol') or a['protocol'] != b.get('protocol'):
            rejected.append(dict(instance_id=key[0],seed=key[1],reason='protocol_missing_or_mismatched'))
            continue
        paired_episodes['left'].append(a)
        paired_episodes['right'].append(b)
        matched.append(dict(instance_id=key[0],seed=key[1],left_episode=a['episode_id'],right_episode=b['episode_id'],
            left_official=a['official'],right_official=b['official'],
            stages={x['id']:dict(left_completed=x['completed'],right_completed=y['completed'],
                                 left_reached=x['reached'],right_reached=y['reached'],
                                 left_attempted=x['attempted'],right_attempted=y['attempted'],
                                 left_attempt_outcome=x['attempt_outcome'],right_attempt_outcome=y['attempt_outcome'],
                                 left_status=x['status'],right_status=y['status'])
                    for x,y in zip(a['stages'],b['stages'])}))
    stage_metrics = {}
    for definition in configuration()['stages']:
        key = definition['id']
        item = dict(name=definition['name'])
        for side in ('left','right'):
            reached = [p['stages'][key][side+'_reached'] for p in matched]
            attempted = [p['stages'][key] for p in matched if p['stages'][key][side+'_attempted'] is True]
            outcomes = [s[side+'_attempt_outcome'] for s in attempted if s[side+'_attempt_outcome'] is not None]
            item[side+'_reach'] = _rate(sum(x is True for x in reached),len(matched),len(matched),sum(x is None for x in reached))
            item[side+'_execution'] = _rate(sum(outcomes),len(outcomes),len(matched),len(matched)-len(outcomes))
        stage_metrics[key] = item
    paired_success = {}
    for side in ('left','right'):
        outcomes = [p[side+'_official']['b1k_success'] for p in matched if p[side+'_official']['b1k_success'] is not None]
        paired_success[side] = _rate(sum(outcomes),len(outcomes),len(matched),len(matched)-len(outcomes))
    return dict(left_run=left['run'],right_run=right['run'],left_summary=left['summary'],right_summary=right['summary'],
                paired_summaries={side:summarize(episodes,report.get('evaluation_mode'))
                                  for side,episodes,report in
                                  (('left',paired_episodes['left'],left),('right',paired_episodes['right'],right))},
                matched_n=len(matched), pairs=matched,rejected=rejected,paired_stage_metrics=stage_metrics,paired_success=paired_success,
                unmatched_left=len(l)-len(matched),unmatched_right=len(r)-len(matched),
                scope='按有效回合的instance+seed+protocol配对；硬件与策略RNG轨迹仍须单独核对，草案指标不支持因果结论')


def archive_run(run):
    report = analyze_run(run)
    output = run/'stage_diagnostics'/'reports'
    output.mkdir(parents=True, exist_ok=True)
    for format in ('json','csv','md'):
        (output/f'task40_stages.{format}').write_text(export(report,format),encoding='utf-8')
    return dict(summary=report['summary'], artifacts=[str(output/f'task40_stages.{f}') for f in ('json','csv','md')])


def recommendations(report):
    """Evidence linked next checks, without assigning errors to model modules."""
    actions = []
    by_stage = {}
    for ep in report['episodes']:
        for failure in ep['failures']:
            by_stage.setdefault(failure['stage_id'], []).append(dict(
                episode_id=ep['episode_id'],identity=ep['identity'],event=failure))
    for stage, evidence in by_stage.items():
        check = ('检查门曾打开后是否重新关闭；对齐抓袋/放入时的当前Open、视频与接触尝试。'
                 if stage in ('S5','S6','S7') else
                 '检查启动时Inside与Open前置条件；区分误触按钮、关门失败和目标选择。'
                 if stage == 'S9' else '复核阶段尝试、物理结果与先后顺序。')
        actions.append(dict(stage_id=stage,observed_events=len(evidence),
                            recommendation=check,evidence=evidence,model_module_cause='unknown'))
    unknown = [ep['episode_id'] for ep in report['episodes'] if not ep['trace_samples']]
    if unknown:
        actions.append(dict(recommendation='旧回合缺物理记录，先复核既有视频候选；不要因增加指标重跑全部模型。',episodes=unknown))
    pending = [ep['episode_id'] for ep in report['episodes'] if ep.get('event_review_required')]
    if pending:
        actions.append(dict(recommendation='VL候选待人工确认或修改，暂不纳入阶段指标。',episodes=pending))
    return dict(run=report['run'],valid_n=report['summary']['valid_n'],actions=actions,
                interpretation='建议用于选择下一项验证，不证明Memory或Body责任；参数/结构修改需配对干预与重复闭环支持。',
                next_experiment='固定基座、数据、协议、实例及种子，同一权重normal/empty/other_episode对照；开发集选型与保留测试分开。')


def export(report, format='json'):
    if format == 'json':
        return json.dumps(dict(report,recommendations=recommendations(report)), ensure_ascii=False, indent=2)
    if format == 'csv':
        out = io.StringIO()
        writer = csv.writer(out)
        writer.writerow(['episode','instance','seed','valid','official_success','official_q','stage','status','reached','attempted','completed','failed','first_completed_action_step','evidence'])
        for episode in report['episodes']:
            for s in episode['stages']:
                writer.writerow([episode['episode_id'],episode['identity']['instance_id'],episode['identity']['seed'],episode['identity']['valid'],episode['official']['b1k_success'],episode['official']['q_score'],s['id'],s['status'],s['reached'],s['attempted'],s['completed'],s['failed'],s['first_completed_step'],json.dumps(s['evidence'],ensure_ascii=False)])
        return out.getvalue()
    if format != 'md':
        raise ValueError('导出格式应为json、csv或md')
    s = report['summary']
    lines = ['# Task40 阶段诊断（草案）', '', s['metric_scope'], '',
             f"有效回合：{s['valid_n']}；官方成功：{s['official_success_rate']['numerator']}/{s['official_success_rate']['denominator']}；官方平均Q：{s['official_mean_q']}", '',
             '|阶段|到达（有证据）|执行成功（已尝试且结果已知）|未知到达|', '|---|---|---|---|']
    for key, stage in s['stages'].items():
        a,b = stage['reach_rate'],stage['execution_success_rate']
        lines.append(f"|{key} {stage['name']}|{a['numerator']}/{a['denominator']}|{b['numerator']}/{b['denominator']}|{a['unknown_n']}|")
    for ep in report['episodes']:
        lines += ['', f"## {ep['episode_id']} instance={ep['identity']['instance_id']} seed={ep['identity']['seed']}", '',
                  ('视频含逐帧动作步/仿真时间索引；帧间状态仍未被视频记录。'
                   if ep['time_mapping'].get('frames') else
                   '视频时间为近似；精确证据轴为 action_steps。'), '']
        for stage in ep['stages']:
            lines.append(f"- {stage['id']} {stage['name']}：{stage['status']}；完成动作步={stage['first_completed_step']}")
        for failure in ep['failures']:
            lines.append(f"- {failure['role']}：{failure['stage_id']} @{failure['action_steps']} {failure['explanation']}；证据={json.dumps(failure['evidence'],ensure_ascii=False)}")
    advice = recommendations(report)
    lines += ['', '## 下一轮验证建议', '', advice['interpretation'], '', advice['next_experiment'], '']
    for item in advice['actions']:
        lines.append('- '+item['recommendation']+' '+json.dumps({k:v for k,v in item.items() if k!='recommendation'},ensure_ascii=False))
    return '\n'.join(lines) + '\n'
