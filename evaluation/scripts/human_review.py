"""Independent, structured human judgements for evaluation Episodes."""
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
import re
import tempfile
import uuid

BEHAVIORS = ('chaotic_motion', 'repetitive_stuck', 'goal_directed_incomplete',
             'near_complete', 'completed', 'unclear')
STAGES = ('not_started', 'approach', 'grasp', 'transport', 'place', 'other', 'none', 'unclear')
REACHED_STAGES = ('not_started', 'decision', 'navigation', 'approach', 'manipulation',
                  'grasp', 'transport_place', 'verification', 'completed', 'unclear')
ROOT_CAUSES = ('decision_error', 'perception_error', 'navigation_error',
               'memory_plan_drift', 'task_order_error', 'manipulation_precision',
               'grasp_contact_failure', 'control_execution', 'premature_stop_timeout',
               'environment_issue', 'other', 'unclear', 'none')
AGREEMENT = ('agree', 'disagree', 'unclear')
FIELDS = {'execution_quality', 'visual_progress', 'behavior_class', 'failure_stage',
          'target_touched', 'stuck', 'vlm_agreement', 'reached_stage',
          'stage_label', 'root_cause', 'root_cause_label', 'specific_reason',
          'specific_reason_label', 'secondary_root_cause', 'secondary_root_cause_label',
          'secondary_specific_reason', 'secondary_specific_reason_label', 'reviewed'}
STAGE_OPTIONS_PATH = Path(__file__).resolve().parents[1] / 'configs' / 'human_review_stages.json'
_DEFAULT_STAGES = {
    '40': [
        '01 到达厨房', '02 选择微波炉', '03 尝试开门', '04 确认门已打开',
        '05 前往爆米花袋', '06 抓取袋子', '07 放入微波炉', '08 关闭炉门',
        '09 启动微波炉',
    ]
}
_DEFAULT_CAUSES = (
    ('D', 'D. 任务决策错误', (
        ('D1', 'D1 初始决策错误'), ('D2', 'D2 错误阶段切换'))),
    ('N', 'N. 导航与定位错误', (
        ('N1', 'N1 导航失败'), ('N2', 'N2 导航位置错误'))),
    ('M', 'M. 操作执行错误', (
        ('M1', 'M1 抓取与接触失败'), ('M2', 'M2 动作精度与可达性不足'),
        ('M3', 'M3 动作控制异常'))),
    ('S', 'S. 成功', (('S1', 'S1 任务完成'),)),
)
_LEGACY_CAUSE_CODES = {
    'decision_error': 'D', 'perception_error': 'D', 'memory_plan_drift': 'D',
    'task_order_error': 'D', 'premature_stop_timeout': 'D',
    'navigation_error': 'N',
    'manipulation_precision': 'M', 'grasp_contact_failure': 'M',
    'control_execution': 'M', 'environment_issue': 'M', 'other': 'M', 'unclear': 'M',
    'none': 'S',
}


def target(run, episode_index=None, video='', videos=()):
    if episode_index is not None and (type(episode_index) is not int or episode_index < 1):
        raise ValueError('回合编号必须是正整数')
    if not isinstance(video, str):
        raise ValueError('录像路径必须为字符串')
    rows = []
    journal = run/'episodes.jsonl'
    if journal.is_file():
        for line in journal.read_text(encoding='utf-8').splitlines():
            try:
                row = json.loads(line)
            except ValueError:
                continue
            if isinstance(row, dict):
                rows.append(row)
    if episode_index is not None:
        matches = [row for row in rows if row.get('episode_index') == episode_index]
        if len(matches) != 1:
            raise ValueError('找不到唯一的评估回合')
        row = matches[0]
        if video and video != row.get('video_file'):
            raise ValueError('回合编号与录像不匹配')
    else:
        matches = [row for row in rows if video and row.get('video_file') == video]
        row = matches[0] if len(matches) == 1 else None
        if row and type(row.get('episode_index')) is int and row['episode_index'] > 0:
            episode_index = row['episode_index']
        elif video and video in videos:
            row = None
        else:
            raise ValueError('请选择已结束的回合或有效录像')
    key = f'episode:{episode_index}' if episode_index is not None else f'video:{video}'
    return dict(key=key, episode_index=episode_index,
                video_file=(row.get('video_file') if row else video),
                **{name: row.get(name) if row else None
                   for name in ('instance_id', 'seed', 'gpu_index', 'env_idx', 'task_id')})


def stage_id(label):
    return hashlib.sha256(label.encode('utf-8')).hexdigest()[:16]


def _load_options():
    try:
        return json.loads(STAGE_OPTIONS_PATH.read_text(encoding='utf-8'))
    except FileNotFoundError:
        return {'schema_version': 2, 'tasks': {}}
    except (OSError, ValueError):
        return {'schema_version': 2, 'tasks': {}}


def _task_config(data, task, create=False):
    tasks = data.setdefault('tasks', {})
    config = tasks.get(task)
    if config is None:
        labels = _DEFAULT_STAGES.get(task, [])
        stages = [{'id': stage_id(label), 'label': label} for label in labels]
        config = {'stages': stages, 'causes': {}, 'notes': {}}
        if create or labels:
            tasks[task] = config
    elif isinstance(config, list):
        config = {'stages': [{'id': stage_id(label), 'label': label} for label in config],
                  'causes': {}, 'notes': {}}
        tasks[task] = config
    # Migrate the first saved format, which stored only an ordered label list.
    elif isinstance(config.get('stages'), list) and config['stages'] and isinstance(config['stages'][0], str):
        config['stages'] = [{'id': stage_id(label), 'label': label} for label in config['stages']]
    config.setdefault('causes', {})
    config.setdefault('notes', {})
    _ensure_default_taxonomy(config)
    config['supplements'] = _stage_supplements(config.get('supplements', {}))
    return config


def _stage_supplements(value):
    """Normalize supplementary reasons to independent lists keyed only by stage."""
    if not isinstance(value, dict):
        return {}
    result = {}
    for stage_id_value, entries in value.items():
        groups = entries.values() if isinstance(entries, dict) else (entries,)
        normalized, seen = [], set()
        for group in groups:
            if not isinstance(group, list):
                continue
            for item in group:
                if not isinstance(item, dict) or not isinstance(item.get('id'), str):
                    continue
                option_id = item['id']
                if option_id in seen:
                    continue
                normalized.append({key: val for key, val in item.items() if key != 'default_code'})
                seen.add(option_id)
        result[stage_id_value] = normalized
    return result


def _ensure_default_taxonomy(config):
    default_codes = {code for code, _, _ in _DEFAULT_CAUSES}
    for stage in config['stages']:
        stage_key = stage['id']
        notes = config['notes'].setdefault(stage_key, {})
        existing = config['causes'].get(stage_key, [])
        grouped = {code: [] for code in default_codes}
        custom = []
        for cause in existing:
            code = cause.get('default_code')
            if code not in default_codes:
                code = cause.get('id') if cause.get('id') in default_codes else None
            if code not in default_codes:
                match = re.match(r'^\s*([DNMS])\s*[.、]', cause.get('label', ''))
                code = match.group(1) if match else _LEGACY_CAUSE_CODES.get(cause.get('id'))
            if code in default_codes:
                grouped[code].append(cause)
            else:
                custom.append(cause)

        normalized = []
        for code, default_label, specifics in _DEFAULT_CAUSES:
            candidates = grouped[code]
            cause = candidates[0] if candidates else {'id': code, 'label': default_label}
            if cause.get('default_code') != code:
                cause['label'] = default_label
            cause['default_code'] = code
            cause_id_value = cause['id']
            merged_notes = []
            seen_labels = set()
            for source in candidates:
                for item in notes.get(source['id'], []):
                    label = item.get('label', '')
                    if label and label not in seen_labels:
                        merged_notes.append(item)
                        seen_labels.add(label)
            default_notes = []
            default_note_ids = set()
            for specific_code, specific_label in specifics:
                candidate = next((item for item in merged_notes
                                  if item.get('default_code') == specific_code
                                  or item.get('id') == specific_code
                                  or re.match(rf'^\s*{specific_code}\s+', item.get('label', ''))), None)
                if candidate is None:
                    candidate = {'id': specific_code, 'label': specific_label}
                elif candidate.get('default_code') != specific_code:
                    candidate['label'] = specific_label
                candidate['default_code'] = specific_code
                default_notes.append(candidate)
                default_note_ids.add(candidate['id'])
            notes[cause_id_value] = default_notes + [
                item for item in merged_notes if item['id'] not in default_note_ids]
            for duplicate in candidates[1:]:
                notes.pop(duplicate['id'], None)
            normalized.append(cause)
        config['causes'][stage_key] = normalized + custom


def stage_options(task_id):
    task = str(task_id)
    config = _task_config(_load_options(), task)
    return [dict(stage, supplements=config['supplements'].get(stage['id'], []),
                 causes=[dict(cause, notes=config['notes'].get(stage['id'], {}).get(cause['id'], []))
                               for cause in config['causes'].get(stage['id'], [])])
            for stage in config['stages']]


def _write_options(data):
    STAGE_OPTIONS_PATH.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(mode='w', encoding='utf-8', dir=STAGE_OPTIONS_PATH.parent,
                                     prefix='.human-review-options-', suffix='.pending', delete=False) as handle:
        temporary = Path(handle.name)
        try:
            json.dump(data, handle, ensure_ascii=False, indent=2)
            handle.write('\n')
            handle.close()
            temporary.replace(STAGE_OPTIONS_PATH)
        finally:
            temporary.unlink(missing_ok=True)


def update_stage_options(task_id, action, label=None, option_id=None, before_id=None):
    if type(task_id) not in (int, str) or not re.fullmatch(r'\d{1,4}', str(task_id)):
        raise ValueError('Task 编号无效')
    task = str(task_id)
    data = _load_options()
    config = _task_config(data, task, create=True)
    stages = config['stages']
    if action == 'add':
        if not isinstance(label, str):
            raise ValueError('阶段名称不能为空')
        label = label.strip()
        if not label or len(label) > 80:
            raise ValueError('阶段名称需为1到80个字符')
        label = re.sub(r'^\s*\d{1,3}\s*[-.、:：)]?\s*', '', label)
        if not label or len(label) > 80:
            raise ValueError('阶段名称需为1到80个字符')
        if any(re.sub(r'^\d{2}\s+', '', stage['label']) == label for stage in stages):
            raise ValueError('这个阶段已经存在')
        stage = {'id': uuid.uuid4().hex[:16], 'label': label}
        if before_id:
            index = next((i for i, item in enumerate(stages) if item['id'] == before_id), None)
            if index is None:
                raise ValueError('找不到插入位置')
            stages.insert(index, stage)
        else:
            stages.append(stage)
    elif action == 'rename':
        _validate_label(label)
        stage = next((item for item in stages if item['id'] == option_id), None)
        if stage is None:
            raise ValueError('找不到要修改的阶段')
        name = re.sub(r'^\s*\d{1,3}\s*[-.、:：)]?\s*', '', label.strip())
        if not name:
            raise ValueError('阶段名称不能为空')
        if any(item['id'] != option_id and
               re.sub(r'^\s*\d{1,3}\s*[-.、:：)]?\s*', '', item['label']) == name
               for item in stages):
            raise ValueError('这个阶段已经存在')
        stage['label'] = name
    elif action == 'delete':
        index = next((i for i, item in enumerate(stages) if item['id'] == option_id), None)
        if index is None:
            raise ValueError('找不到要删除的阶段')
        removed = stages.pop(index)
        config['causes'].pop(removed['id'], None)
        config['notes'].pop(removed['id'], None)
        config['supplements'].pop(removed['id'], None)
    else:
        raise ValueError('不支持的阶段操作')
    for index, item in enumerate(stages, 1):
        name = re.sub(r'^\d{1,3}\s+', '', item['label'])
        item['label'] = f'{index:02d} {name}'
    _write_options(data)
    return stage_options(task)


def update_annotation_options(task_id, kind, action, stage_id_value, cause_id=None,
                              label=None, option_id=None):
    task = str(task_id)
    data = _load_options()
    config = _task_config(data, task, create=True)
    stage = next((item for item in config['stages'] if item['id'] == stage_id_value), None)
    if stage is None:
        raise ValueError('请先选择有效任务阶段')
    if kind == 'cause':
        options = config['causes'].setdefault(stage_id_value, [])
        if action == 'add':
            _validate_label(label)
            option = {'id': uuid.uuid4().hex[:16], 'label': label.strip()}
            if any(item['label'] == option['label'] for item in options):
                raise ValueError('这个根因选项已经存在')
            options.append(option)
        elif action == 'rename':
            _validate_label(label)
            option = next((item for item in options if item['id'] == option_id), None)
            if option is None:
                raise ValueError('找不到要修改的根因选项')
            if any(item['id'] != option_id and item['label'] == label.strip() for item in options):
                raise ValueError('这个根因选项名称已经存在')
            option['label'] = label.strip()
        elif action == 'delete':
            option = next((item for item in options if item['id'] == option_id), None)
            if option is None:
                raise ValueError('找不到要删除的根因选项')
            if option.get('default_code'):
                raise ValueError('默认根因不能删除，可以修改名称')
            options[:] = [item for item in options if item['id'] != option_id]
            config['notes'].setdefault(stage_id_value, {}).pop(option_id, None)
        else:
            raise ValueError('不支持的根因操作')
    elif kind == 'supplement':
        options = config['supplements'].setdefault(stage_id_value, [])
        if action == 'add':
            _validate_label(label, 160)
            option = {'id': uuid.uuid4().hex[:16], 'label': label.strip()}
            if any(item['label'] == option['label'] for item in options):
                raise ValueError('这个阶段已经有同名补充原因')
            options.append(option)
        elif action in ('rename', 'delete'):
            option = next((item for item in options if item['id'] == option_id), None)
            if option is None:
                raise ValueError('找不到当前阶段的补充原因')
            if action == 'delete':
                options.remove(option)
            else:
                _validate_label(label, 160)
                if any(item['id'] != option_id and item['label'] == label.strip() for item in options):
                    raise ValueError('这个阶段已经有同名补充原因')
                option['label'] = label.strip()
        else:
            raise ValueError('不支持的补充原因操作')
    elif kind == 'note':
        if cause_id not in {item['id'] for item in config['causes'].get(stage_id_value, [])}:
            raise ValueError('请先选择有效主要根因')
        note_map = config['notes'].setdefault(stage_id_value, {})
        options = note_map.setdefault(cause_id, [])
        if action == 'add':
            _validate_label(label, 160)
            option = {'id': uuid.uuid4().hex[:16], 'label': label.strip()}
            if any(item['label'] == option['label'] for item in options):
                raise ValueError('这个根因分类选项已经存在')
            options.append(option)
        elif action == 'delete':
            option = next((item for item in options if item['id'] == option_id), None)
            if option is None:
                raise ValueError('找不到要删除的根因分类')
            if option.get('default_code'):
                raise ValueError('默认选项不能删除，可以修改名称')
            options[:] = [item for item in options if item['id'] != option_id]
        elif action == 'rename':
            _validate_label(label, 160)
            option = next((item for item in options if item['id'] == option_id), None)
            if option is None:
                raise ValueError('找不到要修改的根因分类')
            if any(item['id'] != option_id and item['label'] == label.strip() for item in options):
                raise ValueError('这个根因分类名称已经存在')
            option['label'] = label.strip()
        else:
            raise ValueError('不支持的说明操作')
    else:
        raise ValueError('选项类型无效')
    _write_options(data)
    return stage_options(task)


def _validate_label(label, limit=80):
    if not isinstance(label, str) or not label.strip() or len(label.strip()) > limit:
        raise ValueError(f'选项名称需为1到{limit}个字符')


def validate(review, custom_stage_ids=(), custom_cause_ids=(), custom_note_ids=(),
             secondary_cause_ids=(), secondary_note_ids=()):
    if not isinstance(review, dict) or set(review)-FIELDS:
        raise ValueError('不支持的人工评价字段')
    result = {}
    for name in ('execution_quality', 'visual_progress'):
        value = review.get(name)
        if value is not None and (type(value) not in (int, float)
                or not math.isfinite(value) or not 0 <= value <= 100):
            raise ValueError('人工评分及视觉进度必须在 0–100 之间')
        result[name] = value
    for name, choices in (('behavior_class', BEHAVIORS), ('failure_stage', STAGES),
                          ('vlm_agreement', AGREEMENT)):
        value = review.get(name)
        if value is not None and (not isinstance(value, str) or value not in choices):
            raise ValueError('无效的人工评价分类：'+name)
        result[name] = value
    cause = review.get('root_cause')
    if cause is not None and (not isinstance(cause, str) or cause not in set(custom_cause_ids)):
        raise ValueError('所选主要根因不属于当前阶段')
    result['root_cause'] = cause
    cause_label = review.get('root_cause_label') or ''
    if not isinstance(cause_label, str) or len(cause_label) > 80:
        raise ValueError('主要根因名称格式无效')
    result['root_cause_label'] = cause_label.strip()
    specific_reason = review.get('specific_reason')
    if specific_reason is not None and (not isinstance(specific_reason, str)
            or specific_reason not in set(custom_note_ids)):
        raise ValueError('所选根因分类不属于当前根因')
    result['specific_reason'] = specific_reason
    specific_reason_label = review.get('specific_reason_label') or ''
    if not isinstance(specific_reason_label, str) or len(specific_reason_label) > 160:
        raise ValueError('根因分类名称格式无效')
    result['specific_reason_label'] = specific_reason_label.strip()
    secondary_cause = review.get('secondary_root_cause')
    secondary_specific = review.get('secondary_specific_reason')
    if secondary_cause and not secondary_specific:
        raise ValueError('旧版补充原因缺少分类')
    if secondary_cause is not None and (not isinstance(secondary_cause, str)
            or secondary_cause not in set(secondary_cause_ids)):
        raise ValueError('旧版补充原因的根因无效')
    if secondary_specific is not None and (not isinstance(secondary_specific, str)
            or secondary_specific not in set(secondary_note_ids)):
        raise ValueError('所选补充原因分类无效')
    result['secondary_root_cause'] = secondary_cause
    result['secondary_specific_reason'] = secondary_specific
    for name, label, limit in (
        ('secondary_root_cause_label', '旧版补充原因根因名称格式无效', 80),
        ('secondary_specific_reason_label', '补充原因名称格式无效', 160),
    ):
        value = review.get(name) or ''
        if not isinstance(value, str) or len(value) > limit:
            raise ValueError(label)
        result[name] = value.strip()
    stage = review.get('reached_stage')
    if stage is not None and (not isinstance(stage, str) or stage not in set(custom_stage_ids)):
        raise ValueError('所选失败阶段不属于当前 Task 的阶段选项')
    result['reached_stage'] = stage
    stage_label = review.get('stage_label') or ''
    if not isinstance(stage_label, str) or len(stage_label) > 80:
        raise ValueError('阶段名称格式无效')
    result['stage_label'] = stage_label.strip()
    for name in ('target_touched', 'stuck'):
        value = review.get(name)
        if value is not None and type(value) is not bool:
            raise ValueError(name+' 必须为是、否或未判断')
        result[name] = value
    reviewed = review.get('reviewed')
    if reviewed is not None and type(reviewed) is not bool:
        raise ValueError('reviewed 必须为布尔值')
    result['reviewed'] = reviewed
    meaningful = ('execution_quality', 'visual_progress', 'behavior_class', 'failure_stage',
                  'target_touched', 'stuck', 'reached_stage', 'root_cause',
                  'specific_reason')
    if not any(result.get(name) not in (None, '') for name in meaningful):
        raise ValueError('请至少填写一项评分、判断或备注')
    return result


def _path(run, identity):
    name = hashlib.sha256(identity['key'].encode()).hexdigest()
    return run/'human_reviews'/f'{name}.json'


def read(run, identity):
    path = _path(run, identity)
    return json.loads(path.read_text(encoding='utf-8')) if path.is_file() else None


def save(run, identity, review, custom_stage_ids=(), custom_cause_ids=(), custom_note_ids=(),
         secondary_cause_ids=(), secondary_note_ids=()):
    previous = read(run, identity)
    review = dict(review)
    if 'reviewed' not in review and previous:
        review['reviewed'] = previous.get('reviewed')
    fields = validate(review, custom_stage_ids, custom_cause_ids, custom_note_ids,
                      secondary_cause_ids, secondary_note_ids)
    now = datetime.now(timezone.utc).isoformat()
    record = dict(schema_version=1, source='human', **identity, **fields,
                  created_at=previous['created_at'] if previous else now, updated_at=now)
    path = _path(run, identity)
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(mode='w', encoding='utf-8', dir=path.parent,
                                     prefix='.', suffix='.pending', delete=False) as handle:
        temporary = Path(handle.name)
        try:
            json.dump(record, handle, ensure_ascii=False, indent=2, allow_nan=False)
            handle.write('\n')
            handle.close()
            temporary.replace(path)
        finally:
            temporary.unlink(missing_ok=True)
    return record


def set_reviewed(run, identity, reviewed):
    if type(reviewed) is not bool:
        raise ValueError('检验状态必须为布尔值')
    previous = read(run, identity)
    if previous:
        record = dict(previous, reviewed=reviewed,
                      updated_at=datetime.now(timezone.utc).isoformat())
    else:
        now = datetime.now(timezone.utc).isoformat()
        record = dict(schema_version=1, source='human', **identity,
                      reviewed=reviewed, created_at=now, updated_at=now)
    path = _path(run, identity)
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(mode='w', encoding='utf-8', dir=path.parent,
                                     prefix='.', suffix='.pending', delete=False) as handle:
        temporary = Path(handle.name)
        try:
            json.dump(record, handle, ensure_ascii=False, indent=2, allow_nan=False)
            handle.write('\n')
            handle.close()
            temporary.replace(path)
        finally:
            temporary.unlink(missing_ok=True)
    return record


def delete(run, identity):
    path = _path(run, identity)
    existed = path.is_file()
    path.unlink(missing_ok=True)
    return existed


def all_reviews(run):
    reviews = {}
    for path in (run/'human_reviews').glob('*.json'):
        try:
            record = json.loads(path.read_text(encoding='utf-8'))
            if record.get('source') == 'human' and isinstance(record.get('key'), str):
                reviews[record['key']] = record
        except (ValueError, OSError, AttributeError):
            continue
    return reviews


def is_annotated(record):
    if not isinstance(record, dict):
        return False
    return bool(record.get('reached_stage') and record.get('root_cause'))


def needs_update(record, stages):
    """Return whether a saved annotation points at removed or renamed options."""
    if not is_annotated(record):
        return False
    stage = next((item for item in stages if item['id'] == record.get('reached_stage')), None)
    if stage is None:
        return True
    saved_stage_label = record.get('stage_label')
    normalize_stage = lambda value: re.sub(r'^\s*\d{1,3}\s*[-.、:：)]?\s*', '', value).strip()
    if saved_stage_label and normalize_stage(saved_stage_label) != normalize_stage(stage['label']):
        return True
    cause = next((item for item in stage.get('causes', [])
                  if item['id'] == record.get('root_cause')), None)
    if cause is None:
        return True
    saved_label = record.get('root_cause_label')
    if saved_label and saved_label != cause['label']:
        return True
    specific_id = record.get('specific_reason')
    if not specific_id:
        return True
    specific = next((item for item in cause.get('notes', [])
                     if item['id'] == specific_id), None)
    if specific is None:
        return True
    saved_specific_label = record.get('specific_reason_label')
    if saved_specific_label and saved_specific_label != specific['label']:
        return True
    secondary_cause_id = record.get('secondary_root_cause')
    secondary_specific_id = record.get('secondary_specific_reason')
    if secondary_cause_id and not secondary_specific_id:
        return True
    if not secondary_specific_id:
        return False
    if secondary_cause_id:
        secondary_cause = next((item for item in stage.get('causes', [])
                                if item['id'] == secondary_cause_id), None)
        if secondary_cause is None:
            return True
        secondary_specific = next((item for item in secondary_cause.get('notes', [])
                                   if item['id'] == secondary_specific_id), None)
        saved_secondary_cause = record.get('secondary_root_cause_label')
        if saved_secondary_cause and saved_secondary_cause != secondary_cause['label']:
            return True
    else:
        secondary_specific = next((item for item in stage.get('supplements', [])
                                   if item['id'] == secondary_specific_id), None)
    if secondary_specific is None:
        return True
    saved_secondary_specific = record.get('secondary_specific_reason_label')
    return bool(saved_secondary_specific and saved_secondary_specific != secondary_specific['label'])
