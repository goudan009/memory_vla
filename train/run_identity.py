"""Shared experiment identities; stdlib only so the same code can run over SSH."""
import json
from pathlib import Path
import re

NAMING_SCHEMA = 'groot-runs-v3'
# Register a method version only after its scientific meaning is agreed with
# the user. Registration does not authorize training or change a recipe.
MODEL_VERSIONS = {
    'V_0_0': dict(prefix='v0_0', backend='b1k2026', benchmark='b1k2026', display_name='V0_0',
                  family='V0', description='B1K-2026 / V0_0：GR00T N1.7 原生无记忆基线；V0 系列允许代码及训练参数小版本对照',
                  launch_ready=False),
    'V_1': dict(prefix='v1', backend='v1', description='V1 方法及其已记录的历史适配'),
    'V_2': dict(prefix='v2', backend='v1', description='历史 V2；新任务仍须锁定具体方法配置'),
    'V_2_1': dict(prefix='v2_1', backend='v2_1', description='V2.1 方法；使用专用训练入口'),
    'V_1_2': dict(prefix='v1_2', backend='v1_2', description='GR00T N1.7 原生无记忆基线；独立环境', launch_ready=False),
}
RUN_PATTERN = r'v[0-9]+(?:_[0-9]+)*__test[0-9]{2}__exp[0-9]{2}__run(?!0{3,}(?:__|$))[0-9]{3,}'
EXPERIMENT_12_RUN = re.compile(r'v1__test01__exp12__run(?!000)[0-9]{3}')
EXPERIMENT_13_RUN = re.compile(r'v1__test01__exp13__run(?!000)[0-9]{3}')
CANONICAL_RUN = re.compile(RUN_PATTERN)
V2_1_RUN = re.compile(r'v2_1__test[0-9]{2}__exp[0-9]{2}__run(?!000)[0-9]{3}(?:__preflight[0-9]{3})?')
LEGACY_V21_RUN = re.compile(r'v2\.1__test01__exp14__run(?!000)[0-9]{3}(?:__preflight[0-9]{3})?')
# Keep the pre-rename paths readable without using them for new work.
V21_RUN = re.compile(r'(?:' + V2_1_RUN.pattern + r'|' + LEGACY_V21_RUN.pattern + r')')
# These two cancelled Exp12 preflights are evidenced in the experiment record.
# Do not reinterpret arbitrary historical V_12 (or V_10/V_11) names.
LEGACY_IDENTITIES = {
    name: dict(model_version='V_1', test_id='test01', experiment_id='exp12')
    for name in ('V_12_memory_task40_train01', 'V_12_memory_task40_train02')
}


def canonical_version(value):
    """Normalize model version aliases without changing stable run/path IDs."""
    if isinstance(value, str):
        match = re.fullmatch(r'(?:[vV]_?)?([0-9]+(?:[_.][0-9]+)*)', value)
        if match:
            return 'V_' + match[1].replace('.', '_')
    return value


def registered_version(value, backend=None):
    version = canonical_version(value)
    if not isinstance(version, str) or version not in MODEL_VERSIONS:
        raise ValueError('训练标识的模型版本未登记；实验编号不能作为版本号。先确认方法版本并在 train/run_identity.py 登记')
    if backend and MODEL_VERSIONS[version]['backend'] != backend:
        raise ValueError('模型版本与训练入口不匹配')
    return version


def run_fields(run):
    """Parse stable IDs, including the historical dotted V2.1 spelling."""
    pattern = RUN_PATTERN + r'(?:__preflight[0-9]{3,})?'
    normalized = run.replace('v2.1__', 'v2_1__', 1) if isinstance(run, str) else ''
    if not re.fullmatch(pattern, normalized):
        raise ValueError('训练标识须为 v版本__testNN__expNN__runNNN；例如 v2_1__test02__exp01__run001')
    parts = normalized.split('__')
    return dict(model_version=canonical_version(parts[0]), test_id=parts[1], experiment_id=parts[2],
                training_run=parts[3], train_number=int(parts[3][3:]))


def validate_run_identity(run, version, preflight=False):
    version = registered_version(version)
    fields = run_fields(run)
    if fields['model_version'] != version:
        raise ValueError('训练标识中的模型版本与 version 不一致；实验编号须单独写入 expNN')
    if ('__preflight' in run) != preflight or run.startswith('v2.1__'):
        raise ValueError('新训练标识使用规范版本前缀；预检才可追加 __preflightNNN')
    return fields


def validate_run_namespace(version, test_id, experiment_id):
    """Validate the independently supplied method/test/experiment dimensions."""
    version = registered_version(version)
    if not isinstance(test_id, str) or not re.fullmatch(r'test[0-9]{2}', test_id):
        raise ValueError('test_id须为testNN；不得从目录或实验编号推测')
    if not isinstance(experiment_id, str) or not re.fullmatch(r'exp[0-9]{2}', experiment_id):
        raise ValueError('experiment_id须为expNN；不得从模型版本推测')
    return version, test_id, experiment_id


def checkpoint_step(name, run):
    if CANONICAL_RUN.fullmatch(run) or V21_RUN.fullmatch(run):
        match = re.fullmatch(re.escape(run) + r'__step([0-9]{6,})', name)
    else:
        match = re.fullmatch(r'step-([0-9]+)', name)
    return int(match[1]) if match else None


def method_name(arm, method=None):
    default = 'memory' if arm in ('memory', 'memory_v1', 'mem') else 'baseline'
    value = method or default
    value = {'mem': 'memory', 'memory_v1': 'memory', 'bas': 'baseline'}.get(value, value)
    if value not in ('baseline', 'memory', 'official') and not re.fullmatch(r'ablation_[a-z0-9]+(?:_[a-z0-9]+)*', value):
        raise ValueError('方法名应为 baseline、memory 或 ablation_模块名')
    return value


def training_base(version, task, method, *, legacy=False):
    if not legacy:
        registered_version(version)
        raise ValueError('新训练标识必须显式包含 test_id、experiment_id 和 run_number')
    version = canonical_version(version)
    return f'{version}_{method}_task{task}'


def read_identity_json(path):
    try:
        return json.loads(Path(path).read_text())
    except (OSError, ValueError):
        return {}


def training_identity(root, arm, task=None):
    root = Path(root)
    launch = read_identity_json(root/'config/web_launch.json')
    match = re.match(r'^(V_[0-9]+(?:_[0-9]+)*)_', root.name)
    version = canonical_version(launch.get('version') or (match[1] if match else 'V_1'))
    split = read_identity_json(root/'config/data_split.json')
    task = task if task is not None else ('_'.join(map(str, split['tasks']))
        if split.get('tasks') else split.get('task'))
    if task is None:
        parsed = re.search(r'(?:^|_)task([0-9]+)(?:_|$)', root.name)
        task = int(parsed[1]) if parsed else None
    method = method_name(arm, launch.get('method'))
    parsed = re.search(r'_train([0-9]+)$', root.name)
    legacy = re.search(r'_r([0-9]+)$', root.name)
    number = launch.get('train_number') or (int(parsed[1]) if parsed else int(legacy[1]) if legacy else 1)
    fields = {}
    if CANONICAL_RUN.fullmatch(root.name) or V21_RUN.fullmatch(root.name):
        fields = run_fields(root.name)
        version, number = fields['model_version'], fields['train_number']
    if V2_1_RUN.fullmatch(root.name):
        version = 'V_2_1'
        number = int(re.search(r'__run([0-9]{3})', root.name)[1])
        task = launch.get('task', 40)
    elif LEGACY_V21_RUN.fullmatch(root.name):
        version = 'V_2_1'
        number = int(re.search(r'__run([0-9]{3})', root.name)[1])
        task = launch.get('task', 40)
    correction = LEGACY_IDENTITIES.get(root.name)
    if correction:
        fields.update(correction, legacy_version=version, identity_status='corrected_legacy')
        version = correction['model_version']
        fields['display_name'] = f'{version} · test01 / exp12 · 历史预检 {number:02d}'
    manifest = read_identity_json(root/'config/source_manifest.json') or launch.get('source_manifest') or {}
    process = read_identity_json(root/'config/web_process.json')
    extension = read_identity_json(root/'config/extension_5000.json')
    config = read_identity_json(root/'config/run_config.json')
    locations = read_identity_json(root/'config/remote_locations.json')
    source = (process.get('source') or launch.get('source') or extension.get('source') or config.get('source') or
              locations.get('memory_source' if arm == 'memory_v1' else 'baseline_controller_source') or locations.get('source'))
    if source:
        manifest = read_identity_json(Path(source)/'web_source_manifest.json') or manifest
    # Historical IDs retain their real directory name. Never manufacture a
    # new training ID for an old paired run or overwrite its original records.
    fields.pop('train_number', None)
    return dict(version=version, method=method, train_number=number, training_id=root.name,
                task_id=task, method_description=launch.get('method_description',''), source_dir=source, source_digest=manifest.get('digest') or process.get('source_digest'),
                training_seed=launch.get('seed', config.get('seed')), naming_schema=launch.get('naming_schema', 'legacy'),
                **dict({'model_version': version}, **fields))


def reserve_training(training_root, deployments, version, *, test_id, experiment_id):
    """Allocate a canonical run ID in an explicit test/experiment namespace."""
    version, test_id, experiment_id = validate_run_namespace(version, test_id, experiment_id)
    training_root, deployments = Path(training_root), Path(deployments)
    training_root.mkdir(parents=True, exist_ok=True)
    prefix = MODEL_VERSIONS[version]['prefix']
    used = []
    locations = [training_root, deployments]
    trash = training_root/'.trash'
    if trash.is_dir() and not trash.is_symlink():
        locations.append(trash)
    for location in locations:
        if not location.is_dir() or location.is_symlink():
            continue
        for path in location.iterdir():
            if not path.is_dir() or path.is_symlink():
                continue
            try:
                fields = run_fields(path.name)
            except ValueError:
                continue
            if (fields['model_version'], fields['test_id'], fields['experiment_id']) == (version, test_id, experiment_id):
                used.append(fields['train_number'])
    number = max(used, default=0) + 1
    while True:
        name = f'{prefix}__{test_id}__{experiment_id}__run{number:03d}'
        if not any((location/name).exists() for location in locations):
            try:
                (training_root/name).mkdir()
                return name, number
            except FileExistsError:
                pass
        number += 1


def identity_bundle():
    """Install this exact module in a standalone remote Python invocation."""
    code = Path(__file__).read_text()
    return ('import sys, types\n'
            '_identity_module = types.ModuleType("run_identity")\n'
            'sys.modules["run_identity"] = _identity_module\n'
            'exec('+repr(code)+', _identity_module.__dict__)\n')
