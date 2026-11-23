"""Shared experiment identities; stdlib only so the same code can run over SSH."""
import json
from pathlib import Path
import re

NAMING_SCHEMA = 'groot-runs-v2'
EXPERIMENT_12_RUN = re.compile(r'v1__test01__exp12__run(?!000)[0-9]{3}')


def checkpoint_step(name, run):
    if EXPERIMENT_12_RUN.fullmatch(run):
        match = re.fullmatch(re.escape(run) + r'__step([0-9]{6})', name)
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


def training_base(version, task, method):
    if not isinstance(version, str) or not re.fullmatch(r'V_[0-9]+', version):
        raise ValueError('实验版本请使用 V_1、V_2 等格式')
    return f'{version}_{method}_task{task}'


def read_identity_json(path):
    try:
        return json.loads(Path(path).read_text())
    except (OSError, ValueError):
        return {}


def training_identity(root, arm, task=None):
    root = Path(root)
    launch = read_identity_json(root/'config/web_launch.json')
    match = re.match(r'^(V_[0-9]+)_', root.name)
    version = launch.get('version') or (match[1] if match else 'V_1')
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
    if EXPERIMENT_12_RUN.fullmatch(root.name):
        version, number = 'V_1', int(root.name[-3:])
    manifest = read_identity_json(root/'config/source_manifest.json') or launch.get('source_manifest') or {}
    process = read_identity_json(root/'config/web_process.json')
    extension = read_identity_json(root/'config/extension_5000.json')
    config = read_identity_json(root/'config/run_config.json')
    locations = read_identity_json(root/'config/remote_locations.json')
    source = (process.get('source') or launch.get('source') or extension.get('source') or config.get('source') or
              locations.get('memory_source' if arm == 'memory_v1' else 'baseline_controller_source') or locations.get('source'))
    # Historical IDs retain their real directory name. Never manufacture a
    # new training ID for an old paired run or overwrite its original records.
    return dict(version=version, method=method, train_number=number, training_id=root.name,
                task_id=task, method_description=launch.get('method_description',''), source_dir=source, source_digest=manifest.get('digest') or process.get('source_digest'),
                training_seed=launch.get('seed', config.get('seed')), naming_schema=launch.get('naming_schema', 'legacy'))


def reserve_training(training_root, deployments, version, task, method):
    training_root, deployments = Path(training_root), Path(deployments)
    training_root.mkdir(parents=True, exist_ok=True)
    base = training_base(version, task, method)
    used = []
    run_dirs = list(training_root.iterdir())
    trash = training_root/'.trash'
    if trash.is_dir() and not trash.is_symlink():
        run_dirs.extend(trash.iterdir())
    for path in run_dirs:
        if path.is_dir() and not path.is_symlink() and not EXPERIMENT_12_RUN.fullmatch(path.name):
            for arm in ('baseline', 'memory_v1'):
                if not (path/arm).is_dir() and not (path/'config/web_launch.json').is_file():
                    continue
                identity = training_identity(path, arm)
                if (identity['version'], identity['method'], identity['task_id']) == (version, method, task):
                    used.append(identity['train_number'])
    number = max(used, default=0) + 1
    while True:
        name = f'{base}_train{number:02d}'
        if not (deployments/name).exists() and not (trash/name).exists():
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
