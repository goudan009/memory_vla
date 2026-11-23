"""Read the task and progress documents belonging to each experiment."""

from datetime import datetime, timezone
from graphlib import TopologicalSorter
import json
import re


def experiment_names(root):
    if not root.is_dir():
        return []
    root = root.resolve()
    return sorted(
        (path.name for path in root.iterdir()
         if re.fullmatch(r'experiment_[0-9]+', path.name) and path.is_dir()
         and path.resolve() == path),
        key=lambda name: (int(name.removeprefix('experiment_')), name),
    )


def experiment_catalog(output_root):
    output_root = output_root.resolve()
    catalog = {}
    for root in sorted(output_root.glob('test_[0-9]*')):
        if not re.fullmatch(r'test_[0-9]+', root.name) or root.resolve() != root:
            continue
        for name in experiment_names(root):
            # Keep the original Test 01 links addressable by their short IDs.
            key = name if root.name == 'test_01' else f'{root.name}/{name}'
            catalog[key] = (root, name)
    return catalog


def document(directory, *names):
    for name in names:
        path = directory / name
        if path.is_file() and path.resolve().parent == directory.resolve():
            with path.open(encoding='utf-8') as source:
                content = source.read(128 * 1024 + 1)
                updated = datetime.fromtimestamp(path.stat().st_mtime, timezone.utc).isoformat(timespec='seconds')
            if not content.strip():
                continue
            return {'file': name, 'markdown': content[:128 * 1024], 'updated_at': updated,
                    'truncated': len(content) > 128 * 1024}
    return None


def validate_plan(plan):
    if not isinstance(plan, dict) or any(not isinstance(plan.get(key), str) or not plan[key].strip()
                                         for key in ('goal', 'reason', 'created_at', 'author')):
        raise ValueError('缺少目标、变更原因、日期或作者')
    nodes = plan.get('nodes')
    if not isinstance(nodes, list) or not nodes:
        raise ValueError('至少需要一个步骤')
    dependencies = {}
    for node in nodes:
        if not isinstance(node, dict) or any(not isinstance(node.get(key), str) or not node[key].strip()
                                             for key in ('id', 'title', 'acceptance')):
            raise ValueError('每个步骤都需要 ID、标题和验收条件')
        key = node['id']
        if not re.fullmatch(r'[a-z][a-z0-9-]*', key) or key in dependencies:
            raise ValueError(f'步骤 ID 无效或重复：{key}')
        after = node.get('after')
        if not isinstance(after, list) or any(not isinstance(dep, str) for dep in after) or len(set(after)) != len(after):
            raise ValueError(f'依赖格式无效：{key}')
        record = node.get('record', node['title'])
        if not isinstance(record, str) or not record.strip():
            raise ValueError(f'进度记录名称无效：{key}')
        gate = node.get('review_experiment')
        if gate is not None and (not isinstance(gate, str) or not re.fullmatch(r'experiment_[0-9]+', gate)):
            raise ValueError(f'前置实验编号无效：{key}')
        dependencies[key] = after
    if any(dep not in dependencies for after in dependencies.values() for dep in after):
        raise ValueError('有依赖指向不存在的步骤')
    list(TopologicalSorter(dependencies).static_order())
    mainline = plan.get('mainline')
    if not isinstance(mainline, list) or not mainline or any(
            not isinstance(key, str) or key not in dependencies for key in mainline):
        raise ValueError('主线必须包含有效步骤')
    if any(left not in dependencies[right] for left, right in zip(mainline, mainline[1:])):
        raise ValueError('相邻主线步骤必须直接相依')


def workflow_plans(directory):
    folder = directory / 'workflow'
    if not folder.is_dir() or folder.resolve().parent != directory.resolve():
        return {'versions': [], 'errors': []}
    versions, errors = [], []
    for path in sorted(folder.glob('plan-v[0-9][0-9][0-9].json')):
        try:
            if not path.is_file() or path.resolve().parent != folder.resolve():
                raise ValueError('计划必须位于本实验的 workflow 目录')
            plan = json.loads(path.read_text(encoding='utf-8'))
            validate_plan(plan)
        except (OSError, ValueError) as error:
            errors.append(f'{path.name} · {error}')
            continue
        versions.append({**plan, 'version': path.stem.removeprefix('plan-')})
    return {'versions': versions, 'errors': errors}


def experiment_record(root, name):
    directory = root / name
    task_names = ['task.md', 'TASK.md']
    if name == 'experiment_07':
        task_names.append('Task40_Experiment07_Master_Taskbook.md')
    task_names.extend(path.name for path in sorted(directory.glob('Experiment*_Agent任务书.md')))
    workflow = workflow_plans(directory)
    gates = {}
    if workflow['versions']:
        for node in workflow['versions'][-1]['nodes']:
            upstream = node.get('review_experiment')
            if not upstream or upstream == name or (root / upstream).resolve() != root.resolve() / upstream:
                continue
            report = document(root / upstream, 'report.md')
            marker = f'EXPERIMENT_{upstream.removeprefix("experiment_")}_ACCEPTED'
            matches = re.findall(rf'^\s*`?{marker}=(true|false)`?\s*$', report['markdown'], re.M) if report else []
            gates[upstream] = matches[-1] if matches else None
    return {'name': name, 'path': f'Output/{root.name}/{name}',
            'task': document(directory, *task_names),
            'progress': document(directory, 'progress.md', 'PROGRESS.md'),
            'result': document(directory, 'result.md'),
            'report': document(directory, 'report.md'),
            'workflow': workflow, 'gates': gates}
