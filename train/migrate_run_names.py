"""Migrate existing run directories, retaining paths used by live processes."""

import argparse
import datetime
import json
import os
from pathlib import Path
import shutil
import signal
import time


RENAMES = {
    '20261110_1745_task0_baseline_ddp_s42': 'V_1_bas_task0',
    '20261110_2005_task0_memory_v1_ddp_s42': 'V_1_mem_task0',
    'V_1_task34_bas': 'V_1_bas_task34',
    'V_1_task34_mem': 'V_1_mem_task34',
    'V_1_task40_bas': 'V_1_bas_task40',
    'V_1_task40_mem': 'V_1_mem_task40',
}
PAIR = '20261109_2100_task21_ddp_async_s42'


def read(path, default=None):
    return json.loads(path.read_text()) if path.exists() else default


def save(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    pending = path.with_suffix('.pending')
    pending.write_text(json.dumps(value, indent=2, ensure_ascii=False))
    pending.replace(path)


def references(path):
    """Inspect command arguments and working directories, never shell text."""
    result = []
    for entry in Path('/proc').iterdir():
        if not entry.name.isdigit() or int(entry.name) == os.getpid():
            continue
        try:
            stat = (entry/'stat').read_text().rsplit(')', 1)[1].split()
            if stat[0] in ('Z', 'X'):
                continue
            args = (entry/'cmdline').read_bytes().decode(errors='replace').split(chr(0))
            cwd = os.readlink(entry/'cwd')
            prefix = str(path)
            if any(a == prefix or a.startswith(prefix+'/') for a in args) or cwd == prefix or cwd.startswith(prefix+'/'):
                result.append(int(entry.name))
        except (OSError, ValueError):
            pass
    return result


def replace_paths(value, mapping):
    if isinstance(value, str):
        for old, new in sorted(mapping.items(), key=lambda item:len(item[0]), reverse=True):
            # Replacement strings never contain a source prefix.
            value = value.replace(old, new)
        return value
    if isinstance(value, list):
        return [replace_paths(v, mapping) for v in value]
    if isinstance(value, dict):
        return {k:replace_paths(v, mapping) for k,v in value.items()}
    return value


def update_json(root, mapping):
    mapping = dict(mapping)
    # Local mirrors contain both local-relative and A100-absolute run paths.
    for old, new in list(mapping.items()):
        marker = 'v1_groot_memoryvla/training/'
        if marker in old and marker in new:
            mapping[marker+old.split(marker, 1)[1]] = marker+new.split(marker, 1)[1]
    for path in root.rglob('*.json'):
        if path.is_symlink() or any(parent.is_symlink() for parent in path.parents if parent != root.parent):
            continue
        value = read(path)
        updated = replace_paths(value, mapping)
        if updated != value:
            save(path, updated)


def split_pair(training):
    old = training/PAIR
    if not old.is_dir():
        return
    if references(old):
        raise RuntimeError('Task21 paired run still has live references; no split performed')
    memory, baseline = training/'V_1_mem_task21', training/'V_1_bas_task21'
    if memory.exists() or baseline.exists():
        raise FileExistsError('Task21 destination already exists')
    baseline.mkdir()
    for child in old.iterdir():
        if child.name in ('memory_v1', 'baseline'):
            continue
        if child.is_dir():
            shutil.copytree(child, baseline/child.name)
        else:
            shutil.copy2(child, baseline/child.name)
    (old/'baseline').rename(baseline/'baseline')
    old.rename(memory)
    for root, arm in [(memory, 'memory_v1'), (baseline, 'baseline')]:
        mapping = {str(training/PAIR/'memory_v1'):str(memory/'memory_v1'),
                   str(training/PAIR/'baseline'):str(baseline/'baseline'), str(training/PAIR):str(root)}
        update_json(root, mapping)
        for name in ('ddp_launch.json', 'ddp_resume_step-001000.json'):
            path = root/'config'/name
            config = read(path)
            if config and 'commands' in config:
                config['commands'] = {k:v for k,v in config['commands'].items() if k == arm}
                save(path, config)
        for name in ('status.json', 'selected_checkpoints.json'):
            path = root/name
            value = read(path)
            if isinstance(value, dict):
                if 'selected_checkpoints' in value:
                    value['selected_checkpoints'] = {k:v for k,v in value['selected_checkpoints'].items() if k == arm}
                elif set(value) <= {'baseline', 'memory_v1'}:
                    value = {k:v for k,v in value.items() if k == arm}
                save(path, value)


def migrate(training, record):
    training = Path(training).resolve()
    split_pair(training)
    state = read(record, dict(migrations={}, aliases={}))
    for old_name, new_name in RENAMES.items():
        old, new = training/old_name, training/new_name
        if old.is_dir() and not old.is_symlink():
            if new.exists():
                raise FileExistsError(str(new))
            # Pause only processes addressing this directory across rename+link.
            # Resume in finally; neither training nor model services are killed.
            paused = []
            try:
                for pid in references(old):
                    try:
                        os.kill(pid, signal.SIGSTOP)
                        paused.append(pid)
                    except ProcessLookupError:
                        pass
                old.rename(new)
                old.symlink_to(new.name, target_is_directory=True)
            finally:
                for pid in paused:
                    try:
                        os.kill(pid, signal.SIGCONT)
                    except ProcessLookupError:
                        pass
            state['migrations'][old_name] = new_name
            # Mutable metadata used by running controllers keeps its old paths
            # until the compatibility link can be retired.
        if old.is_symlink() and old.resolve() == new and new.is_dir():
            active = references(old)
            queue = read(training/'queues/V_1_task34_40/status.json', {})
            queue_active = queue.get('status') not in ('completed', 'failed', None)
            # The existing queue holds old output paths in memory, including
            # the previous Task0 run and future Task40 launch names.
            hold = queue_active and (old_name.startswith('V_1_task') or '2005_task0' in old_name)
            if not active and not hold:
                update_json(new, {str(old):str(new)})
                old.unlink()
            else:
                state['aliases'][old_name] = dict(target=new_name, pids=active, queue=hold)
                continue
            state['aliases'].pop(old_name, None)
        elif new.is_dir() and not old.exists():
            state['aliases'].pop(old_name, None)
    state.update(updated_at=datetime.datetime.now().isoformat())
    save(record, state)
    return state


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--training', type=Path, required=True)
    parser.add_argument('--record', type=Path, required=True)
    parser.add_argument('--watch', action='store_true')
    args = parser.parse_args()
    while True:
        result = migrate(args.training, args.record)
        if not args.watch:
            print(json.dumps(result, ensure_ascii=False))
            break
        # Include future queue launches. Stop once the queue and all aliases end.
        queue = read(args.training/'queues/V_1_task34_40/status.json', {})
        if not result['aliases'] and queue.get('status') in ('completed', 'failed', None):
            break
        time.sleep(10)
