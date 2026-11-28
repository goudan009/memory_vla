"""On-demand local/SSH resource browsing; no recursive scan during navigation."""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import re
import shlex
import shutil
import subprocess
import sys
import threading
import time
import uuid


def clean_path(value):
    value = str(value or '')
    if ':/' in value:
        value = '/' + value.split(':/', 1)[1]
    return value


def checked_path(value, roots):
    path = Path(clean_path(value))
    if not path.is_absolute() or any(part in ('.workbench-trash','.workbench-downloads') for part in path.parts):
        raise ValueError('请选择资源范围内的绝对路径')
    resolved = path.resolve()
    if not any(resolved.is_relative_to(root.resolve()) for root in roots):
        raise ValueError('路径超出资源范围')
    return path


def disk_info(path):
    mount = path.resolve()
    if not mount.is_dir():
        mount = mount.parent
    while not os.path.ismount(mount) and mount != mount.parent:
        mount = mount.parent
    usage = shutil.disk_usage(path)
    return dict(mount=str(mount), total=usage.total, used=usage.used, free=usage.free,
                percent=round(usage.used/usage.total*100, 1), checked_at=time.time())


def entry(path):
    stat = path.lstat()
    directory = path.is_dir()
    return dict(name=path.name, path=str(path), type='directory' if directory else 'file',
                size=None if directory else stat.st_size, modified=stat.st_mtime,
                is_symlink=path.is_symlink(), link_target=str(path.resolve()) if path.is_symlink() else None)


def operate(request):
    roots = [Path(value) for value in request['roots']]
    action = request.get('action', 'browse')
    if action == 'disks':
        disks = {}
        for root in roots:
            if root.exists():
                info = disk_info(root)
                disks[info['mount']] = info
        return dict(disks=list(disks.values()), roots=[str(p) for p in roots if p.is_dir()])
    path = checked_path(request.get('path'), roots)
    if action == 'restore':
        token = request.get('token', '')
        if not re.fullmatch(r'[a-f0-9]{32}', token):
            raise ValueError('回收记录不正确')
        trash = path.parent/'.workbench-trash'
        if trash.is_symlink():
            raise ValueError('回收目录不能是软链接')
        metadata = trash/(token+'.json')
        record = json.loads(metadata.read_text())
        if record['path'] != str(path):
            raise ValueError('回收记录与路径不匹配')
        if os.path.lexists(path):
            raise ValueError('原路径已存在，不能覆盖新数据')
        (trash/token).rename(path)
        metadata.unlink()
        return dict(restored=str(path))
    if not path.exists():
        raise ValueError('路径不存在或设备尚未挂载')
    if action == 'mkdir':
        name = request.get('name', '')
        if not isinstance(name, str) or name in ('', '.', '..', '.workbench-trash') or '/' in name or '\x00' in name or len(name)>200:
            raise ValueError('请输入有效的文件夹名称，不含路径分隔符')
        target = checked_path(str(path/name), roots)
        target.mkdir()
        return entry(target)
    if action == 'trash':
        if path.is_symlink():
            raise ValueError('软链接请由管理员核对后处理')
        # Managed training weights must pass the existing dependency checks.
        if any(part.startswith(('checkpoint-', 'step-')) or part=='training' for part in path.parts):
            raise ValueError('训练权重请通过训练详情的受保护 checkpoint 删除入口')
        protected = [Path(p).resolve() for p in request.get('protected', [])]
        if path.resolve() in protected or len(path.resolve().parts) <= 3:
            raise ValueError('不能删除存储根目录，请选择其中的资源')
        trash = path.parent/'.workbench-trash'
        if trash.is_symlink():
            raise ValueError('回收目录不能是软链接')
        trash.mkdir(exist_ok=True)
        token = uuid.uuid4().hex
        metadata = trash/(token+'.json')
        metadata.write_text(json.dumps(dict(path=str(path), deleted_at=time.time()), ensure_ascii=False))
        try:
            path.rename(trash/token)
        except OSError:
            metadata.unlink()
            raise
        return dict(path=str(path), trash_path=str(trash/token), token=token)
    if action == 'size':
        result = subprocess.run(['du', '-sb', '--', str(path)], capture_output=True, text=True, timeout=20, check=True)
        return dict(size=int(result.stdout.split()[0]))
    if action not in ('browse', 'stat'):
        raise ValueError('未知的资源操作')
    result = entry(path)
    result['disk'] = disk_info(path)
    result['can_delete'] = (path.resolve() not in [Path(p).resolve() for p in request.get('protected', [])]
                            and len(path.resolve().parts)>3 and not path.is_symlink()
                            and not any(part.startswith(('checkpoint-','step-')) or part=='training' for part in path.parts))
    if action == 'stat':
        return result
    parent = path.parent
    result['parent'] = str(parent) if any(parent.resolve().is_relative_to(root.resolve()) for root in roots) else None
    if path.is_dir():
        query = str(request.get('query', '')).casefold()
        children = [child for child in path.iterdir() if child.name not in ('.workbench-trash','.workbench-downloads') and query in child.name.casefold()]
        children.sort(key=lambda child: (not child.is_dir(), child.name.casefold()))
        offset, limit = max(0, int(request.get('offset', 0))), min(200, max(1, int(request.get('limit', 100))))
        rows = []
        for child in children[offset:offset+limit]:
            try:
                rows.append(entry(child))
            except OSError:
                rows.append(dict(name=child.name, path=str(child), type='unavailable', size=None))
        result.update(entries=rows, total=len(children), offset=offset, has_more=offset+limit<len(children))
    else:
        with path.open('rb') as stream:
            content = stream.read(65_537)
        try:
            preview = content[:65_536].decode('utf-8') if b'\x00' not in content else None
        except UnicodeDecodeError:
            preview = None
        result.update(preview=preview, truncated=len(content)>65_536)
    return result


class StorageService:
    def __init__(self, project, devices, catalog):
        self.project = Path(project)
        self.devices = devices
        self.catalog = catalog
        self.record = self.project/'outputs/registry_resources.json'
        self.lock = threading.Lock()

    def registered(self):
        try:
            return json.loads(self.record.read_text())
        except FileNotFoundError:
            return []

    def hosts(self):
        return [{'host':'4090','label':'4090 · 本机'}] + [
            dict(host='A100' if key=='a100' else 'A6000' if key=='a6000' else key,
                 label=device.get('label',key)) for key,device in self.devices.items()]

    def request(self, host, payload):
        device = next((value for key,value in self.devices.items() if host == {'a100':'A100','a6000':'A6000'}.get(key,key)), None)
        if host != '4090' and device is None:
            raise ValueError('未知存放设备')
        project = str(self.project) if host=='4090' else device.get('paths',{}).get('project', '/nvme3/projects/memory_vla')
        paths = (device or {}).get('paths',{})
        defaults = [project+'/assets/datasets',project+'/checkpoints', paths.get('training',project+'/training'),
                    paths.get('datasets','/data/datasets'), '/data/models','/nvme3/datasets','/nvme3/models',
                    '/nvme3/huggingface',str(Path(project).parent.parent/'datasets'),str(Path(project).parent.parent/'models')]
        items = self.catalog().get('items', []) + self.registered()
        roots = list(dict.fromkeys(defaults + [clean_path(item['path']) for item in items
                    if item.get('host')==host and item.get('path') and item.get('kind') in ('dataset','weights')]))
        request = {**payload, 'roots':roots, 'protected':defaults}
        action = payload.get('action','browse')
        if action == 'register':
            # Validate an existing folder on the selected device before saving its index entry.
            raw = Path(clean_path(payload.get('path')))
            if not raw.is_absolute() or len(raw.parts)<4 or '..' in raw.parts:
                raise ValueError('请填写数据或权重目录的完整路径')
            if not any(raw.is_relative_to(Path(base)) for base in [project,'/data','/nvme1','/nvme2','/nvme3','/mnt/data', '/home']):
                raise ValueError('目录必须位于数据存储范围')
            request.update(action='stat', roots=roots+[str(raw)])
        if host == '4090':
            result = operate(request)
        else:
            source = Path(__file__).read_text()
            completed = subprocess.run(['ssh','-o','BatchMode=yes','-o','ConnectTimeout=5', device['host'],
                                        'python3 -c '+shlex.quote(source)], input=json.dumps(request), text=True,
                                       capture_output=True, timeout=30)
            if completed.returncode:
                raise ValueError((completed.stderr or '设备离线或暂时不可达').strip()[-1000:])
            result = json.loads(completed.stdout)
            if result.get('error'):
                raise ValueError(result['error'])
        if action == 'register':
            if result['type'] != 'directory' or payload.get('kind') not in ('dataset','weights'):
                raise ValueError('请选择数据集或模型权重文件夹')
            item = dict(id='resource:'+hashlib.sha256((host+str(raw)).encode()).hexdigest()[:20],
                        host=host, path=str(raw), kind=payload['kind'], name=str(payload.get('name') or raw.name)[:200],
                        registered=True, updated=time.strftime('%Y-%m-%d %H:%M:%S'))
            with self.lock:
                saved = [row for row in self.registered() if (row['host'],clean_path(row['path'])) != (host,str(raw))]
                self.record.parent.mkdir(parents=True,exist_ok=True)
                temp=self.record.with_suffix('.pending')
                temp.write_text(json.dumps(saved+[item],ensure_ascii=False,indent=2))
                temp.replace(self.record)
            result={'item':item}
        return {**result,'host':host}


if __name__ == '__main__':
    try:
        print(json.dumps(operate(json.load(sys.stdin)),ensure_ascii=False))
    except (OSError, ValueError, subprocess.SubprocessError) as error:
        print(json.dumps({'error':str(error)},ensure_ascii=False))
