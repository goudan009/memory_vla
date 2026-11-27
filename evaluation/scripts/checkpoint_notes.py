"""Workbench metadata for individual weights; never writes into model directories."""
import json
import os
from pathlib import Path, PurePosixPath
import threading
from datetime import datetime, timezone


class CheckpointNotes:
    def __init__(self, path):
        self.path = Path(path)
        self.lock = threading.Lock()

    def read(self):
        try:
            data = json.loads(self.path.read_text(encoding='utf-8'))
        except FileNotFoundError:
            return {}
        if not isinstance(data, dict):
            raise ValueError('checkpoint 备注文件格式不正确')
        return data

    def save(self, request, devices):
        if not isinstance(request, dict) or set(request) != {'device', 'checkpoint', 'notes'}:
            raise ValueError('请提供设备、checkpoint 路径和备注')
        device, checkpoint, notes = (request[k] for k in ('device', 'checkpoint', 'notes'))
        if not isinstance(device, str) or device not in devices:
            raise ValueError('未知 checkpoint 设备')
        if (not isinstance(checkpoint, str) or not checkpoint.startswith('/')
                or len(checkpoint) > 4096 or any(ord(c) < 32 for c in checkpoint)
                or '..' in PurePosixPath(checkpoint).parts or checkpoint == '/'):
            raise ValueError('请填写确切 checkpoint 的绝对路径')
        checkpoint = '/' + '/'.join(part for part in checkpoint.split('/') if part and part != '.')
        if checkpoint == '/':
            raise ValueError('请填写确切 checkpoint 的绝对路径')
        if not isinstance(notes, str) or len(notes) > 2000:
            raise ValueError('checkpoint 备注最多 2000 个字符')
        key = json.dumps([device, checkpoint], ensure_ascii=False, separators=(',', ':'))
        with self.lock:
            data = self.read()
            entry = dict(device=device, checkpoint=checkpoint, notes=notes.strip(),
                         updated_at=datetime.now(timezone.utc).isoformat())
            if entry['notes']:
                data[key] = entry
            else:
                data.pop(key, None)
            self.path.parent.mkdir(parents=True, exist_ok=True)
            temp = self.path.with_suffix('.json.tmp')
            temp.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding='utf-8')
            os.replace(temp, self.path)
        return entry
