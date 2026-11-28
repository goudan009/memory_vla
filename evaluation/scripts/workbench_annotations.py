"""Persistent, benchmark-scoped annotations for native policy rollouts."""
import json
import threading
import tempfile
from pathlib import Path
from datetime import datetime, timezone

FIELDS = {'notes': 2000, 'stage': 120, 'cause': 120, 'reason': 500}

class AnnotationStore:
    def __init__(self, path):
        self.path = Path(path)
        self.lock = threading.RLock()

    def _read(self):
        if not self.path.is_file():
            return {}
        data = json.loads(self.path.read_text(encoding='utf-8'))
        if not isinstance(data, dict):
            raise ValueError('标注文件格式不正确')
        return data

    def read(self, benchmark, run, keys):
        with self.lock:
            rows = self._read().get(benchmark, {}).get(run, {})
            return {k: rows[k] for k in keys if k in rows}

    def save(self, benchmark, run, keys, episode, value):
        if episode not in keys:
            raise ValueError('回合不存在或尚未结束')
        if not isinstance(value, dict) or set(value) != set(FIELDS):
            raise ValueError('标注字段不正确')
        for field, limit in FIELDS.items():
            if not isinstance(value[field], str) or len(value[field]) > limit:
                raise ValueError('标注字段长度不正确')
        with self.lock:
            data = self._read()
            rows = data.setdefault(benchmark, {}).setdefault(run, {})
            record = {k: v.strip() for k, v in value.items()}
            record['updated_at'] = datetime.now(timezone.utc).isoformat()
            if any(record[k] for k in FIELDS):
                rows[episode] = record
            else:
                rows.pop(episode, None)
            self.path.parent.mkdir(parents=True, exist_ok=True)
            with tempfile.NamedTemporaryFile(mode='w', encoding='utf-8', dir=self.path.parent,
                                             prefix='.annotations-', delete=False) as handle:
                temporary = Path(handle.name)
                try:
                    json.dump(data, handle, ensure_ascii=False, indent=2)
                    handle.close()
                    temporary.replace(self.path)
                finally:
                    temporary.unlink(missing_ok=True)
            return record

def libero_episode_key(row):
    return f"task:{row['task']}:trial:{row['trial']}:initial:{row['initial_state_index']}"

def episode_keys(benchmark, run, launch_manager, benchmark_manager):
    if not isinstance(run, str) or not run or len(run) > 128:
        raise ValueError('运行编号不正确')
    if benchmark == 'b1k2026':
        job = next((j for j in launch_manager.snapshot()
                    if j.get('id') == run and j.get('kind') == 'b1k2026_websocket'), None)
        if job is None:
            raise ValueError('找不到 2026 评估')
        return set(job.get('native_episode_keys', []))
    if benchmark == 'libero':
        detail = benchmark_manager.detail({'run': run})
        if detail['job']['kind'] != 'libero_eval':
            raise ValueError('仅支持 LIBERO 策略评估回合')
        return {libero_episode_key(row) for report in detail['reports']
                if isinstance(report.get('data'), dict) and report['data'].get('type') == 'libero_eval'
                for row in report['data'].get('episodes', [])}
    raise ValueError('不支持此 benchmark')
