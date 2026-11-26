"""One observational workbench patrol tick, scheduled by a systemd user timer.

Preserves user pause controls. It never starts a new experiment or changes a
training configuration; the existing remote controller manages the GPU child.
"""
from datetime import datetime, timedelta
import fcntl
import json
import math
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'train'))
from dashboard import TrainingManager

RUN = 'v2.1__test01__exp14__run001'
EXPERIMENT = ROOT / 'Output/test_01/experiment_14'
DIRECTORY = EXPERIMENT / 'evidence/patrol'


def write(path, value):
    pending = path.with_suffix('.pending')
    pending.write_text(json.dumps(value, ensure_ascii=False, indent=2) + '\n')
    pending.replace(path)


def main():
    DIRECTORY.mkdir(parents=True, exist_ok=True)
    with (DIRECTORY / 'patrol.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        state_path = DIRECTORY / 'watchdog_state.json'
        state = json.loads(state_path.read_text()) if state_path.exists() else {}
        now = datetime.now().astimezone()
        record = dict(checked_at=now.isoformat(), run=RUN)
        manager = TrainingManager(ROOT)
        manager.close()
        try:
            detail = manager.detail(dict(run=RUN, arm='baseline'))
            current = detail['state']
            record.update(state=current, checkpoints=detail['checkpoints'])
            remote_root = detail['config']['source']  # immutable snapshot
            output = detail['config']['argv']
            output = output[output.index('--output') + 1]
            code = (
                'from pathlib import Path\nimport json,shutil,subprocess\n'
                'p=Path(' + repr(output) + ')\n'
                'print(json.dumps(dict(free_bytes=shutil.disk_usage(p).free,\n'
                ' stop=json.loads((p/"stop.json").read_text()) if (p/"stop.json").exists() else None,\n'
                ' gpu=subprocess.check_output(["nvidia-smi","--query-gpu=index,uuid,memory.used,utilization.gpu",'
                '"--format=csv,noheader"],text=True))))\n'
            )
            record['resources'] = json.loads(manager._ssh('python3 -', code, timeout=30))
            record['immutable_source'] = remote_root
            last = current.get('last') or {}
            finite = all(math.isfinite(last[key]) for key in ('fm', 'grad_norm') if key in last)
            record['finite_metrics'] = finite
            # A concrete numeric failure is paused only through the workbench.
            if not finite and current['status'] == 'training' and not record['resources']['stop']:
                record['action'] = manager.stop(dict(run=RUN, arm='baseline'))
            step = current.get('step', 0)
            if step != state.get('last_step'):
                state['last_progress_at'] = now.isoformat()
            record['disk_warning'] = record['resources']['free_bytes'] < 50 * 2**30
            record['status'] = current['status']
            if current['status'] == 'completed' and step == 6000:
                record['next_stage'] = 'Stage1 complete; memory-stage real preflight required before its formal run'
            state.update(last_status=current['status'], last_step=step, error=None)
            if current['status'] in ('completed', 'failed', 'cancelled'):
                state['terminal'] = record
                subprocess.run(['systemctl', '--user', 'stop', 'memory-vla-v21-patrol.timer'],
                               check=True, timeout=15)
        except Exception as error:
            record['error'] = repr(error)
            state['error'] = repr(error)
        state.update(last_checked_at=now.isoformat(),
                     next_check_at=None if 'terminal' in state else (now + timedelta(seconds=120)).isoformat(),
                     next_check_reason='actual workbench progress, finite metrics, complete checkpoints, GPU and disk')
        write(state_path, state)
        with (DIRECTORY / 'watchdog_log.jsonl').open('a') as stream:
            stream.write(json.dumps(record, ensure_ascii=False) + '\n')
        print(json.dumps(state, ensure_ascii=False))


if __name__ == '__main__':
    main()
