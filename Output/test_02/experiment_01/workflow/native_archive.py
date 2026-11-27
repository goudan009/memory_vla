"""Archive workbench-owned native run evidence without copying model tensors."""
import json
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[4]
EXPERIMENT = ROOT / 'Output/test_02/experiment_01'
RUN = 'v2_1__test02__exp01__run003'
TRAINING = '/nvme3/data_wyh/outputs/memory_vla/v1_groot_memoryvla/training'
REMOTE_EVIDENCE = '/nvme3/data_wyh/outputs/memory_vla/v2_1/test02/exp01/evidence'
EVIDENCE = EXPERIMENT / 'evidence/native_random_v2'


def write(path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + '.pending')
    temporary.write_text(json.dumps(data, ensure_ascii=False, indent=2) + '\n')
    temporary.replace(path)


def ssh(code, timeout=120):
    result = subprocess.run(['ssh', '-o', 'BatchMode=yes', '-o', 'ConnectTimeout=5',
                             '172.28.62.17', 'python3 -'], input=code, text=True,
                            capture_output=True, timeout=timeout, check=True)
    return json.loads(result.stdout)


def archive(run):
    if run not in (RUN, RUN + '__preflight001'):
        raise ValueError('Only the current authorized run is in scope')
    code = '''
import json, hashlib, subprocess
from pathlib import Path
root = Path(TRAINING) / RUN
destination = Path(REMOTE_EVIDENCE) / (RUN + '-archive.json')
files = {}
for relative in ['config/web_launch.json', 'config/source_manifest.json',
                 'config/web_process.json', 'config/resume-event-001.json',
                 'run/status.json', 'baseline/status.json', 'run/preflight.json',
                 'run/data_split.json', 'run/native_contract.json',
                 'run/native_runtime.json', 'run/prepared/normalization_audit.json',
                 'run/prepared/PREPARED.json', 'run/checkpoints/latest.json']:
    path = root / relative
    if path.is_file(): files[relative] = json.loads(path.read_text())
for path in sorted((root/'run/checkpoints').glob('*/COMPLETE.json')):
    files[str(path.relative_to(root))] = json.loads(path.read_text())
files['run/metrics.jsonl'] = (root/'run/metrics.jsonl').read_text()
files['logs/controller.log'] = (root/'logs/controller.log').read_text()
live = []
for path in Path('/proc').glob('[0-9]*/cmdline'):
    try:
        argv = path.read_bytes().decode().strip('\\0').split('\\0')
        if any(flag in argv and argv[argv.index(flag)+1] == target for flag, target in
               [('--job-root',str(root)),('--controller',str(root/'config/v21_controller.json'))]):
            live.append(int(path.parent.name))
    except (OSError, UnicodeError, IndexError): pass
evidence = dict(run=RUN, files=files, live_pids=live,
    gpu=subprocess.check_output(['nvidia-smi','--query-gpu=index,uuid,memory.used,utilization.gpu',
                                 '--format=csv,noheader'],text=True))
if files['run/status.json']['status'] != 'completed' or live:
    raise RuntimeError('Archive requires completed run and released owned processes')
if '__preflight' in RUN:
    gate=files['run/preflight.json']
    if not (gate['passed'] and gate['strict_resume'] and gate['step']==4
            and all(gate['restore_checks'].get(k) is True for k in ('model','optimizer','scheduler','rng'))):
        raise RuntimeError('Strict native preflight gate failed')
    gate_path=destination.with_name(RUN+'-preflight.json')
    gate_path.parent.mkdir(parents=True,exist_ok=True)
    gate_path.write_text(json.dumps(gate,indent=2))
    evidence['gate_path']=str(gate_path)
destination.parent.mkdir(parents=True,exist_ok=True)
destination.write_text(json.dumps(evidence,indent=2))
evidence['archive_path']=str(destination)
print(json.dumps(evidence))
'''
    code = '\n'.join(f'{key}={value!r}' for key,value in
                     [('TRAINING',TRAINING),('REMOTE_EVIDENCE',REMOTE_EVIDENCE),('RUN',run)]) + '\n' + code
    result = ssh(code)
    write(EVIDENCE / (run + '-archive.json'), result)
    return result


if __name__ == '__main__':
    result = archive(RUN + '__preflight001' if sys.argv[1:] == ['--preflight'] else RUN)
    print(json.dumps({k:v for k,v in result.items() if k != 'files'},indent=2))
