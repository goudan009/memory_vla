"""Detached benchmark runner: progress and reports survive dashboard restarts."""
import argparse
import json
from pathlib import Path
import signal
import subprocess
import time
import xml.etree.ElementTree as ET

from benchmark_dashboard import atomic_json, read_json


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--run', type=Path, required=True)
    args = parser.parse_args()
    plan = read_json(args.run/'plan.json')
    start = time.monotonic()
    state = dict(status='running', completed=0, total=len(plan['stages']), stage=None)
    child = None
    stopping = False

    def save(**changes):
        state.update(changes, elapsed_seconds=time.monotonic()-start)
        atomic_json(args.run/'state.json', state)

    def stop(signum, frame):
        nonlocal stopping
        stopping = True
        if child is not None and child.poll() is None:
            child.terminate()

    signal.signal(signal.SIGTERM, stop)
    signal.signal(signal.SIGINT, stop)
    save()
    try:
        with (args.run/'run.log').open('a', buffering=1) as log:
            for index, stage in enumerate(plan['stages']):
                if stopping or (args.run/'stop_requested').exists():
                    save(status='stopped')
                    return
                save(stage=stage['name'], step=None, steps=stage.get('steps'))
                log.write('\n'+stage['name']+'\n')
                argv = [value.replace('{run}', str(args.run)) for value in stage['argv']]
                child = subprocess.Popen(argv, cwd=args.root, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                                         text=True, bufsize=1)
                for line in child.stdout:
                    log.write(line)
                    if line.startswith('{'):
                        try:
                            row = json.loads(line)
                        except ValueError:
                            continue
                        if type(row.get('step')) is int:
                            save(step=row['step'], loss=row.get('loss'), cognitive_q_gradient=row.get('cognitive_q_gradient'))
                code = child.wait()
                if stopping:
                    save(status='stopped', returncode=code)
                    return
                if stage.get('unit') and (args.run/'unit.xml').is_file():
                    suites = list(ET.parse(args.run/'unit.xml').getroot().iter('testsuite'))
                    counts = {k: sum(int(s.attrib.get(k, 0)) for s in suites) for k in ('tests', 'failures', 'errors', 'skipped')}
                    atomic_json(args.run/stage['report'], dict(type='unit', passed=code == 0, **counts))
                report = read_json(args.run/stage['report'], {})
                if code or report.get('passed') is not True:
                    save(status='failed', returncode=code, error='该阶段未通过，查看指标与日志')
                    return
                save(completed=index+1, step=None, steps=None)
            save(status='complete', stage='全部检查完成')
    except Exception as error:
        if child is not None and child.poll() is None:
            child.terminate()
            child.wait()
        save(status='failed', error=str(error))
        raise


if __name__ == '__main__':
    main()
