#!/usr/bin/env python3
"""CPU worker behind the existing workbench dataset action."""
import argparse
import json
from pathlib import Path
import sys
import traceback

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))
from v1_2_groot_baseline.b1k2026_training_data import materialize


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--request', required=True)
    parser.add_argument('--status', required=True, type=Path)
    args = parser.parse_args()
    request = json.loads(args.request)
    try:
        manifest = materialize(request['source'], request['destination'], request['split'],
                               request['upstream'], request['language_mode'])
        state = dict(state='complete', data_manifest=str(Path(request['destination']) / 'data_manifest.json'),
                     episodes={k: len(v['episodes']) for k,v in manifest['views'].items()})
        code = 0
    except Exception as error:
        traceback.print_exc()
        state, code = dict(state='failed', error=str(error)), 1
    temporary = args.status.with_suffix('.pending')
    temporary.write_text(json.dumps(state))
    temporary.replace(args.status)
    return code


if __name__ == '__main__':
    raise SystemExit(main())
