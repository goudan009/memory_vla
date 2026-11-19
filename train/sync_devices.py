"""Publish the 4090 working source to training/evaluation devices over SSH."""
import argparse
import json
from pathlib import Path
from dashboard import TrainingManager


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--device', choices=['all','a100','a6000'], default='all')
    parser.add_argument('--auto', action='store_true', help='also enable dashboard background sync/retry')
    args=parser.parse_args()
    root=Path(__file__).resolve().parents[1]
    devices=json.loads((root/'configs/workbench_devices.json').read_text())
    failed=False
    for key in (devices if args.device=='all' else [args.device]):
        device=devices[key]
        manager=TrainingManager(root,device['host'],key,device)
        try:
            request={'once':True}
            if args.auto: request['enabled']=True
            result=manager.sync(request)
            print(json.dumps(dict(device=key,**result),ensure_ascii=False),flush=True)
        except Exception as error:
            failed=True
            print(json.dumps(dict(device=key,error=str(error)),ensure_ascii=False),flush=True)
        finally:
            manager.close()
    return int(failed)


if __name__=='__main__':
    raise SystemExit(main())
