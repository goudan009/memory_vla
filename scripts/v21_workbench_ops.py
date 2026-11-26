"""Use the existing TrainingManager from a review worktree without restarting UI."""
import argparse
import json
from pathlib import Path
import sys

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'train'))
from dashboard import TrainingManager


def main():
    parser=argparse.ArgumentParser(__doc__)
    parser.add_argument('action',choices=['prepare','start','snapshot','detail','stop','resume','delete','sync'])
    parser.add_argument('--request',type=Path)
    parser.add_argument('--evidence',type=Path)
    args=parser.parse_args()
    request=json.loads(args.request.read_text()) if args.request else {}
    manager=TrainingManager(ROOT)
    manager.close()  # one-shot same manager/RPC, no second background sync loop
    result=getattr(manager,args.action)(request) if args.action!='snapshot' else manager.snapshot()
    if args.evidence:
        args.evidence.parent.mkdir(parents=True,exist_ok=True)
        args.evidence.write_text(json.dumps(result,indent=2,ensure_ascii=False)+'\n')
    print(json.dumps(result,ensure_ascii=False))


if __name__=='__main__':main()
