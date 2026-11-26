"""Use the existing manager's preview/start and worker; no server restart."""
import argparse
import hashlib
import json
from pathlib import Path
import shlex
import sys

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/'evaluation/scripts'))
from benchmark_dashboard import BenchmarkManager


class PreflightManager(BenchmarkManager):
    def prepare(self,request):
        base=super().prepare(dict(kind='v2_state_decode',gpu=request['gpu'],notes='V2.1 read-only raw state extraction; zero training'))
        script=ROOT/'v2_1_preflight/scripts/sim_extract.py'
        stage=dict(name='V2.1 official state extraction / 200 demos',argv=[
            'env','OMNIGIBSON_GPU_ID=0','OMNIGIBSON_APPDATA_PATH={run}/isaac_appdata',
            str(ROOT/'scripts/b1k_python.sh'),str(script),'--output','{run}/preflight_extract.json'],
            report='preflight_extract.json',steps=200)
        base.update(title='V2.1 preflight raw state extraction',stages=[stage],commands=[shlex.join(stage['argv'])])
        base['config'].update(category='v2_1_preflight_readonly',training_steps=0,
            extraction_script=str(script),extraction_sha256=hashlib.sha256(script.read_bytes()).hexdigest())
        return base


def main():
    p=argparse.ArgumentParser();p.add_argument('action',choices=['preview','start']);p.add_argument('--gpu',type=int,default=0)
    a=p.parse_args();m=PreflightManager(ROOT);request=dict(gpu=a.gpu)
    result=m.prepare(request) if a.action=='preview' else m.start(request)
    path=ROOT/'v2_1_preflight/samples'/('workbench_'+a.action+'.json');path.parent.mkdir(parents=True,exist_ok=True)
    path.write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result,ensure_ascii=False,indent=2))


if __name__=='__main__':main()
