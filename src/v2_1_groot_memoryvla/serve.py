"""V2_1 dedicated policy service; original V1 service remains unchanged."""
import argparse
import json
from pathlib import Path
from .config import check_scale
def main():
    p=argparse.ArgumentParser(__doc__)
    p.add_argument("--model-path",required=True);p.add_argument("--device",default="cuda:0")
    p.add_argument("--host",default="0.0.0.0");p.add_argument("--port",type=int,default=5557)
    p.add_argument("--embodiment-tag",choices=["BEHAVIOR_R1_PRO","behavior_r1_pro"],default="BEHAVIOR_R1_PRO")
    p.add_argument("--use-sim-policy-wrapper",action="store_true")
    p.add_argument("--residual-scale",type=float,default=1.)
    p.add_argument("--history-mode",choices=["normal","empty","wrong","other_episode"],default="normal")
    p.add_argument("--seed",type=int,default=42)
    args=p.parse_args();check_scale(args.residual_scale)
    if (Path(args.model_path)/'v2_1.json').is_file():
        from .policy import Policy
    else:
        run_file=Path(args.model_path)/'v2_1_run.json'
        run=json.loads(run_file.read_text()) if run_file.is_file() else {}
        if not (Path(args.model_path)/'v2_1_stage1.json').is_file() and run.get('stage')!='continued':
            p.error('Expected V2_1 memory sidecar or Stage1 lineage metadata')
        from .baseline_policy import BaselinePolicy as Policy
    from v1_groot_memoryvla.data.embodiment_tags import EmbodimentTag
    from v1_groot_memoryvla.policy.gr00t_policy import Gr00tSimPolicyWrapper
    from v1_groot_memoryvla.policy.server_client import PolicyServer
    from v1_groot_memoryvla.experiment.first_training import seeded
    policy=Policy(EmbodimentTag("behavior_r1_pro"),args.model_path,device=args.device,
                  history_mode=args.history_mode,residual_scale=args.residual_scale)
    if args.use_sim_policy_wrapper:policy=Gr00tSimPolicyWrapper(policy)
    seeded(args.seed)
    print(vars(args),flush=True)
    PolicyServer(policy=policy,host=args.host,port=args.port).run()
if __name__=="__main__":main()
