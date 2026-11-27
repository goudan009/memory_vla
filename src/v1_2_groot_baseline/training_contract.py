"""Candidate N1.7 native training spec; no launch and no implicit scientific defaults."""
from pathlib import Path
from .contracts import absolute, inspect_upstream, inspect_checkpoint, B1K2025, VERSION
from .data_contract import audit_2025


def batch_accounting(world_size,microbatch,accumulation):
    if any(type(n) is not int or n<1 for n in (world_size,microbatch,accumulation)):
        raise ValueError("explicit positive batch factors required")
    # Upstream N1.7 CLI global_batch_size is PRE-accumulation.
    return dict(global_batch_size=world_size*microbatch,
                effective_batch=world_size*microbatch*accumulation)


def native_train_candidate(*, python,upstream,base,dataset,modality,job_root,run_id,
                           gpus,microbatch,accumulation,steps,save_steps,lr,wd,warmup_ratio):
    """Build a native command for review after scientific choices are locked.

    The existing workbench still has to implement the native controller,
    canonical atomic checkpoint publication and strict resume. This candidate
    must NOT be submitted as a direct or fallback launch.
    """
    import re
    paths=map(absolute,(python,upstream,base,dataset,modality,job_root))
    python,upstream,base,dataset,modality,job_root=paths
    if not re.fullmatch(r"v1_2__test[0-9]{2}__exp[0-9]{2}__run(?!0+$)[0-9]{3,}",run_id) or job_root.name!=run_id:
        raise ValueError("explicit workbench-allocated V_1_2 identity required")
    if not python.is_file() or not modality.is_file():
        raise ValueError("model environment and job-specific modality missing")
    inspect_upstream(upstream);inspect_checkpoint(base,role="generic");audit_2025(dataset)
    if not isinstance(gpus,list) or not gpus or len(set(gpus))!=len(gpus) or any(type(g) is not int or g<0 for g in gpus):
        raise ValueError("explicit allocated GPU list required")
    batch=batch_accounting(len(gpus),microbatch,accumulation)
    if any(type(n) is not int or n<1 for n in (steps,save_steps)) or not (lr>0 and wd>=0 and 0<=warmup_ratio<1):
        raise ValueError("explicit approved training budget and optimizer parameters required")
    # Upstream output is staging only. Do not publish checkpoint-N as final ID.
    native=job_root/"run/native"
    argv=[str(python),"-m","torch.distributed.run","--standalone",
          "--nproc-per-node="+str(len(gpus)),str(upstream/"scripts/b1k/train_b1k.py"),
          "--base-model-path",str(base),"--dataset-path",str(dataset),
          "--embodiment-tag","NEW_EMBODIMENT","--modality-config-path",str(modality),
          "--num-gpus",str(len(gpus)),"--global-batch-size",str(batch["global_batch_size"]),
          "--gradient-accumulation-steps",str(accumulation),
          "--learning-rate",str(lr),"--weight-decay",str(wd),"--warmup-ratio",str(warmup_ratio),
          "--max-steps",str(steps),"--save-steps",str(save_steps),
          "--save-total-limit",str(steps+1),"--output-dir",str(native),"--experiment-name",run_id,
          "--decode-only-used-frames"]
    return dict(version=VERSION,protocol=B1K2025,run_id=run_id,argv=argv,
                cwd=str(upstream),gpu_indices=gpus,**batch,
                strict_resume_required=steps>2000,launch_ready=False,
                blockers=["verified train-only split/statistics","native workbench controller",
                          "canonical checkpoint and strict resume preflight","approved training task"])
