"""Native GR00T Stage1 worker, exclusively launched by the existing workbench."""
import json
import os
from pathlib import Path
import argparse
import copy
import math
import time
import bisect
import hashlib
import random
import sys

from stage1_native_prepare import read, write, sha, verify_prepared, TAG, RELATIVE


STEPS = 6000
GPU = 4
MICROBATCH = 2
ACCUM = 8


def workbench_layout(job_root):
    """Derive the only allowed published checkpoint namespace from a real job.

    Never reserve a run here; the workbench owns reservation and source snapshots.
    Native checkpoint-N folders are internal staging, not public identities.
    """
    repo = Path(__file__).resolve().parents[1]
    sys.path.insert(0, str(repo/"train"))
    from run_identity import validate_run_identity
    root = Path(job_root)
    if root.is_symlink():
        raise ValueError("workbench job root cannot be a symlink")
    config = read(root/"config/web_launch.json")
    run = config.get("run_id")
    if not isinstance(run, str) or run != root.name or config.get("run_name") != run:
        raise ValueError("workbench run identity mismatch")
    if config.get("version") != "V_2_1" or config.get("stage") != "stage1":
        raise ValueError("expected V_2_1 Stage1 workbench job")
    if config.get("recipe") != "native_random_v2":
        raise ValueError("explicit native_random_v2 recipe required")
    preflight = config.get("preflight_only")
    if type(preflight) is not bool:
        raise ValueError("workbench preflight identity required")
    fields = validate_run_identity(run, config["version"], preflight=preflight)
    if fields["test_id"] != "test02" or fields["experiment_id"] != "exp01":
        raise ValueError("recipe belongs to current Test02/Exp01 task; do not infer IDs")
    return root/"run", run


def canonical_checkpoint(job_root, step):
    output, run = workbench_layout(job_root)
    if type(step) is not int or step < 0:
        raise ValueError("step must be cumulative optimizer step")
    return output/"checkpoints"/f"{run}__step{step:06d}"


def plan(args, steps):
    return dict(steps=steps, global_batch=GPU*MICROBATCH*ACCUM, num_gpus=GPU,
                per_device_batch=MICROBATCH, accumulation=ACCUM, learning_rate=1e-4,
                lr_scheduler="cosine", warmup_ratio=.05, weight_decay=1e-5,
                seed=42, optimizer="adamw_torch_fused", max_grad_norm=1.,
                bf16=True, tune_llm=False, tune_visual=False, tune_top_llm_layers=0,
                tune_projector=True, tune_diffusion_model=True, tune_vlln=True,
                horizon=50, valid_action_steps=32, inference_steps=4,
                execute_steps=8, episode_sampling_rate=.1, shard_size=1024,
                data_shuffle=True, allow_padding=False, native_eval_strategy="no",
                output=str(args.output.resolve()), prepared=str(args.prepared.resolve()),
                official_root=str(args.official_root.resolve()))


def official_files(root):
    names = ("gr00t/configs/finetune_config.py",
             "gr00t/configs/training/training_config.py",
             "gr00t/configs/data/embodiment_configs.py",
             "gr00t/experiment/experiment.py", "gr00t/experiment/trainer.py",
             "gr00t/model/gr00t_n1d6/setup.py",
             "gr00t/model/gr00t_n1d6/gr00t_n1d6.py",
             "gr00t/model/gr00t_n1d6/processing_gr00t_n1d6.py",
             "gr00t/data/dataset/factory.py",
             "gr00t/data/dataset/sharded_single_step_dataset.py",
             "gr00t/data/dataset/sharded_mixture_dataset.py",
             "gr00t/data/state_action/state_action_processor.py",
             "gr00t/data/stats.py")
    return {n:sha(root/n) for n in names}


def identity(args):
    return dict(prepared_sha256=sha(args.prepared/"PREPARED.json"),
                launcher_sha256=sha(__file__),
                preparer_sha256=sha(Path(__file__).with_name("stage1_native_prepare.py")),
                official_root=str(args.official_root.resolve()),
                official_sources=official_files(args.official_root),
                gpu_count=GPU, microbatch=MICROBATCH, accumulation=ACCUM)


def assert_processor(checkpoint, prepared):
    import numpy as np
    pc = read(checkpoint/"processor_config.json")["processor_kwargs"]
    modes = pc["modality_configs"][TAG]
    mc = read(prepared/"train/meta/modality.json")
    if (pc["use_percentiles"] or not pc["use_relative_action"] or not pc["clip_outliers"]
        or pc["max_action_dim"] != 128 or pc["max_state_dim"] != 128
        or pc["max_action_horizon"] != 50 or modes["action"]["delta_indices"] != list(range(32))
        or modes["state"]["delta_indices"] != [0] or modes["video"]["delta_indices"] != [0]
        or modes["state"]["modality_keys"] != list(mc["state"])
        or modes["action"]["modality_keys"] != list(mc["action"])):
        raise ValueError("native processor does not match the audited R1Pro contract")
    actual = read(checkpoint/"statistics.json")[TAG]
    expected = read(prepared/"train/meta/relative_stats.json")
    absolute = read(prepared/"train/meta/stats.json")
    for key, fields in expected.items():
        for f in ("min","max","q01","q99"):
            np.testing.assert_allclose(actual["relative_action"][key][f],fields[f],
                                       atol=1e-7,rtol=1e-6)
    for mod, feature in (("state","observation.state"),("action","action")):
        for k, spec in mc[mod].items():
            for f in ("min","max"):
                np.testing.assert_allclose(actual[mod][k][f],
                    absolute[feature][f][spec["start"]:spec["end"]],atol=1e-7,rtol=1e-6)



def rank_schedule(mixture, rank, world, total):
    """Index the upstream mixture's shuffled shard schedule; no data reseeding."""
    selected=[];ends=[];n=0
    for occurrence,(ds,shard) in enumerate(mixture.shard_sampling_schedule):
        if occurrence % world != rank:continue
        size=int(mixture.datasets[ds].get_shard_length(shard))
        if size<=0:raise ValueError("empty native shard")
        selected.append((occurrence,int(ds),int(shard),size))
        n+=size;ends.append(n)
        if n>=total:break
    if n<total:raise ValueError("native schedule shorter than budget")
    return selected,ends


def locate(schedule,ends,index,seed):
    import numpy as np
    i=bisect.bisect_right(ends,index)
    occurrence,ds,shard,size=schedule[i]
    offset=index-(ends[i-1] if i else 0)
    order=np.random.default_rng(np.random.SeedSequence([seed,occurrence])).permutation(size)
    return occurrence,ds,shard,int(order[offset])


def cursor(step):
    if type(step) is not int or not 0<=step<=STEPS:raise ValueError("invalid optimizer step")
    return step*MICROBATCH*ACCUM


def atomic(path,value):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    pending=path.with_name(path.name+".pending")
    write(pending,value);pending.replace(path)


def source_identity(root):
    # Check the actual upstream source, not a guessed checkout name.
    files=official_files(root)
    for p in sorted((root/"gr00t").rglob("*.py")):
        files[str(p.relative_to(root))]=sha(p)
    return files


def verify_checkpoint_files(checkpoint):
    marker=read(checkpoint/"COMPLETE.json")
    for name,digest in marker["files"].items():
        p=checkpoint/name
        if not p.is_file() or p.is_symlink() or sha(p)!=digest:
            raise ValueError("incomplete or modified native checkpoint: "+name)
    return marker


def publish(job_root,native,step,contract,manifest):
    import torch
    output,run=workbench_layout(job_root)
    final=canonical_checkpoint(job_root,step)
    if final.exists():raise FileExistsError(final)
    source=native/f"checkpoint-{step}"
    state=read(source/"trainer_state.json")
    if state["global_step"]!=step:raise ValueError("native optimizer step mismatch")
    assert_processor(source,output/"prepared")
    required=["optimizer.pt","scheduler.pt","trainer_state.json","config.json",
              "processor_config.json","statistics.json","embodiment_id.json",
              *[f"rng_state_{r}.pth" for r in range(GPU)]]
    model_files=sorted(p.name for p in source.glob("*.safetensors"))
    if not model_files:raise ValueError("missing native model weights")
    if (source/"model.safetensors.index.json").exists():
        required.append("model.safetensors.index.json")
        index=read(source/"model.safetensors.index.json")
        if not set(index["weight_map"].values())<=set(model_files):
            raise ValueError("missing model shards")
    required+=model_files
    if any(not (source/f).is_file() for f in required):
        raise ValueError("missing native model/optimizer/scheduler/RNG files")
    cfg=read(source/"config.json")
    if cfg.get("tune_llm") or cfg.get("tune_visual") or cfg.get("tune_top_llm_layers")!=0:
        raise ValueError("checkpoint freeze contract")
    # A small index references the real full HF optimizer/scheduler/RNG files.
    # The existing workbench sees this as a complete checkpoint, but resume
    # always routes to the native trainer, never the sequential trainer.
    torch.save(dict(format="native_hf_v2",step=step,world_size=GPU,
                    per_rank_sample_cursor=cursor(step),contract=contract,
                    optimizer_file="optimizer.pt",scheduler_file="scheduler.pt",
                    rng_files=[f"rng_state_{r}.pth" for r in range(GPU)]),
               source/"training_state.pt")
    write(source/"v2_1_stage1.json",dict(step=step,split=read(output/"prepared/data_split.json"),
              raw_lo=manifest["raw_lo"],raw_hi=manifest["raw_hi"],
              statistics_sha256=sha(source/"statistics.json"),training_recipe="native_random_v2"))
    write(source/"v2_1_run.json",dict(stage="stage1",step=step,run_id=run,recipe="native_random_v2",
              source_digest=contract["source_digest"],seed=42))
    files={f:sha(source/f) for f in required+["training_state.pt","v2_1_stage1.json","v2_1_run.json"]}
    write(source/"COMPLETE.json",dict(step=step,world_size=GPU,run_id=run,
            recipe="native_random_v2",source_digest=contract["source_digest"],
            per_rank_sample_cursor=cursor(step),files=files))
    final.parent.mkdir(parents=True,exist_ok=True)
    source.rename(final) # same filesystem; publish only once fully saved
    atomic(final.parent/"latest.json",dict(step=step,path=str(final),world_size=GPU))
    return final


def worker(args):
    import numpy as np
    import torch
    import torch.distributed as dist
    from torch.utils.data import Dataset,DataLoader
    from transformers import TrainerCallback
    from types import SimpleNamespace

    if args.job_root.is_symlink():raise ValueError('workbench job root must not be symlink')
    job_root=args.job_root.resolve()
    output,run=workbench_layout(job_root)
    job=read(job_root/"config/web_launch.json")
    root=Path(job["official_root"]).resolve()
    source=Path(job["source"]).resolve()
    if Path(__file__).resolve().parent.parent!=source:
        raise RuntimeError("worker must run in the immutable workbench source snapshot")
    if int(os.environ.get("WORLD_SIZE","0"))!=GPU:
        raise ValueError("workbench must launch four A100 ranks")
    local=int(os.environ["LOCAL_RANK"]);torch.cuda.set_device(local)
    dist.init_process_group(backend="nccl")
    rank=dist.get_rank()
    if "A100" not in torch.cuda.get_device_name(local):
        raise RuntimeError("expected an actual A100")
    # Snapshot upstream hash and use exact external source; do not modify it.
    sys.path.insert(0,str(root))
    from gr00t.configs.base_config import Config
    from gr00t.configs.model.gr00t_n1d6 import Gr00tN1d6Config
    from gr00t.configs.data.data_config import SingleDatasetConfig
    from gr00t.configs.data.embodiment_configs import MODALITY_CONFIGS
    import gr00t.experiment.experiment as exp
    import gr00t.data.dataset.factory as factory
    from stage1_native_prepare import prepare

    if not Path(exp.__file__).resolve().is_relative_to(root):
        raise RuntimeError("wrong upstream gr00t import")
    prepared=output/"prepared"
    if rank==0:
        output.mkdir(parents=True,exist_ok=True)
        if not prepared.exists():
            prepare(Path(job["dataset"]),Path(job["split"]),Path(job["checkpoint"]),prepared,"guarded")
        manifest=verify_prepared(prepared)
        # Confirm the supplied split matches the original Stage1 seed42
        # permutation rather than simply being another 180/20 split.
        split=read(prepared/"data_split.json")
        ids=sorted(split["train"]+split["validation"])
        shuffled=np.random.default_rng(42).permutation(ids).tolist()
        if sorted(shuffled[:20])!=sorted(split["validation"]) or sorted(shuffled[20:])!=sorted(split["train"]):
            raise ValueError("split differs from original Stage1 seed42")
        upstream=source_identity(root)
        base=Path(job["checkpoint"])
        contract=dict(recipe="native_random_v2",source_digest=job["source_manifest"]["digest"],
                      upstream=upstream,base=str(base.resolve()),
                      base_weights={p.name:sha(p) for p in sorted(base.glob("*.safetensors"))},
                      dataset=str(Path(job["dataset"]).resolve()),
                      split=read(prepared/"data_split.json"),statistics=manifest["hashes"],
                      world_size=GPU,microbatch=MICROBATCH,accumulation=ACCUM,
                      steps=STEPS,video_backend=job["video_backend"],workers=job["workers"],
                      seed=42,lr=1e-4,scheduler="cosine",warmup_ratio=.05,weight_decay=1e-5)
        if not contract["base_weights"]:raise ValueError("generic weight shards missing")
        # Hash the numeric training and validation source, plus video identity
        # (path, size, mtime). No held-out stats used to fit train.
        data={}
        for partition in ("train","validation"):
            for p in sorted((prepared/partition/"data").glob("*/*.parquet")):
                data[str(p.relative_to(prepared))]=sha(p)
            for p in sorted((prepared/partition/"videos").glob("*/*/*.mp4")):
                st=p.stat()
                data[str(p.relative_to(prepared))]=[str(p.resolve()),st.st_size,st.st_mtime_ns]
        contract["data"]=data
        if (output/"native_contract.json").exists():
            if read(output/"native_contract.json")!=contract:
                raise ValueError("resume source/data/recipe changed")
        else:
            atomic(output/"native_contract.json",contract)
        if not job["preflight_only"]:
            gate=read(Path(job["accept_preflight"]))
            if not gate.get("passed") or not gate.get("strict_resume") or gate.get("contract")!=contract or not all(gate.get("restore_checks",{}).get(k) is True for k in ("model","optimizer","scheduler","rng")):
                raise ValueError("formal run requires matching real strict-resume preflight")
        atomic(output/"data_split.json",contract["split"])
    dist.barrier()
    manifest=verify_prepared(prepared);contract=read(output/"native_contract.json")
    if source_identity(root)!=contract['upstream']:raise ValueError('upstream source changed')
    resume=args.resume.resolve() if args.resume else None
    resumed_step=0
    if resume:
        if rank==0:
            marker=verify_checkpoint_files(resume)
            if marker["run_id"]!=run or resume!=canonical_checkpoint(job_root,marker["step"]):
                raise ValueError("resume checkpoint not owned by this job")
        dist.barrier()
        saved=torch.load(resume/"training_state.pt",map_location="cpu",weights_only=False)
        resumed_step=saved["step"]
        if saved["contract"]!=contract or saved["per_rank_sample_cursor"]!=cursor(resumed_step):
            raise ValueError("resume contract/cursor mismatch")
    preflight=job["preflight_only"]
    # Use the same 6000-step scheduler in preflight; disposable run stops at
    # step 2, releases GPUs and must resume through workbench to step 4.
    if preflight and resumed_step>=4:raise ValueError("preflight already complete")
    cap=(4 if resume else 2) if preflight else STEPS
    args=SimpleNamespace(output=output,prepared=prepared,official_root=root,
                         workers=job["workers"],video_backend=job["video_backend"],
                         preflight=preflight)
    steps=STEPS
    # The upstream factory rewrites metadata; audited train-only files are
    # already computed. Only suppress redundant generation for this view.
    def checked_stats(path,*unused):
        if Path(path).resolve()!=(prepared/"train").resolve():
            raise ValueError("unexpected native stats dataset")
        verify_prepared(prepared)
    factory.generate_stats=checked_stats
    factory.generate_rel_stats=checked_stats

    config = Config()
    config.model = Gr00tN1d6Config(**read(prepared/"generic_base/config.json"))
    config.data.datasets = [SingleDatasetConfig(dataset_paths=[str(prepared/"train")],
                                                embodiment_tag=TAG)]
    config.data.modality_configs = {TAG:copy.deepcopy(MODALITY_CONFIGS[TAG])}
    mods = config.data.modality_configs[TAG]
    mc = read(prepared/"train/meta/modality.json")
    if (list(mods["action"].delta_indices) != list(range(32)) or
        list(mods["state"].delta_indices) != [0] or
        list(mods["video"].delta_indices) != [0] or
        list(mods["state"].modality_keys) != list(mc["state"]) or
        list(mods["action"].modality_keys) != list(mc["action"])):
        raise ValueError("upstream modality mismatch")
    config.data.override_pretraining_statistics=True
    config.data.seed=42
    config.data.shuffle=True
    config.data.allow_padding=False
    config.data.episode_sampling_rate=.1
    config.data.shard_size=1024
    config.data.num_shards_per_epoch=100000
    config.data.video_backend=args.video_backend
    t=config.training
    t.start_from_checkpoint=str(prepared/"generic_base")
    t.output_dir=str(args.output/"native")
    t.experiment_name=None
    t.max_steps=steps
    # In this upstream revision global_batch_size is divided only by world size,
    # not by accumulation. Explicit batch_size override is mandatory.
    t.global_batch_size=64
    t.batch_size=MICROBATCH
    t.gradient_accumulation_steps=ACCUM
    t.num_gpus=GPU
    t.learning_rate=1e-4
    t.weight_decay=1e-5
    t.warmup_ratio=.05
    t.warmup_steps=0
    t.lr_scheduler_type="cosine"
    t.optim="adamw_torch_fused"
    t.max_grad_norm=1.
    t.fp16=False
    t.bf16=True
    t.use_ddp=True
    t.dataloader_num_workers=args.workers
    t.logging_steps=1 if args.preflight else 10
    t.save_steps=1000
    t.save_total_limit=7  # upstream typed config; actual HF rotation disabled by CheckedTrainer
    t.eval_strategy="no"  # Native sharded DatasetFactory asserts this.
    t.use_wandb=False
    t.transformers_local_files_only=True


    native=output/"native"
    NativeTrainer=exp.Gr00tTrainer

    class IndexedShards(Dataset):
        def __init__(self,mixture):
            self.mixture=mixture
            self.total=cursor(STEPS)
            self.schedule,self.ends=rank_schedule(mixture,rank,GPU,self.total)
            self.cache=None;self.cached_occurrence=None
        def __len__(self):return self.total
        def __getitem__(self,index):
            occurrence,ds,shard,offset=locate(self.schedule,self.ends,int(index),42)
            if occurrence!=self.cached_occurrence:
                # Deterministic per-shard-occurrence augmentation independent of
                # process restart, worker identity and prefetch position.
                seed=int(np.random.SeedSequence([42,occurrence,171]).generate_state(1)[0])
                random.seed(seed);np.random.seed(seed);torch.manual_seed(seed)
                dataset=self.mixture.datasets[ds]
                transform=dataset.processor.train_image_transform
                if hasattr(transform,"set_random_seed"):transform.set_random_seed(seed)
                self.cache=dataset.get_shard(shard)
                if len(self.cache)!=int(dataset.get_shard_length(shard)):
                    raise RuntimeError("native shard length changed")
                self.cached_occurrence=occurrence
            return self.cache[offset]

    def same_tree(a,b):
        if torch.is_tensor(a) and torch.is_tensor(b):
            return a.dtype==b.dtype and a.shape==b.shape and torch.equal(a.detach().cpu(),b.detach().cpu())
        if isinstance(a,np.ndarray) and isinstance(b,np.ndarray):return np.array_equal(a,b)
        if isinstance(a,dict) and isinstance(b,dict):
            return a.keys()==b.keys() and all(same_tree(a[k],b[k]) for k in a)
        if isinstance(a,(list,tuple)) and isinstance(b,(list,tuple)):
            return len(a)==len(b) and all(same_tree(x,y) for x,y in zip(a,b))
        return a==b

    def sample_hash(model,trainable):
        h=hashlib.sha256()
        for name,p in model.named_parameters():
            if p.requires_grad==trainable:
                h.update(name.encode())
                h.update(p.detach().reshape(-1)[:64].cpu().contiguous().view(torch.uint8).numpy().tobytes())
        return h.hexdigest()

    class WorkbenchCallback(TrainerCallback):
        def __init__(self,trainer):
            self.trainer=trainer;self.start=time.monotonic();self.initial_trainable=None;self.initial_frozen=None
        def on_train_begin(self,a,s,c,model=None,**kw):
            if preflight:
                self.initial_trainable=sample_hash(model,True);self.initial_frozen=sample_hash(model,False)
            if rank==0:
                atomic(output/'status.json',dict(status='training',step=resumed_step,total_steps=STEPS,world_size=GPU))
        def on_step_end(self,a,s,c,**kw):
            step=s.global_step
            if step%a.logging_steps==0 or step>=cap:
                totals=torch.cat((self.trainer.part_sum,self.trainer.part_count))
                dist.all_reduce(totals)
                self.trainer.part_latest=dict(zip(('base','torso','left_arm','left_gripper','right_arm','right_gripper'),
                    (totals[:6]/totals[6:].clamp_min(1)).tolist()))
                self.trainer.part_sum.zero_();self.trainer.part_count.zero_()
            control=read(output/"stop.json") if rank==0 and (output/"stop.json").exists() else {}
            stop=step>=int(control.get("stop_after_step",STEPS))
            # Read only rank0 control, broadcast decision, all ranks save.
            flag=torch.tensor(int(stop or step>=cap),device=torch.device("cuda",local))
            dist.broadcast(flag,src=0)
            if flag.item():
                c.should_save=True;c.should_training_stop=True;c.should_log=True
            if rank==0:
                atomic(output/"status.json",dict(status="training",step=step,total_steps=STEPS,
                    world_size=GPU,seen_samples=step*GPU*MICROBATCH*ACCUM))
            return c
        def on_log(self,a,s,c,logs=None,**kw):
            if rank!=0 or not logs or "loss" not in logs:return
            if not math.isfinite(logs["loss"]) or not math.isfinite(logs.get("grad_norm",float("nan"))):
                raise FloatingPointError("nonfinite native loss/gradient")
            row=dict(step=s.global_step,fm=logs["loss"],state=0.,grad_norm=logs["grad_norm"],
                     lr=logs.get("learning_rate"),seen_samples=s.global_step*64,
                     elapsed_seconds=time.monotonic()-self.start,
                     gpu_peak_allocated=torch.cuda.max_memory_allocated(local),part_loss=self.trainer.part_latest)
            with (output/"metrics.jsonl").open("a") as f:f.write(json.dumps(row)+"\n")
        def on_save(self,a,s,c,**kw):
            dist.barrier() # official callback has copied processor files; RNG all ranks saved
            if preflight:
                model=kw['model']
                if sample_hash(model,True)==self.initial_trainable or sample_hash(model,False)!=self.initial_frozen:
                    raise RuntimeError('preflight parameter update/freeze check failed')
                if resume and not all(self.trainer.restore_checks.values()):
                    raise RuntimeError('strict resume checks incomplete')
            if rank==0:publish(job_root,native,s.global_step,contract,manifest)
            dist.barrier()
            return c

    class CheckedTrainer(NativeTrainer):
        def __init__(self,*a,**kw):
            super().__init__(*a,**kw)
            model=self.model
            if any(p.requires_grad for p in model.backbone.parameters()):
                raise RuntimeError("unexpected trainable Eagle")
            trainable=[p for p in model.parameters() if p.requires_grad]
            if not trainable or any(p.dtype!=torch.float32 for p in trainable):
                raise RuntimeError("expected FP32 trainable action expert")
            if self.args.per_device_train_batch_size!=MICROBATCH or self.args.gradient_accumulation_steps!=ACCUM:
                raise RuntimeError("native trainer batch mismatch")
            self.args.ignore_data_skip=True
            self.args.save_total_limit=None
            # Preserve upstream native train/compute_loss/optimizer/scheduler.
            # Only the dataloader's restart semantics and workbench hooks differ.
            self.indexed=IndexedShards(self.train_dataset)
            self.restore_checks=dict(model=False,optimizer=False,scheduler=False,rng=False) if resume and preflight else {}
            self.part_sum=torch.zeros(6,device=torch.device('cuda',local),dtype=torch.float64)
            self.part_count=torch.zeros_like(self.part_sum)
            self.part_latest={}
            dist.barrier()
            assert_processor(native/"processor",prepared)
            if rank==0:
                atomic(output/"native_runtime.json",dict(trainable=sum(p.numel() for p in trainable),
                    total=sum(p.numel() for p in model.parameters()),batch=MICROBATCH,accumulation=ACCUM,
                    scheduler=str(self.args.lr_scheduler_type),optimizer=str(self.args.optim)))
        def get_train_dataloader(self):
            # Explicit cursor, no reseed or replay of discarded batches.
            gen=torch.Generator().manual_seed(42+rank)
            return DataLoader(self.indexed,batch_size=MICROBATCH,
                sampler=range(cursor(resumed_step),cursor(STEPS)),
                collate_fn=self.data_collator,num_workers=args.workers,
                persistent_workers=True,pin_memory=True,generator=gen,
                multiprocessing_context=self.multiprocessing_context,drop_last=True)
        def _load_from_checkpoint(self,resume_from_checkpoint,model=None):
            result=super()._load_from_checkpoint(resume_from_checkpoint,model)
            if preflight and resume:
                # Check every serialized tensor on rank0, not just a reload success message.
                if rank==0:
                    from safetensors import safe_open
                    actual=(model or self.model).state_dict()
                    for p in sorted(resume.glob("*.safetensors")):
                        with safe_open(str(p),framework="pt",device="cpu") as f:
                            for name in f.keys():
                                if name not in actual or not torch.equal(actual[name].detach().cpu(),f.get_tensor(name)):
                                    raise RuntimeError("resumed model tensor mismatch: "+name)
                dist.barrier()
                self.restore_checks["model"]=True
            return result
        def _load_optimizer_and_scheduler(self,checkpoint):
            result=super()._load_optimizer_and_scheduler(checkpoint)
            if preflight and resume:
                if rank==0:
                    saved=torch.load(resume/"optimizer.pt",map_location="cpu",weights_only=False,mmap=True)
                    if not same_tree(self.optimizer.state_dict(),saved):
                        raise RuntimeError("resumed optimizer mismatch")
                    del saved
                if not same_tree(self.lr_scheduler.state_dict(),torch.load(resume/"scheduler.pt",map_location="cpu",weights_only=False)):
                    raise RuntimeError("resumed scheduler mismatch")
                dist.barrier()
                self.restore_checks["optimizer"]=True;self.restore_checks["scheduler"]=True
            return result
        def _load_rng_state(self,checkpoint):
            result=super()._load_rng_state(checkpoint)
            if preflight and resume:
                saved=torch.load(resume/f"rng_state_{rank}.pth",map_location="cpu",weights_only=False)
                cuda=saved.get("cuda")
                actual_cuda=torch.cuda.get_rng_state_all() if isinstance(cuda,list) else torch.cuda.get_rng_state(local)
                if not all((same_tree(saved["cpu"],torch.get_rng_state()),
                            same_tree(saved["python"],random.getstate()),
                            same_tree(saved["numpy"],np.random.get_state()),
                            same_tree(cuda,actual_cuda))):
                    raise RuntimeError("resumed per-rank RNG mismatch")
                self.restore_checks["rng"]=True
            return result
        def compute_loss(self,model,inputs,return_outputs=False,**kw):
            loss,result=super().compute_loss(model,inputs,return_outputs=True,**kw)
            if not torch.isfinite(loss):raise FloatingPointError("nonfinite native FM loss")
            if model.training:
                action=result["action_loss"].detach().float()
                mask=result["action_mask"].detach().float()
                for i,(lo,hi) in enumerate(((0,3),(3,7),(7,14),(14,15),(15,22),(22,23))):
                    self.part_sum[i]+=action[...,lo:hi].sum()
                    self.part_count[i]+=mask[...,lo:hi].sum()
            return (loss,result) if return_outputs else loss
        def training_step(self,model,inputs,*a,**kw):
            mask=inputs["inputs"]["action_mask"]
            if tuple(mask.shape)!=(MICROBATCH,50,128) or mask.sum().item()!=MICROBATCH*32*23:
                raise ValueError("invalid native action mask")
            return super().training_step(model,inputs,*a,**kw)
        def train(self,resume_from_checkpoint=None,**kw):
            # exp.run adds upstream formatting callback before calling train.
            self.add_callback(WorkbenchCallback(self))
            return super().train(resume_from_checkpoint=str(resume) if resume else None,**kw)
        def save_model(self,output_dir=None,_internal_call=False):
            # Official final save would duplicate model outside the canonical
            # checkpoint namespace; all real saves have explicit output_dir.
            if output_dir is None and not _internal_call:return
            return super().save_model(output_dir,_internal_call=_internal_call)

    exp.Gr00tTrainer=CheckedTrainer
    exp.run(config)
    dist.barrier()
    if rank==0:
        latest=read(output/"checkpoints/latest.json");step=latest["step"]
        if step<=resumed_step:raise ValueError("no optimizer update after start/resume")
        final=Path(latest["path"])
        if preflight and step>=4:
            rows=[json.loads(l) for l in (output/"metrics.jsonl").read_text().splitlines()]
            if (not resume or resumed_step>=step ):
                raise ValueError("preflight needs an actual workbench resume")
            if not list((job_root/"config").glob("resume-event-*.json")):
                raise ValueError("missing workbench resume event")
            if not all(any(r["step"]==i and r["grad_norm"]>0 and math.isfinite(r["fm"]) for r in rows)
                       for i in range(1,step+1)):
                raise ValueError("preflight did not establish finite nonzero gradients")
            atomic(output/"preflight.json",dict(passed=True,strict_resume=True,contract=contract,
                run_id=run,resumed_from=resumed_step,step=step,checkpoint=final.name,
                cursor=cursor(step),restore_checks={"model":True,"optimizer":True,"scheduler":True,"rng":True},rows=rows))
        terminal="completed" if (step>=4 if preflight else step>=STEPS) else "paused"
        atomic(output/"status.json",dict(status=terminal,step=step,total_steps=STEPS,
                    world_size=GPU,training_complete=terminal=="completed"))
    dist.barrier()
    dist.destroy_process_group()


def main():
    p=argparse.ArgumentParser(__doc__)
    p.add_argument("--job-root",type=Path,required=True)
    p.add_argument("--resume",type=Path)
    a=p.parse_args()
    worker(a)


if __name__=="__main__":
    main()
