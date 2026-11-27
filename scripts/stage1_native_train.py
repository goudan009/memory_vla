"""V2.1 Stage1 rerun: actual upstream GR00T pipeline, dataset and Gr00tTrainer.

Preview by default; 3-update disposable native smoke test and then formal run.
No change to the existing V2.1 training loop, weights or Stage2.
"""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys

from stage1_native_prepare import read, write, sha, verify_prepared, TAG, RELATIVE


STEPS = 6000
GPU = 4
MICROBATCH = 2
ACCUM = 8


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


def worker(args):
    import copy
    import numpy as np
    import torch
    from gr00t.configs.base_config import Config
    from gr00t.configs.model.gr00t_n1d6 import Gr00tN1d6Config
    from gr00t.configs.data.data_config import SingleDatasetConfig
    from gr00t.configs.data.embodiment_configs import MODALITY_CONFIGS
    import gr00t.experiment.experiment as exp

    if int(os.environ.get("WORLD_SIZE","0")) != GPU:
        raise ValueError("launch with 4 ranks using the parent command")
    steps = 3 if args.preflight else STEPS
    root = args.official_root.resolve()
    if not Path(exp.__file__).resolve().is_relative_to(root):
        raise RuntimeError("wrong gr00t import; official root required")
    prepared = args.prepared.resolve()
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
    t.save_steps=steps if args.preflight else 1000
    t.save_total_limit=7
    t.eval_strategy="no"  # Native sharded DatasetFactory asserts this.
    t.use_wandb=False
    t.transformers_local_files_only=True

    # Only an assertion wrapper; the official Trainer, optimizer, sampler,
    # forward/loss, scheduler and checkpoint callback are unchanged.
    NativeTrainer=exp.Gr00tTrainer
    class CheckedTrainer(NativeTrainer):
        def __init__(self,*a,**kw):
            super().__init__(*a,**kw)
            model=self.model
            if any(p.requires_grad for p in model.backbone.parameters()):
                raise RuntimeError("unexpected trainable VLM parameter")
            if not any(p.requires_grad for p in model.action_head.parameters()):
                raise RuntimeError("action expert frozen")
            if any(p.requires_grad and p.dtype!=torch.float32 for p in model.parameters()):
                raise RuntimeError("expected FP32 trainable masters")
            if (self.args.per_device_train_batch_size!=MICROBATCH or
                self.args.gradient_accumulation_steps!=ACCUM):
                raise RuntimeError("native trainer batch size mismatch")
            # Upstream writes identical processor artifacts on each rank.
            # Wait until every rank has finished before reading them.
            if torch.distributed.is_initialized():torch.distributed.barrier()
            assert_processor(args.output/"native/processor",prepared)
            if int(os.environ["RANK"])==0:
                write(args.output/"native_runtime.json",dict(
                    trainable=sum(p.numel() for p in model.parameters() if p.requires_grad),
                    total=sum(p.numel() for p in model.parameters()),
                    batch=self.args.per_device_train_batch_size,
                    accumulation=self.args.gradient_accumulation_steps,
                    lr=self.args.learning_rate, optimizer=str(self.args.optim),
                    scheduler=str(self.args.lr_scheduler_type)))
    exp.Gr00tTrainer=CheckedTrainer
    exp.run(config)


def validate(checkpoint, prepared, backend, episodes=None):
    """Fixed 5 full action chunks per held-out episode; native loader/codec."""
    import numpy as np
    import torch
    from gr00t.model.gr00t_n1d6.gr00t_n1d6 import Gr00tN1d6
    from gr00t.model.gr00t_n1d6.processing_gr00t_n1d6 import Gr00tN1d6Processor
    from gr00t.data.dataset.lerobot_episode_loader import LeRobotEpisodeLoader
    from gr00t.data.dataset.sharded_single_step_dataset import extract_step_data
    from gr00t.data.types import EmbodimentTag,MessageType
    assert_processor(checkpoint,prepared)
    model, info = Gr00tN1d6.from_pretrained(checkpoint,torch_dtype=torch.float32,
                                          output_loading_info=True,local_files_only=True)
    if any(info.get(k) for k in ("missing_keys","unexpected_keys","mismatched_keys","error_msgs")):
        raise RuntimeError(f"checkpoint reload mismatch {info}")
    model.backbone.bfloat16()
    model.action_head.float()
    model.cuda().eval()
    proc=Gr00tN1d6Processor.from_pretrained(checkpoint)
    proc.eval()
    mods=proc.modality_configs[TAG]
    loader=LeRobotEpisodeLoader(prepared/"validation",mods,video_backend=backend)
    tag=EmbodimentTag(TAG)
    import math
    total=0.;count=0
    errors={k:0. for k in mods["action"].modality_keys}
    units={k:0 for k in errors}
    anchors=[]
    limit=len(loader.episodes_metadata) if episodes is None else episodes
    with torch.inference_mode():
        for e in loader.episodes_metadata[:limit]:
            eid=int(e["episode_index"])
            df=loader._load_parquet_data(eid)
            idx=np.unique(np.linspace(0,len(df)-32,5,dtype=int))
            rgb=loader._load_video_data(eid,idx)
            for j,frame in enumerate(idx):
                # Select 32 real numeric frames; image is only needed at offset0.
                chunk=df.iloc[frame:frame+32].copy().reset_index(drop=True)
                for k,frames in rgb.items():
                    chunk["video."+k]=[frames[j]]*32
                step=extract_step_data(chunk,0,mods,tag,False)
                inp=proc.collator([proc([{"type":MessageType.EPISODE_STEP.value,"content":step}])])["inputs"]
                mask=inp["action_mask"]
                if tuple(mask.shape)!=(1,50,128) or mask.sum().item()!=32*23:
                    raise RuntimeError("incorrect physical action mask")
                torch.manual_seed(42+count)
                torch.cuda.manual_seed_all(42+count)
                with torch.autocast("cuda",dtype=torch.bfloat16):
                    result=model(inp)
                    value=float(result["loss"].float())
                    if not math.isfinite(value):raise FloatingPointError("nonfinite held-out FM loss")
                    total+=value
                    pred=model.get_action(inp)["action_pred"].float().cpu().numpy()
                decoded=proc.decode_action(pred,tag,{k:v[None] for k,v in step.states.items()})
                for k in errors:
                    d=np.abs(decoded[k][0,:8]-step.actions[k][:8])
                    if not np.isfinite(d).all():raise FloatingPointError("nonfinite decoded action")
                    errors[k]+=float(d.sum());units[k]+=d.size
                anchors.append([eid,int(frame)])
                count+=1
    del model,proc
    torch.cuda.empty_cache()
    return dict(checkpoint=str(checkpoint),samples=count,anchors=anchors,
                fm=total/count,mae_first8={k:errors[k]/units[k] for k in errors})


def export_final(args, manifest):
    checkpoint=args.output/"native"/f"checkpoint-{STEPS}"
    state=read(checkpoint/"trainer_state.json")
    if state["global_step"]!=STEPS:
        raise RuntimeError("native training not complete")
    assert_processor(checkpoint,args.prepared)
    cfg=read(checkpoint/"config.json")
    if (cfg.get("memory_enabled") or cfg.get("tune_llm") or cfg.get("tune_visual") or
        cfg.get("tune_top_llm_layers")!=0 or cfg["num_inference_timesteps"]!=4):
        raise ValueError("final checkpoint config mismatch")
    if not list(checkpoint.glob("*.safetensors")):
        raise ValueError("expected HF safetensors model weights")
    # A native HF checkpoint can directly become Stage2 input, without
    # changing native weights or the codec's effective statistics.
    write(checkpoint/"v2_1_stage1.json",dict(step=STEPS,split=read(args.prepared/"data_split.json"),
        raw_lo=manifest["raw_lo"],raw_hi=manifest["raw_hi"],
        statistics_sha256=sha(checkpoint/"statistics.json"),training_recipe="native_random_v2"))
    write(checkpoint/"COMPLETE.json",dict(step=STEPS,world_size=GPU,trainer="upstream_gr00t"))
    write(args.output/"latest.json",dict(step=STEPS,path=str(checkpoint.resolve())))
    return checkpoint


def launch(args):
    manifest=verify_prepared(args.prepared)
    ident=identity(args)
    steps=3 if args.preflight else STEPS
    print(json.dumps(dict(plan=plan(args,steps),identity=ident,
                          normalization=manifest["accepted_cells"]),indent=2),flush=True)
    if not args.execute:
        return
    import torch
    if torch.cuda.device_count()!=GPU:
        raise RuntimeError("explicitly expose exactly four allocated A100 GPUs")
    if not args.preflight:
        if args.accept_preflight is None:
            raise ValueError("pass successful disposable preflight.json")
        gate=read(args.accept_preflight)
        if not gate.get("passed") or gate.get("identity")!=ident or gate.get("video_backend")!=args.video_backend or gate.get("workers")!=args.workers:
            raise ValueError("preflight identity mismatch")
    args.output.mkdir(parents=True,exist_ok=False)
    write(args.output/"plan.json",plan(args,steps))
    write(args.output/"identity.json",ident)
    command=[sys.executable,"-m","torch.distributed.run","--standalone",
             "--nproc_per_node=4",str(Path(__file__).resolve()),"--worker",
             "--official-root",str(args.official_root.resolve()),"--prepared",str(args.prepared.resolve()),
             "--output",str(args.output.resolve()),"--video-backend",args.video_backend,
             "--workers",str(args.workers)]
    if args.preflight:command.append("--preflight")
    subprocess.run(command,check=True,env=os.environ.copy())
    native=args.output/"native"
    if args.preflight:
        checkpoint=native/"checkpoint-3"
        state=read(checkpoint/"trainer_state.json")
        if state["global_step"]!=3:raise RuntimeError("native smoke failed")
        import math
        rows=[r for r in state["log_history"] if "loss" in r]
        if len(rows)<3 or not all(math.isfinite(r["loss"]) and
            math.isfinite(r.get("grad_norm",float("nan"))) and r["grad_norm"]>0 for r in rows):
            raise RuntimeError("native smoke loss/grad invalid")
        val=validate(checkpoint,args.prepared,args.video_backend,episodes=2)
        write(args.output/"preflight.json",dict(passed=True,identity=ident,video_backend=args.video_backend,
            workers=args.workers,loss_rows=rows,reload_validation=val))
    else:
        # Native factory does not provide eval; use the fixed held-out set
        # on saved checkpoints after training, not as part of the train sampler.
        for step in range(1000,STEPS+1,1000):
            checkpoint=native/f"checkpoint-{step}"
            val=validate(checkpoint,args.prepared,args.video_backend)
            write(args.output/f"validation-{step:06d}.json",val)
        final=export_final(args,manifest)
        print(f"Stage1 completed: {final}",flush=True)


def main():
    p=argparse.ArgumentParser(__doc__)
    for key in ("official-root","prepared","output"):
        p.add_argument("--"+key,type=Path,required=True)
    p.add_argument("--video-backend",choices=["torchcodec","pyav","decord"],default="pyav")
    p.add_argument("--workers",type=int,default=2)
    p.add_argument("--preflight",action="store_true")
    p.add_argument("--accept-preflight",type=Path)
    p.add_argument("--execute",action="store_true")
    p.add_argument("--worker",action="store_true",help=argparse.SUPPRESS)
    a=p.parse_args()
    if a.workers<0:raise ValueError("workers must be nonnegative")
    a.prepared=a.prepared.resolve();a.output=a.output.resolve();a.official_root=a.official_root.resolve()
    root=a.official_root.resolve()
    if not (root/"gr00t/experiment/experiment.py").is_file():raise FileNotFoundError(root)
    sys.path.insert(0,str(root))
    os.environ["PYTHONPATH"]=str(root)+os.pathsep+os.environ.get("PYTHONPATH","")
    os.environ["HF_HUB_OFFLINE"]="1"
    os.environ["TRANSFORMERS_OFFLINE"]="1"
    if a.worker:
        worker(a)
    else:
        launch(a)


if __name__=="__main__":
    main()
