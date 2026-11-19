"""Small NCCL reproduction of checkpoint barriers; writes only to --output."""

import argparse
from datetime import timedelta
import faulthandler
import json
import os
from pathlib import Path
from types import SimpleNamespace

import torch
import torch.distributed as dist
from torch import nn
from torch.nn.parallel import DistributedDataParallel

from v3_bodymemory.experiment.task21_ddp import checkpoint, TrainableStep, EpisodePlan, backward_windows


class Probe(nn.Module):
    def __init__(self):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(1024, 1024, device="cuda"))
        self.memory = None

    def forward(self):
        return self.weight.sum()

    def save_pretrained(self, directory, **kwargs):
        directory.mkdir(parents=True, exist_ok=True)
        torch.save(self.state_dict(), directory / "probe.pt")


class Processor:
    def save_pretrained(self, directory):
        pass


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--bind-device", action="store_true")
    p.add_argument("--real-checkpoint", type=Path)
    p.add_argument("--dataset", type=Path)
    p.add_argument("--host-control", action="store_true")
    p.add_argument("--memory", action="store_true")
    args = p.parse_args()
    device = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(device)
    torch.set_num_threads(1)
    faulthandler.dump_traceback_later(25, repeat=True)
    dist.init_process_group("nccl", timeout=timedelta(seconds=70),
        **({"device_id": torch.device("cuda", device)} if args.bind_device else {}))
    control_group = dist.new_group(backend="gloo", timeout=timedelta(seconds=90)) if args.host_control else None
    original = dist.barrier

    def traced(*a, **kw):
        print(json.dumps({"rank": dist.get_rank(), "stage": "barrier_enter",
            "cuda_device": torch.cuda.current_device()}), flush=True)
        original(*a, **kw)
        print(json.dumps({"rank": dist.get_rank(), "stage": "barrier_exit"}), flush=True)

    dist.barrier = traced
    try:
        if args.real_checkpoint:
            from v3_bodymemory.experiment.task21_diagnostic import load_model
            from v3_bodymemory.model.gr00t_n1d6.processing_gr00t_n1d6 import Gr00tN1d6Processor
            model = load_model(args.real_checkpoint, args.memory, 42)
            model.backbone.cuda()
            processor = Gr00tN1d6Processor.from_pretrained(args.real_checkpoint)
            ddp = DistributedDataParallel(TrainableStep(model), device_ids=[device], broadcast_buffers=False,
                find_unused_parameters=True, gradient_as_bucket_view=True)
            optimizer = torch.optim.AdamW([p for p in ddp.parameters() if p.requires_grad], lr=1e-5)
            optimizer.load_state_dict(torch.load(args.real_checkpoint.parent / "latest_optimizer.pt",
                map_location="cpu", weights_only=True, mmap=True))
            if args.dataset:
                from v3_bodymemory.data.dataset.episode_windows import EpisodeWindows
                from v3_bodymemory.experiment.first_training import cache_sequence
                from v3_bodymemory.experiment.task21_formal import restore_memory_runtime
                split = json.loads((args.real_checkpoint.parents[2] / "config/data_split.json").read_text())
                dataset = EpisodeWindows(args.dataset, processor, split["train"])
                plan = EpisodePlan(dataset, 2, 42)
                state = torch.load(args.real_checkpoint.parent / f"latest_rank-{dist.get_rank():03d}.pt",
                    map_location="cpu", weights_only=True)
                restore_memory_runtime(model, state["memory_runtime"], "cuda")
                model.train()
                model.backbone.eval()
                seq = cache_sequence(model, processor, dataset, plan.batch(state["position"])[dist.get_rank()], 4)
                optimizer.zero_grad(set_to_none=True)
                backward_windows(ddp, [seq], [42+100000*state["step"]+1000*dist.get_rank()], 2)
                torch.nn.utils.clip_grad_norm_(ddp.parameters(), 1.)
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)
                torch.cuda.synchronize()
                dist.all_reduce(torch.ones(4, device="cuda"), op=dist.ReduceOp.MAX)
                dist.all_reduce(torch.ones((), device="cuda", dtype=torch.float64))
        else:
            model = Probe()
            processor = Processor()
            ddp = DistributedDataParallel(model, device_ids=[device], broadcast_buffers=False)
            optimizer = torch.optim.AdamW(model.parameters(), lr=1e-5)
        args.output.joinpath("checkpoints").mkdir(parents=True, exist_ok=True)
        traced()
        for step in range(1, 2 if args.real_checkpoint else 4):
            if not args.real_checkpoint:
                optimizer.zero_grad(set_to_none=True)
                ddp().backward()
                optimizer.step()
            dist.all_reduce(torch.ones(4, device="cuda"))
            checkpoint(model, processor, optimizer, args.output,
                SimpleNamespace(global_windows_per_step=2, seed=42), step, 0, step,
                **({"control_group": control_group} if args.host_control else {}))
            print(json.dumps({"rank": dist.get_rank(), "stage": "checkpoint_return", "step": step}), flush=True)
        if args.real_checkpoint and args.dataset:
            seq = cache_sequence(model, processor, dataset, plan.batch(state["position"]+1)[dist.get_rank()], 4)
            optimizer.zero_grad(set_to_none=True)
            backward_windows(ddp, [seq], [42+100000*(state["step"]+1)+1000*dist.get_rank()], 2)
            torch.nn.utils.clip_grad_norm_(ddp.parameters(), 1.)
            optimizer.step()
            torch.cuda.synchronize()
            print(json.dumps({"rank": dist.get_rank(), "stage": "training_continued_after_checkpoint",
                "memory_entries": sum(len(v) for b in (model.memory.perceptual, model.memory.cognitive)
                    for v in b.bank.values()) if model.memory else 0}), flush=True)
        dist.all_reduce(torch.ones(4, device="cuda"))
        print(json.dumps({"rank": dist.get_rank(), "stage": "continued_collective"}), flush=True)
    finally:
        faulthandler.cancel_dump_traceback_later()
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
