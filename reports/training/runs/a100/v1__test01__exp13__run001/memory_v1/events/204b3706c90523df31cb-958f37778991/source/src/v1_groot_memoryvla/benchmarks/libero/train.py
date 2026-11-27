"""LIBERO training through the existing GR00T/MemoryVLA model and flow loss."""
import argparse
import json
from pathlib import Path
import signal
import time

import torch

from .dataset import LiberoGroups, make_processor
from .protocol import CONTRACT, DATA_REVISION, HORIZONS, atomic_json, read_json
from v1_groot_memoryvla.experiment.first_training import cache_sequence, seeded
from v1_groot_memoryvla.experiment.task21_diagnostic import load_model
from v1_groot_memoryvla.experiment.task21_formal import fused_window
from v1_groot_memoryvla.experiment.finetuning_options import add_finetuning_options


def train(args):
    torch.set_num_threads(4)
    args.output.mkdir(parents=True, exist_ok=True)
    dataset = LiberoGroups(args.suite, args.tasks, group_size=args.group_size, seed=args.seed)
    previous = read_json(args.checkpoint/'libero.json')
    signature = dict(suite=args.suite, tasks=sorted(args.tasks), arm=args.arm,
                     fine_tuning_mode=args.fine_tuning_mode, group_size=args.group_size,
                     seed=args.seed, learning_rate=args.learning_rate,
                     memory_length=16, contract=CONTRACT, dataset_revision=DATA_REVISION,
                     dataset_sha256={str(m['task']): m['sha256'] for m in dataset.manifests})
    if previous:
        for key, value in signature.items():
            if previous[key] != value:
                raise ValueError(f'Resume configuration mismatch: {key}')
        if args.steps <= previous['step']:
            raise ValueError('Target steps must exceed the saved checkpoint step')
        from v1_groot_memoryvla.model.gr00t_n1d6.processing_gr00t_n1d6 import Gr00tN1d6Processor
        processor = Gr00tN1d6Processor.from_pretrained(args.checkpoint)
    else:
        processor = make_processor(args.checkpoint, dataset.manifests)
    processor.train()
    dataset.processor = processor
    model = load_model(args.checkpoint, args.arm=='memory', args.seed, finetuning=args,
                       memory_conditioning='direct', memory_length=16,
                       memory_cognitive_qk_norm='layernorm')
    model.backbone.cuda().eval()
    parameters = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(parameters, lr=args.learning_rate, weight_decay=args.weight_decay, foreach=False)
    first = 0
    if previous:
        saved = torch.load(args.checkpoint/'optimizer.pt', map_location='cpu', weights_only=True, mmap=True)
        optimizer.load_state_dict(saved['optimizer']); first = saved['step']
        assert first == previous['step']
        del saved
    atomic_json(args.output/'training_config.json', dict(**signature, target_steps=args.steps,
        initialization=str(args.checkpoint), resume=bool(previous), finetuning=model.finetuning_summary))
    stopping = False
    def stop(signum, frame):
        nonlocal stopping
        stopping = True
    signal.signal(signal.SIGTERM, stop); signal.signal(signal.SIGINT, stop)
    started = time.monotonic()
    last_checkpoint = None
    def save(step):
        nonlocal last_checkpoint
        target = args.output/'checkpoints'/f'step-{step:06d}'
        temporary = target.with_name(target.name+'.saving')
        temporary.mkdir(parents=True)
        model.save_pretrained(temporary, max_shard_size='5GB')
        processor.save_pretrained(temporary)
        torch.save(dict(step=step, optimizer=optimizer.state_dict()), temporary/'optimizer.pt')
        atomic_json(temporary/'libero.json', dict(**signature, step=step))
        temporary.rename(target)
        last_checkpoint = str(target.resolve())
        atomic_json(args.output/'latest.json', dict(step=step, path=last_checkpoint))
        print(json.dumps(dict(checkpoint=last_checkpoint, step=step)), flush=True)
    with (args.output/'training.jsonl').open('a', buffering=1) as metrics:
        for index in range(first, args.steps):
            tick = time.monotonic()
            seeded(args.seed+index*1000)
            model.train(); model.backbone.eval()
            optimizer.zero_grad(set_to_none=True)
            sequence = cache_sequence(model, processor, dataset, index, args.microbatch,
                                      augmentation_seed=args.seed+index*1000)
            loss, _, _ = fused_window(model, sequence, args.seed+index*1000, backward=True, microbatch=args.microbatch)
            q_gradient = None
            if model.memory is not None:
                grad = model.memory.cognitive.retrieval_blocks[0].q_proj.weight.grad
                q_gradient = float(grad.norm()) if grad is not None else None
            norm = float(torch.nn.utils.clip_grad_norm_(parameters, args.max_grad_norm, error_if_nonfinite=True))
            optimizer.step()
            row = dict(step=index+1, loss=loss, gradient_norm=norm, cognitive_q_gradient=q_gradient,
                       task=sequence['task'], episode=sequence['episode'], frames=sequence['frame_indices'],
                       seconds=time.monotonic()-tick)
            metrics.write(json.dumps(row)+'\n'); print(json.dumps(row), flush=True)
            if (index+1)%args.save_interval==0 or index+1==args.steps or stopping:
                optimizer.zero_grad(set_to_none=True)
                save(index+1)
            if stopping: break
    report = dict(type='libero_train', passed=True, stopped=stopping, suite=args.suite,
        tasks=args.tasks, arm=args.arm, steps=index+1, checkpoint=last_checkpoint,
        training=[json.loads(line) for line in (args.output/'training.jsonl').read_text().splitlines()],
        elapsed_seconds=time.monotonic()-started,
        note='Training loss only; evaluate the saved checkpoint in LIBERO for task success')
    atomic_json(args.report, report)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--checkpoint', type=Path, required=True)
    parser.add_argument('--suite', choices=HORIZONS, required=True)
    parser.add_argument('--tasks', nargs='+', type=int, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--report', type=Path, required=True)
    parser.add_argument('--arm', choices=['baseline','memory'], default='memory')
    parser.add_argument('--steps', type=int, default=300)
    parser.add_argument('--group-size', type=int, default=16)
    parser.add_argument('--microbatch', type=int, default=1)
    parser.add_argument('--save-interval', type=int, default=100)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--learning-rate', type=float, default=1e-5)
    add_finetuning_options(parser)
    parser.set_defaults(lora_targets='action_dit')
    args = parser.parse_args()
    if min(args.steps,args.group_size,args.microbatch,args.save_interval)<1:
        parser.error('Steps and batch sizes must be positive')
    train(args)


if __name__ == '__main__': main()
