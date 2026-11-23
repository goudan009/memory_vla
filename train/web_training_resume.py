"""Resume with the saved training code, including older no-validation runs."""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import time


def read(path):
    return json.loads(path.read_text()) if path.exists() else {}


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary=path.with_suffix('.pending')
    temporary.write_text(json.dumps(value,indent=2))
    temporary.replace(path)


def run(config):
    spec=read(config);root=Path(spec['output']);arm=spec['arm']
    env=dict(os.environ,PYTHONPATH=str(Path(spec['source'])/'src'),PYTHONNOUSERSITE='1',
        PYTHONDONTWRITEBYTECODE='1',HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1',
        NO_ALBUMENTATIONS_UPDATE='1',OMP_NUM_THREADS='4')
    # Older public launchers reject no-validation resumes; their saved DDP
    # workers already restore the model, AdamW, rank cursors and memory correctly.
    argv=spec['argv'] if spec['validation_enabled'] else spec['worker_argv']
    if not spec['validation_enabled']:
        env['CUDA_VISIBLE_DEVICES']=','.join(map(str,spec['gpus']))
    if spec.get('monitor_interval'):
        launch_index = argv.index('--module')
        argv = [*argv[:launch_index], str(Path(__file__).resolve()), '--instrument-worker',
                *argv[launch_index+2:]]
        env['MEMORYVLA_RESUME_MONITOR_INTERVAL'] = str(spec['monitor_interval'])
    child=None
    def record_restored():
        write(root/arm/f"config/resume-step-{spec['resume_step']:06d}.json",dict(
            checkpoint=dict(step=spec['resume_step'],path=spec['checkpoint']),
            optimizer_restored=True,rank_memory_and_cursor_restored=True,
            total_steps=spec['steps'],source=spec['source']))
    try:
        with (root/f'logs/{arm}_torchrun.log').open('ab') as log:
            child=subprocess.Popen(argv,cwd=spec['source'],env=env,stdin=subprocess.DEVNULL,
                stdout=None if spec['validation_enabled'] else log,stderr=subprocess.STDOUT)
            write(root/'status.json',dict(status='training',pid=os.getpid(),
                processes={'resume_child':child.pid},resume_step=spec['resume_step'],max_steps=spec['steps']))
            recorded=False
            while child.poll() is None:
                state=read(root/arm/'status.json')
                if not recorded and state.get('step',0)>spec['resume_step']:
                    # A completed DDP update proves every rank restored successfully.
                    record_restored();recorded=True
                time.sleep(1)
        if child.returncode:
            raise RuntimeError(f'Resumed training exited {child.returncode}; see logs/{arm}_torchrun.log')
        if not spec['validation_enabled']:
            state=read(root/arm/'status.json')
            if not state.get('training_complete') or state.get('step',0)<=spec['resume_step']:
                raise RuntimeError('Worker exited without completing a resumed training update')
            if not recorded: record_restored()
            write(root/'status.json',dict(status='completed',steps_each=state['step']))
    except BaseException as error:
        write(root/'status.json',dict(status='failed',error=repr(error)))
        state=read(root/arm/'status.json')
        write(root/arm/'status.json',dict(state,status='failed',training_complete=False,error=repr(error)))
        raise
    finally:
        if child is not None and child.poll() is None:
            child.terminate();child.wait()


def instrument_worker():
    """Measure the saved worker without changing its loss or update."""
    from datetime import timedelta
    import torch
    import torch.distributed as dist
    from v1_groot_memoryvla.experiment import task21_ddp as worker
    from v1_groot_memoryvla.data.convert_b1k import ACTION_SLICES

    interval = int(os.environ['MEMORYVLA_RESUME_MONITOR_INTERVAL'])
    groups = [(name, *limits) for name, limits in ACTION_SLICES.items()]
    initialize = worker.initialize
    backward = worker.backward_windows
    save = worker.checkpoint
    clip = torch.nn.utils.clip_grad_norm_
    adam_step = torch.optim.AdamW.step
    sample = {}

    def observe_head(_, inputs, output):
        if sample.get('losses') is None:
            return
        loss = output['action_loss'].detach()
        mask = inputs[1]['action_mask']
        for row, (_, start, end) in enumerate(groups):
            sample['losses'][row, 0] += loss[..., start:end].sum(dtype=torch.float64)
            sample['losses'][row, 1] += mask[..., start:end].sum(dtype=torch.float64)

    def init(args, checkpoint):
        processor, train, validation, model = initialize(args, checkpoint)
        model.action_head.register_forward_hook(observe_head)
        sample.update(step=args.resume_step, first=args.resume_step+1, last=args.steps,
                      model=model, dataset=train, output=args.output/args.arm, losses=None)
        return processor, train, validation, model

    def windows(*args, **kwargs):
        sample['step'] += 1
        step = sample['step']
        sample['losses'] = (torch.zeros(len(groups), 2, dtype=torch.float64, device='cuda')
                           if step in (sample['first'], sample['last']) or step % interval == 0 else None)
        return backward(*args, **kwargs)

    def grad_clip(*args, **kwargs):
        norm = clip(*args, **kwargs)
        if sample.get('losses') is not None:
            sample['pre_clip'] = float(norm)
        return norm

    def update(optimizer, *args, **kwargs):
        losses = sample.get('losses')
        if losses is None:
            return adam_step(optimizer, *args, **kwargs)
        dist.all_reduce(losses)
        rows = {}
        if dist.get_rank() == 0:
            for name, p in sample['model'].memory.named_parameters():
                if p.requires_grad:
                    rows[name] = (p, p.detach().clone(), None if p.grad is None else p.grad.detach().square().sum(),
                                  None if p.grad is None else p.grad.detach().count_nonzero())
        result = adam_step(optimizer, *args, **kwargs)
        if rows:
            accumulated = {}
            for name, (p, before, norm2, nonzero) in rows.items():
                group = name.split('.')[0]
                stats = accumulated.setdefault(group, torch.zeros(5, dtype=torch.float64, device=p.device))
                stats[0] += before.square().sum(dtype=torch.float64)
                stats[1] += (p.detach()-before).square().sum(dtype=torch.float64)
                if norm2 is not None:
                    stats[2] += norm2
                    stats[3] += nonzero
                stats[4] += p.numel()
            metrics = {}
            for group, summary in accumulated.items():
                denominator, delta, grad, count, total = summary.tolist()
                metrics[group] = dict(relative_update=delta**.5/(denominator**.5+1e-12),
                                      grad_norm_after_clip=grad**.5, nonzero_gradient_fraction=count/total)
            worker.append_record(sample['output']/'metrics/train_monitor.jsonl',
                dict(step=sample['step'], lr=optimizer.param_groups[0]['lr'], pre_clip=sample['pre_clip'],
                     action_loss={name: value/count if count else None
                                  for (name, *_), (value, count) in zip(groups, losses.tolist())},
                     memory=metrics))
        return result

    def checkpoint(model, processor, optimizer, root, args, step, epoch, position, control_group=None):
        path = save(model, processor, optimizer, root, args, step, epoch, position, control_group)
        if dist.get_rank() == 0:
            state = path/'training_state'
            state.mkdir(exist_ok=True)
            names = ['latest_optimizer.pt'] + [
                f'latest_rank-{rank:03d}.pt' for rank in range(dist.get_world_size())]
            for name in names:
                temporary = state/(name+'.pending')
                temporary.unlink(missing_ok=True)
                os.link(root/'checkpoints'/name, temporary)
                temporary.replace(state/name)
            manifest = read(root/'checkpoints/latest.json')
            write(state/'latest.json', dict(manifest, scheduler=None, grad_scaler=None,
                  amp='bfloat16 autocast', rng='seed + 100000 * (step-1) + 1000 * virtual_slot + chunk_start'))
            if step in (6000, 8000, 10000, args.steps):
                verify_reload(model, processor, args, step, epoch, position, path, state)
        dist.monitored_barrier(group=control_group, timeout=timedelta(minutes=5), wait_all_ranks=True)
        return path

    @torch.no_grad()
    def verify_reload(model, processor, args, step, epoch, position, path, state):
        dataset = sample['dataset']
        plan = worker.EpisodePlan(dataset, args.global_windows_per_step, args.seed+epoch)
        if position == plan.steps:
            plan = worker.EpisodePlan(dataset, args.global_windows_per_step, args.seed+epoch+1)
            position = 0
        batch = plan.batch(position)
        owned = range(0, args.global_windows_per_step, dist.get_world_size())
        slot = next((index for index in owned if batch[index] is not None), None)
        if slot is None:
            write(state/'reload_check.json', dict(step=step, reason='rank0 has no next window'))
            return
        seed = args.seed + 100000 * step + 1000 * slot
        runtime = worker.memory_runtime(model)
        sequence = worker.cache_sequence(model, processor, dataset, batch[slot], args.microbatch,
                                         augmentation_seed=seed)

        def forward(candidate):
            end = min(args.microbatch, len(sequence['timesteps']))
            with torch.autocast('cuda', dtype=torch.bfloat16):
                features = candidate.memory(worker.slice_features(sequence['features'], 0, end),
                    sequence['input_ids'][:end].cuda(), worker.window_context(sequence, 0, end))
                worker.seeded(seed)
                result = candidate.action_head(features, worker.slice_features(sequence['actions'], 0, end))
            return features['backbone_features'].float().cpu(), result['action_loss'].float().cpu()

        try:
            expected = forward(model)
        finally:
            worker.restore_memory_runtime(model, runtime, 'cuda')
        _, _, _, reloaded = initialize(args, path)
        reloaded.train()
        reloaded.backbone.eval()
        saved = torch.load(state/'latest_rank-000.pt', map_location='cpu', weights_only=True, mmap=True)
        worker.restore_memory_runtime(reloaded, saved['memory_runtime'], 'cuda')
        actual = forward(reloaded)
        maximum = [(a-b).abs().max().item() for a,b in zip(expected, actual)]
        write(state/'reload_check.json', dict(step=step, episode=sequence['episode'],
              frames=sequence['frame_indices'][:args.microbatch], slot=slot, seed=seed,
              fused_condition_max_abs=maximum[0], action_loss_max_abs=maximum[1]))
        if maximum != [0., 0.]:
            raise RuntimeError(f'Saved check failed at step {step}: {maximum}')

    worker.initialize = init
    worker.backward_windows = windows
    worker.checkpoint = checkpoint
    torch.nn.utils.clip_grad_norm_ = grad_clip
    torch.optim.AdamW.step = update
    worker.main()


if __name__=='__main__' and sys.argv[1:2] == ['--instrument-worker']:
    del sys.argv[1]
    instrument_worker()
elif __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config',type=Path,required=True)
    run(parser.parse_args().config)
