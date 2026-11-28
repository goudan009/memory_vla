#!/usr/bin/env python3
"""Workbench worker for the bounded native N1.7 training acceptance."""
import argparse
import hashlib
import importlib.metadata
import json
import math
import os
from pathlib import Path
import sys
import time
import traceback

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))


def write_json(path, value):
    temporary = path.with_suffix('.pending')
    temporary.write_text(json.dumps(value, indent=2))
    temporary.replace(path)


def native_config(job, plan, manifest, resume_checkpoint=None):
    """Build the pinned upstream configuration without loading model/GPU state."""
    upstream = Path(plan['upstream'])
    request = plan['request']
    sys.path.insert(0, str(upstream))
    from gr00t.configs.base_config import get_default_config
    from gr00t.configs.finetune_config import FinetuneConfig
    from gr00t.eval.eval_b1k_wrapper import load_modality_config
    ft = FinetuneConfig(**request['finetune'], base_model_path=request['base_checkpoint'],
        dataset_path=manifest['views']['train']['path'], embodiment_tag='NEW_EMBODIMENT',
        output_dir=str(job / 'native'), experiment_name=None, num_gpus=1,
        save_only_model=False, skip_weight_loading=False, resume_from_checkpoint=bool(resume_checkpoint))
    load_modality_config(str(upstream / 'examples/b1k/r1pro.py'))
    config = get_default_config().load_dict({'data': {'download_cache': False, 'datasets': [dict(
        dataset_paths=[ft.dataset_path], mix_ratio=1.0, embodiment_tag='new_embodiment')]}})
    config.load_config_path = None
    for key in ('tune_llm', 'tune_visual', 'tune_projector', 'tune_diffusion_model', 'state_dropout_prob',
                'random_rotation_angle', 'color_jitter_params'):
        setattr(config.model, key, getattr(ft, key))
    config.model.load_bf16 = False
    config.model.reproject_vision = False
    config.model.backbone_trainable_params_fp32 = True
    config.model.use_relative_action = True
    for key in ('global_batch_size', 'gradient_accumulation_steps', 'dataloader_num_workers',
                'learning_rate', 'weight_decay', 'warmup_ratio', 'max_steps', 'save_steps'):
        setattr(config.training, key, getattr(ft, key))
    config.training.start_from_checkpoint = ft.base_model_path
    config.training.optim = 'adamw_torch'
    config.training.output_dir = str(job / 'native')
    config.training.experiment_name = None
    config.training.num_gpus = 1
    config.training.use_wandb = False
    config.training.logging_steps = 1
    config.training.save_steps = ft.save_steps
    config.training.save_total_limit = ft.max_steps + 1
    config.training.save_only_model = False
    config.training.resume_from_checkpoint = str(resume_checkpoint) if resume_checkpoint else False
    config.data.seed = request['seed']
    config.data.video_backend = 'opencv'
    config.data.decode_only_used_frames = True
    for key in ('shard_size', 'episode_sampling_rate', 'num_shards_per_epoch'):
        setattr(config.data, key, getattr(ft, key))
    config.training.gradient_checkpointing = request['gradient_checkpointing']
    return config, ft


def train(job, resume_checkpoint=None):
    from v1_2_groot_baseline.b1k2026_training_data import digest, verify_manifest
    from v1_2_groot_baseline.b1k2026_resume import resumable_type
    from v1_2_groot_baseline.contracts import (inspect_upstream, inspect_checkpoint, checkpoint_digest,
        inspect_training_checkpoint, TRAINING_STATE_FILES, file_sha256)

    plan = json.loads((job / 'job.json').read_text())
    request = plan['request']
    for name, expected in plan['adapter_source_sha256'].items():
        if digest(Path(__file__).resolve().parents[1] / name) != expected:
            raise ValueError('adapter source changed since preview: ' + name)
    upstream = Path(plan['upstream'])
    inspect_upstream(upstream)
    inspect_checkpoint(request['base_checkpoint'], role='generic')
    if checkpoint_digest(request['base_checkpoint']) != request['base_sha256']:
        raise ValueError('base checkpoint changed since preview')
    manifest = verify_manifest(request['data_manifest'])
    if digest(request['data_manifest']) != plan['data_manifest_sha256']:
        raise ValueError('training data manifest changed')
    if resume_checkpoint:
        identity = inspect_training_checkpoint(resume_checkpoint)
        if (identity['recipe_sha256'] != plan['recipe_sha256'] or identity['run_id'] != request['run_id']
                or identity['data_manifest_sha256'] != plan['data_manifest_sha256']):
            raise ValueError('resume training lineage changed')
    config, ft = native_config(job, plan, manifest, resume_checkpoint)
    import torch
    from transformers import TrainerCallback
    from gr00t.experiment import experiment
    from gr00t.data.dataset import factory
    factory.ShardedMixtureDataset = resumable_type(factory.ShardedMixtureDataset)
    write_json(job / 'runtime_environment.json', dict(python=sys.version, executable=sys.executable,
        torch=torch.__version__, cuda=torch.version.cuda,
        packages={name:importlib.metadata.version(name) for name in
                  ('transformers','accelerate','albumentations','pyarrow')},
        upstream_revision=plan['source_revision'], adapter_source_sha256=plan['adapter_source_sha256']))

    holder = {}
    metrics = []
    snapshots = {}
    paused = False
    started = time.time()
    previous = dict(time=started, samples=0)
    if (job / 'metrics.jsonl').is_file():
        metrics = [json.loads(line) for line in (job / 'metrics.jsonl').read_text().splitlines()]

    class Observer(TrainerCallback):
        def on_train_begin(self, args, state, control, model=None, optimizer=None, **kwargs):
            expected = {name: p for name, p in model.named_parameters() if p.requires_grad}
            actual = {id(p) for group in optimizer.param_groups for p in group['params']}
            if not expected or any(id(p) not in actual for p in expected.values()):
                raise RuntimeError('trainable parameters missing from optimizer')
            write_json(job / 'trainable_parameters.json', {name: list(p.shape) for name, p in expected.items()})
            if resume_checkpoint:
                if not optimizer.state:
                    raise RuntimeError('resume did not restore optimizer state')
                steps = [float(value['step']) for value in optimizer.state.values() if 'step' in value]
                if not steps or max(steps) != state.global_step:
                    raise RuntimeError('restored optimizer steps differ from trainer step')
                with (job / 'resume_events.jsonl').open('a') as stream:
                    stream.write(json.dumps(dict(checkpoint=str(resume_checkpoint),step=state.global_step,
                        optimizer_entries=len(optimizer.state),optimizer_max_step=max(steps),
                        scheduler=holder['trainer'].lr_scheduler.state_dict(),
                        training_state_sha256=identity['training_state_sha256']))+'\n')

        def on_pre_optimizer_step(self, args, state, control, model=None, **kwargs):
            snapshots.clear()
            for name, p in model.named_parameters():
                if p.requires_grad and p.grad is not None:
                    if not torch.isfinite(p.grad).all():
                        raise RuntimeError('nonfinite gradient: ' + name)
                    if len(snapshots) < 4 and p.grad.float().norm().item() > 0:
                        position = p.grad.detach().flatten().abs().argmax().item()
                        snapshots[name] = (p, position, p.detach().flatten()[position].float().cpu().clone(),
                                           p.grad.detach().float().norm().item())
            if not snapshots:
                raise RuntimeError('no nonzero trainable gradient')

        def on_step_end(self, args, state, control, **kwargs):
            nonlocal paused
            delta = {name: (p.detach().flatten()[position].float().cpu() - before).abs().item()
                     for name, (p, position, before, _) in snapshots.items()}
            now = time.time()
            samples = holder['trainer'].train_dataset.cursor['samples']
            row = dict(step=state.global_step, elapsed_seconds=now - started,
                lr=holder['trainer'].get_learning_rates()[0] if hasattr(holder['trainer'], 'get_learning_rates')
                   else holder['trainer'].optimizer.param_groups[0]['lr'],
                gradient_norms={name: g for name, (_, _, _, g) in snapshots.items()}, parameter_delta=delta,
                samples=samples, step_seconds=now-previous['time'],
                samples_per_second=(samples-previous['samples'])/(now-previous['time']),
                cuda_peak_bytes=torch.cuda.max_memory_allocated())
            previous.update(time=now, samples=samples)
            metrics.append(row)
            with (job / 'metrics.jsonl').open('a') as stream:
                stream.write(json.dumps(row) + '\n')
            write_json(job / 'status.json', dict(state='running', step=state.global_step, pid=os.getpid()))
            if (job / 'pause.request').exists() or (not resume_checkpoint and request.get('pause_after_steps') == state.global_step):
                paused = True
                control.should_save = control.should_training_stop = True
            if state.global_step >= ft.max_steps:
                control.should_save = True

        def on_log(self, args, state, control, logs=None, **kwargs):
            if logs and 'loss' in logs:
                if not math.isfinite(logs['loss']):
                    raise RuntimeError('nonfinite logged loss')
                with (job / 'loss.jsonl').open('a') as stream:
                    stream.write(json.dumps(dict(step=state.global_step, **logs)) + '\n')

    original_trainer = experiment.Gr00tTrainer

    class CursorTrainer(original_trainer):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            holder['trainer'] = self
            self.add_callback(Observer())

        def get_train_dataloader(self):
            # Exact cursor restore replaces the upstream reseed-on-resume path.
            self.args.ignore_data_skip = True
            if resume_checkpoint:
                self.train_dataset.load_state_dict(json.loads((resume_checkpoint / 'dataset_state.json').read_text()))
                previous['samples'] = self.train_dataset.cursor['samples']
            generator = torch.Generator().manual_seed(request['seed'])
            return torch.utils.data.DataLoader(self.train_dataset, batch_size=self._train_batch_size,
                collate_fn=self.data_collator, num_workers=0, generator=generator)

        def compute_loss(self, *args, **kwargs):
            result = super().compute_loss(*args, **kwargs)
            loss = result[0] if isinstance(result, tuple) else result
            if not torch.isfinite(loss).all():
                raise RuntimeError('nonfinite actual training loss')
            return result

    original_format = experiment.CheckpointFormatCallback

    class PublishCheckpoint(original_format):
        def on_save(self, args, state, control, **kwargs):
            super().on_save(args, state, control, **kwargs)
            checkpoint = Path(args.output_dir) / f'checkpoint-{state.global_step}'
            write_json(checkpoint / 'dataset_state.json', holder['trainer'].train_dataset.state_dict())
            required = ['optimizer.pt', 'scheduler.pt', 'rng_state.pth', 'trainer_state.json', 'processor_config.json', 'statistics.json']
            if any(not (checkpoint / name).is_file() for name in required):
                raise RuntimeError('native checkpoint is missing complete training state')
            identity = dict(benchmark='b1k2026', version='V_1_2', run_id=request['run_id'],
                optimizer_step=state.global_step, tasks=manifest['views']['train']['tasks'],
                source_revision=plan['source_revision'], base_sha256=request['base_sha256'],
                adapter_source_sha256=plan['adapter_source_sha256'],
                data_manifest_sha256=plan['data_manifest_sha256'], recipe_sha256=plan['recipe_sha256'],
                language_mode=manifest['language_mode'], task_prompts=manifest['views']['train']['task_prompts'],
                training_state_sha256={name:file_sha256(checkpoint / name) for name in TRAINING_STATE_FILES},
                checkpoint_sha256=checkpoint_digest(checkpoint), complete=True,
                parameter_update_verified=any(any(v > 0 for v in row['parameter_delta'].values()) for row in metrics))
            write_json(checkpoint / 'b1k2026_training.json', identity)
            canonical = job / 'checkpoints' / f"{request['run_id']}__step{state.global_step:06d}"
            canonical.parent.mkdir(exist_ok=True)
            if canonical.exists():
                if canonical.resolve() != checkpoint.resolve():
                    raise ValueError('canonical checkpoint already belongs to another output')
            else:
                canonical.symlink_to(checkpoint, target_is_directory=True)
            write_json(job / 'checkpoint_index.json', dict(checkpoint=str(canonical), **identity))

    experiment.Gr00tTrainer = CursorTrainer
    experiment.CheckpointFormatCallback = PublishCheckpoint
    config.save(job / 'effective_config.yaml')
    experiment.run(config)
    index = json.loads((job / 'checkpoint_index.json').read_text())
    if not index['parameter_update_verified']:
        raise RuntimeError('no measured parameter update; training is not accepted')
    write_json(job / 'status.json', dict(state='paused' if paused else 'complete',
        step=index['optimizer_step'], checkpoint=index['checkpoint'], pid=os.getpid()))
    (job / 'training_report.md').write_text(
        f"# B1K-2026 engineering training\n\nRun: {request['run_id']}\n\n"
        f"Optimizer steps: {index['optimizer_step']}\n\nState: {'paused' if paused else 'complete'}\n\n"
        f"Checkpoint: {index['checkpoint']}\n\nParameter update measured: true. "
        "Model reload, strict-resume comparison and simulation task success require their separate acceptance.\n")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--job', required=True, type=Path)
    parser.add_argument('--resume-checkpoint', type=Path)
    args = parser.parse_args()
    try:
        train(args.job, args.resume_checkpoint)
    except Exception as error:
        traceback.print_exc()
        write_json(args.job / 'status.json', dict(state='failed', error=str(error), pid=os.getpid()))
        return 1
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
