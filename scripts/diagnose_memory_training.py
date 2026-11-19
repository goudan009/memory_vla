"""Bounded Task-2 audit of the existing V1 Memory + GR00T implementation.

Uses real B1K specialist weights and real videos, freezes Eagle, and compares
100 matched optimization steps with/without training the original action head.
Does not change production model code or contact policy/simulation services.
"""

import argparse
from collections import defaultdict
import gc
import hashlib
import json
import math
from pathlib import Path
import shutil
import time

import numpy as np
import torch
from transformers.feature_extraction_utils import BatchFeature

from v1_groot_memoryvla.data.dataset.history_dataset import HistoryDataset
from v1_groot_memoryvla.experiment.first_training import cache_sequence, seeded, slice_features
from v1_groot_memoryvla.experiment.task21_diagnostic import load_model


ROOT = Path(__file__).resolve().parents[1]
GROUPS = {'base': (0, 3), 'torso': (3, 7), 'left_arm': (7, 14),
          'left_gripper': (14, 15), 'right_arm': (15, 22), 'right_gripper': (22, 23)}
MODULES = {'encoder': 'memory.perceptual_encoder',
           'per_retrieval0': 'memory.perceptual.retrieval_blocks.0',
           'per_retrieval1': 'memory.perceptual.retrieval_blocks.1',
           'cog_retrieval0': 'memory.cognitive.retrieval_blocks.0',
           'cog_retrieval1': 'memory.cognitive.retrieval_blocks.1',
           'per_time': 'memory.perceptual.timestep_encoder',
           'cog_time': 'memory.cognitive.timestep_encoder',
           'per_gate': 'memory.perceptual.gate_fusion_blocks',
           'cog_gate': 'memory.cognitive.gate_fusion_blocks',
           'projection': 'memory.perceptual_projection',
           'dit': 'action_head.model', 'vlln': 'action_head.vlln',
           'state_encoder': 'action_head.state_encoder',
           'action_encoder': 'action_head.action_encoder',
           'action_decoder': 'action_head.action_decoder',
           'position': 'action_head.position_embedding'}


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + '\n')


def emit(path, value):
    with path.open('a') as file:
        file.write(json.dumps(value, ensure_ascii=False) + '\n')


def sha(path):
    h = hashlib.sha256()
    with path.open('rb') as file:
        for block in iter(lambda: file.read(8 * 1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def backup(args):
    destination = args.output / 'backup'
    if (destination / 'manifest.json').exists():
        return
    destination.mkdir(parents=True, exist_ok=True)
    for name in ('src/v1_groot_memoryvla', 'configs', 'train'):
        shutil.copytree(ROOT / name, destination / name,
                        ignore=shutil.ignore_patterns('.venv', '__pycache__', '*.pyc'), dirs_exist_ok=True)
    shutil.copy2(Path(__file__), destination / Path(__file__).name)
    shutil.copytree(args.checkpoint, destination / 'official_checkpoint', dirs_exist_ok=True)
    files = {str(p.relative_to(destination)): dict(bytes=p.stat().st_size, sha256=sha(p))
             for p in sorted(destination.rglob('*')) if p.is_file()}
    write(destination / 'manifest.json', dict(checkpoint=str(args.checkpoint), files=files,
          note='Complete byte copies of the selected starting weights, active V1 source, train/ and configs/. Original runs untouched.'))
    print('BACKUP_COMPLETE', len(files), flush=True)


def model_load(args, memory=True):
    return load_model(str(args.checkpoint), memory, args.seed,
                      memory_conditioning='direct', memory_length=64,
                      memory_cognitive_qk_norm='layernorm')


def build_cache(args):
    destination = args.output / 'cache'
    destination.mkdir(exist_ok=True)
    from v1_groot_memoryvla.model.gr00t_n1d6.processing_gr00t_n1d6 import Gr00tN1d6Processor
    processor = Gr00tN1d6Processor.from_pretrained(str(args.checkpoint),
        transformers_loading_kwargs={'trust_remote_code': True, 'local_files_only': True})
    processor.eval()
    manifest = []
    for task in (0, 34, 40):
        dataset = ROOT / f'assets/datasets/b1k2025_groot_task{task}_v1_200'
        info = json.loads((dataset / 'meta/info.json').read_text())
        splits = info['splits']
        for split, key in [('train', 'train_episode_ids'), ('validation', 'validation_episode_ids')]:
            # Pre-existing Episode split. Selection never uses model errors.
            for eid in sorted(splits[key])[:2]:
                manifest.append(dict(task=task, episode=eid, split=split,
                                     path=str(destination / f'episode{eid}.pt'), dataset=str(dataset)))
    write(args.output / 'data_manifest.json', manifest)
    missing = [r for r in manifest if not Path(r['path']).exists()]
    if not missing:
        return manifest
    model = model_load(args)
    model.backbone.cuda().eval()
    for task in (0, 34, 40):
        path = ROOT / f'assets/datasets/b1k2025_groot_task{task}_v1_200'
        dataset = HistoryDataset([path], processor, group_size=32, history_stride=8)
        for row in [r for r in missing if r['task'] == task]:
            windows = []
            selected = [(i, g) for i, g in enumerate(dataset.groups) if g[1] == row['episode']]
            for wi, (index, group) in enumerate(selected):
                seq = cache_sequence(model, processor, dataset, index, microbatch=2)
                seq.update(task=task, episode=row['episode'], split=row['split'],
                           episode_start=wi == 0, episode_end=wi == len(selected) - 1)
                windows.append(seq)
            # Pad variable text lengths before joining episode tensors.
            max_length = max(w['features']['backbone_features'].shape[1] for w in windows)
            for window in windows:
                current = window['features']['backbone_features'].shape[1]
                padding = max_length - current
                if padding:
                    for name in ('backbone_features', 'backbone_attention_mask', 'image_mask'):
                        value = window['features'][name]
                        pad = (0, 0, 0, padding) if value.ndim == 3 else (0, padding)
                        window['features'][name] = torch.nn.functional.pad(value, pad)
                    window['input_ids'] = torch.nn.functional.pad(window['input_ids'], (0, padding))
            combined = dict(row, episode_key=windows[0]['episode_key'], windows=windows,
                            features={k: torch.cat([w['features'][k] for w in windows]) for k in windows[0]['features']},
                            input_ids=torch.cat([w['input_ids'] for w in windows]),
                            actions={k: torch.cat([w['actions'][k] for w in windows]) for k in windows[0]['actions']},
                            timesteps=sum([w['timesteps'] for w in windows], []),
                            frame_indices=sum([w['frame_indices'] for w in windows], []))
            torch.save(combined, row['path'])
            print('CACHE_EPISODE', task, row['episode'], len(combined['timesteps']), flush=True)
    del model
    gc.collect()
    torch.cuda.empty_cache()
    return manifest


def parameters(model):
    return {n: p for n, p in model.named_parameters() if n.startswith(('memory.', 'action_head.'))}


def module_for(name):
    for label, prefix in MODULES.items():
        if name == prefix or name.startswith(prefix + '.'):
            return label
    raise ValueError(name)


def param_audit(model, initial, optimizer, gradients=False):
    groups = defaultdict(list)
    details = {}
    for name, p in parameters(model).items():
        label = module_for(name)
        if gradients:
            grad = p.grad
            details[name] = dict(requires_grad=p.requires_grad, grad_none=grad is None,
                                 grad_norm=float(grad.detach().float().norm()) if grad is not None else None)
        else:
            now = p.detach().cpu().float()
            before = initial[name]
            numerator = float((now - before).double().square().sum())
            denominator = float(before.double().square().sum())
            state = optimizer.state.get(p, {})
            details[name] = dict(requires_grad=p.requires_grad, numel=p.numel(),
                changed_scalars=int((now != before).sum()), delta_sq=numerator, initial_sq=denominator,
                relative_delta=math.sqrt(numerator / max(denominator, 1e-30)),
                optimizer_step=int(state['step']) if 'step' in state else 0,
                exp_avg_norm=float(state['exp_avg'].float().norm()) if 'exp_avg' in state else None)
        groups[label].append(details[name])
    result = {}
    for label, entries in groups.items():
        if gradients:
            result[label] = dict(tensors=len(entries), trainable=sum(e['requires_grad'] for e in entries),
                grad_none=sum(e['grad_none'] for e in entries),
                grad_norm=math.sqrt(sum((e['grad_norm'] or 0) ** 2 for e in entries)))
        else:
            result[label] = dict(tensors=len(entries), numel=sum(e['numel'] for e in entries),
                trainable=sum(e['requires_grad'] for e in entries),
                relative_delta=math.sqrt(sum(e['delta_sq'] for e in entries) / max(sum(e['initial_sq'] for e in entries), 1e-30)),
                changed_scalars=sum(e['changed_scalars'] for e in entries),
                optimizer_steps=sorted({e['optimizer_step'] for e in entries}))
    return dict(modules=result, parameters=details)


def gradient_modules(model):
    # One GPU-to-CPU transfer for the grouped tensor norms, before clipping.
    names, norms = [], []
    for name, p in parameters(model).items():
        if p.grad is not None:
            names.append(name)
            norms.append(p.grad.detach().float().norm())
    values = torch.stack(norms).cpu().tolist()
    by_module = defaultdict(float)
    for name, value in zip(names, values):
        by_module[module_for(name)] += value * value
    return {k: math.sqrt(v) for k, v in by_module.items()}


def train_window(model, seq, seed, microbatch):
    count = len(seq['timesteps'])
    units = float(seq['actions']['action_mask'].float().sum())
    total, part_sums, part_counts = 0., defaultdict(float), defaultdict(float)
    for start in range(0, count, microbatch):
        end = min(start + microbatch, count)
        feat = slice_features(seq['features'], start, end)
        actions = slice_features(seq['actions'], start, end)
        context = dict(mode='train_sequence', sequence_start=seq['episode_start'] and start == 0,
                       sequence_end=seq['episode_end'] and end == count,
                       episode_keys=[seq['episode_key']] * (end - start), timesteps=seq['timesteps'][start:end])
        with torch.autocast('cuda', dtype=torch.bfloat16):
            fused = model.memory(feat, seq['input_ids'][start:end].cuda(), context)
            seeded(seed + start)
            output = model.action_head(fused, actions)
            weight = float(actions.action_mask.float().sum()) / units
            loss = output['loss']
        assert torch.isfinite(loss)
        total += float(loss.detach()) * weight
        for name, (lo, hi) in GROUPS.items():
            part_sums[name] += float(output['action_loss'][:, :, lo:hi].detach().sum())
            part_counts[name] += float(output['action_mask'][:, :, lo:hi].float().sum())
        (loss * weight).backward()
    parts = {k: part_sums[k] / part_counts[k] for k in GROUPS}
    assert math.isclose(total, sum(part_sums.values()) / sum(part_counts.values()), rel_tol=1e-5, abs_tol=1e-7)
    return total, parts


def panel(cache):
    result = []
    for episode in cache:
        if episode['split'] != 'validation':
            continue
        n = len(episode['timesteps'])
        target = episode['actions']['action'][:, :32, :23].float()
        # Uniform phases plus highest actual arm-motion target. No prediction-dependent selection.
        arm_intensity = target[:, :, list(range(7, 14)) + list(range(15, 22))].abs().mean((1, 2))
        arm_intensity[:min(8, n)] = -1
        selected = {min(n - 1, max(8, int(n * f))) for f in (.25, .5, .75, .9)}
        selected.add(int(arm_intensity.argmax()))
        for i in sorted(selected):
            result.append((episode['episode'], i))
    return result


def ambiguous_candidates(cache):
    """Operationally ambiguous current observations; these are candidates, not proof of necessity."""
    result = []
    for episode in cache:
        if episode['split'] != 'validation':
            continue
        features = episode['features']['visual_features'].float().mean((1, 2))
        features = torch.nn.functional.normalize(features, dim=-1)
        states = episode['actions']['state'].float().flatten(1)
        actions = episode['actions']['action'][:, :8, :23].float().mean(1)
        n = len(states)
        cosine = features @ features.T
        state_rms = torch.cdist(states, states) / math.sqrt(states.shape[1])
        action_rms = torch.cdist(actions, actions) / math.sqrt(23)
        index = torch.arange(n)
        eligible = ((index[:, None] - index[None]).abs() >= 32) & (index[:, None] >= 8) & (index[None] >= 8)
        eligible &= (cosine >= .98) & (state_rms <= .05) & (action_rms >= .02)
        score = (1 - cosine) + state_rms
        score[~eligible] = torch.inf
        pairs = []
        used = set()
        for flat in score.flatten().argsort()[:100].tolist():
            i, j = divmod(flat, n)
            if not torch.isfinite(score[i, j]) or i in used or j in used:
                continue
            pairs.append(dict(episode=episode['episode'], task=episode['task'], i=i, j=j,
                              visual_cosine=float(cosine[i, j]), state_rms=float(state_rms[i, j]),
                              action_rms=float(action_rms[i, j])))
            used.update((i, j))
            if len(pairs) == 2:
                break
        result.extend(pairs)
    return result


def bank_snapshot(adapter, clone=False):
    return [(dict((key, [(t, f.detach().clone() if clone else f) for t, f in value]) for key, value in bank.bank.items()),
             bank.last_written.copy(), bank.merge_count.copy()) for bank in (adapter.perceptual, adapter.cognitive)]


def bank_restore(adapter, saved):
    for bank, (history, last, merges) in zip((adapter.perceptual, adapter.cognitive), saved):
        bank.bank = {key: list(value) for key, value in history.items()}
        bank.last_written, bank.merge_count = last.copy(), merges.copy()


@torch.no_grad()
def evaluate(model, cache, targets, args, label, step, plain=False, donors=None):
    model.eval()
    saved_runtime = bank_snapshot(model.memory) if model.memory is not None else None
    rows, arrays, mechanisms = [], {}, []
    from v1_groot_memoryvla.model.memory.bank import GateFusion
    active = {'on': False}
    handles = []
    if model.memory is not None:
        for bank_name in ('perceptual', 'cognitive'):
            bank = getattr(model.memory, bank_name)
            def gate_hook(module, inputs, bank_name=bank_name):
                if active['on']:
                    current, retrieved = inputs
                    value = torch.sigmoid(module.proj(torch.cat((current, retrieved), -1))).float()
                    mechanisms.append(dict(kind='gate', bank=bank_name, episode=active['episode'], index=active['index'],
                        current_coefficient_mean=float(value.mean()), saturated_fraction=float(((value < .01) | (value > .99)).float().mean()),
                        current_rms=float(current.float().square().mean().sqrt()), retrieved_rms=float(retrieved.float().square().mean().sqrt())))
            handles.append(bank.gate_fusion_blocks.register_forward_pre_hook(gate_hook))
            for li, block in enumerate(bank.retrieval_blocks):
                def attn_hook(module, inputs, bank_name=bank_name, li=li):
                    if active['on']:
                        query, key, _ = inputs
                        if module.qk_norm == 'layernorm':
                            query = torch.nn.functional.layer_norm(query, (query.shape[-1],))
                            key = torch.nn.functional.layer_norm(key, (key.shape[-1],))
                        q, k = module.q_proj(query).float(), module.k_proj(key).float()
                        probabilities = ((q @ k.transpose(-1, -2)) / math.sqrt(q.shape[-1])).softmax(-1)
                        entropy = -(probabilities * probabilities.clamp_min(1e-30).log()).sum(-1).mean()
                        mechanisms.append(dict(kind='attention', bank=bank_name, layer=li,
                            episode=active['episode'], index=active['index'], keys=k.shape[1],
                            entropy=float(entropy), maximum_probability=float(probabilities.max(-1).values.mean()),
                            effective_keys=float(entropy.exp())))
                handles.append(block.register_forward_pre_hook(attn_hook))
    try:
        for episode in cache:
            indices = sorted(i for eid, i in targets if eid == episode['episode'])
            if not indices:
                continue
            if model.memory is not None:
                model.memory.reset()
            for i in range(max(indices) + 1):
                feat = slice_features(episode['features'], i, i + 1)
                ids = episode['input_ids'][i:i + 1].cuda()
                key, t = episode['episode_key'], episode['timesteps'][i]
                context = dict(mode='stream', episode_keys=[key], timesteps=[t])
                active.update(on=i in indices, episode=episode['episode'], index=i)
                with torch.autocast('cuda', dtype=torch.bfloat16):
                    before = bank_snapshot(model.memory) if model.memory is not None and i in indices else None
                    normal = model.memory(feat, ids, context) if model.memory is not None else feat
                active['on'] = False
                if i not in indices:
                    continue
                after = bank_snapshot(model.memory) if model.memory is not None else None
                conditions = {'correct': normal}
                if model.memory is not None:
                    alternatives = ('wrong_episode', 'wrong_task') if donors is not None else ('shuffled', 'empty')
                    for condition in alternatives:
                        bank_restore(model.memory, before)
                        if condition == 'shuffled':
                            # Keep exact timestamps/capacity, permute only stored feature contents.
                            for bank in (model.memory.perceptual, model.memory.cognitive):
                                history = bank.bank.get(key, [])
                                if len(history) > 1:
                                    permutation = np.random.default_rng(args.seed + episode['episode'] + i).permutation(len(history))
                                    if np.all(permutation == np.arange(len(history))):
                                        permutation = np.roll(permutation, 1)
                                    bank.bank[key] = [(time, history[int(j)][1]) for (time, _), j in zip(history, permutation)]
                        if condition in ('wrong_episode', 'wrong_task'):
                            donor = donors[(episode['episode'], i, condition)]
                            for bank, source in zip((model.memory.perceptual, model.memory.cognitive), donor['snapshot']):
                                history = bank.bank[key]
                                donor_history = source[0][donor['key']]
                                positions = np.linspace(0, len(donor_history) - 1, len(history)).round().astype(int)
                                bank.bank[key] = [(time, donor_history[j][1]) for (time, _), j in zip(history, positions)]
                        with torch.autocast('cuda', dtype=torch.bfloat16):
                            conditions[condition] = model.memory(feat, ids, dict(context, mode='empty') if condition == 'empty' else context)
                    bank_restore(model.memory, after)
                action = slice_features(episode['actions'], i, i + 1)
                target, mask = action.action[:, :32, :23].float(), action.action_mask[:, :32, :23].float()
                predictions = {}
                for condition, fused in conditions.items():
                    seed = 700000 + episode['episode'] * 11 + i
                    with torch.autocast('cuda', dtype=torch.bfloat16):
                        seeded(seed)
                        loss_output = model.action_head(fused, action)
                        seeded(seed + 1)
                        pred = model.action_head.get_action(fused, action)['action_pred'][:, :32, :23].float()
                    predictions[condition] = pred
                    error = pred - target
                    parts = {}
                    for name, (lo, hi) in GROUPS.items():
                        m = mask[:, :, lo:hi]
                        parts[name] = dict(flow_mse=float(loss_output['action_loss'][:, :32, lo:hi].sum() / m.sum()),
                            mae32=float((error[:, :, lo:hi].abs() * m).sum() / m.sum()),
                            mae8=float((error[:, :8, lo:hi].abs() * m[:, :8]).sum() / m[:, :8].sum()))
                    rows.append(dict(arm=label, step=step, episode=episode['episode'], task=episode['task'],
                        index=i, frame=episode['frame_indices'][i], condition=condition,
                        flow_loss=float(loss_output['loss']), mae32=float((error.abs() * mask).sum() / mask.sum()),
                        mae8=float((error[:, :8].abs() * mask[:, :8]).sum() / mask[:, :8].sum()), parts=parts))
                    arrays[f"{episode['episode']}_{i}_{condition}"] = pred.cpu().numpy()
                if len(predictions) > 1:
                    for condition in alternatives:
                        difference = predictions['correct'] - predictions[condition]
                        row = next(r for r in reversed(rows) if r['episode'] == episode['episode'] and r['index'] == i and r['condition'] == condition)
                        row['correct_output_shift_mae8'] = float(difference[:, :8].abs().mean())
                        row['correct_output_shift_mae32'] = float(difference.abs().mean())
                        row['correct_output_shift_rms32'] = float(difference.square().mean().sqrt())
                        row['correct_feature_shift_rms'] = float((normal.backbone_features.float() - conditions[condition].backbone_features.float()).square().mean().sqrt())
    finally:
        for handle in handles:
            handle.remove()
        if saved_runtime is not None:
            bank_restore(model.memory, saved_runtime)
    for row in rows:
        emit(args.output / f'{label}/evaluation.jsonl', row)
    np.savez_compressed(args.output / f'{label}/predictions_step{step:03d}.npz', **arrays)
    write(args.output / f'{label}/mechanisms_step{step:03d}.json', mechanisms)
    print('EVALUATED', label, step, len(rows), flush=True)
    return rows


@torch.no_grad()
def extra_checks(args):
    """No optimization: distinguish history content from temporal permutation."""
    manifest = json.loads((args.output / 'data_manifest.json').read_text())
    cache = [torch.load(r['path'], map_location='cpu', weights_only=False)
             for r in manifest if r['split'] == 'validation']
    for episode in cache:
        episode['actions']['action_mask'] = episode['actions']['action_mask'].float()
    targets = [(r['episode'], r['decision']) for r in json.loads((args.output / 'evaluation_panel.json').read_text())]
    for arm in ('A_joint', 'B_frozen'):
        label = arm + '_history_content'
        directory = args.output / label
        if (directory / 'complete.json').exists():
            continue
        directory.mkdir(exist_ok=True)
        model = model_load(args)
        state = torch.load(args.output / arm / 'final_training_state.pt', map_location='cpu', weights_only=False, mmap=True)
        model.memory.load_state_dict(state['memory'], strict=True)
        model.action_head.load_state_dict(state['action_head'], strict=True)
        del state
        snapshots, provenance = {}, []
        for episode in cache:
            model.memory.reset()
            selected = {i for eid, i in targets if eid == episode['episode']}
            for i in range(max(selected) + 1):
                if i in selected:
                    snapshots[(episode['episode'], i)] = bank_snapshot(model.memory)
                with torch.autocast('cuda', dtype=torch.bfloat16):
                    model.memory(slice_features(episode['features'], i, i + 1), episode['input_ids'][i:i + 1].cuda(),
                        dict(mode='stream', episode_keys=[episode['episode_key']], timesteps=[episode['timesteps'][i]]))
        donors = {}
        for episode in cache:
            for eid, i in targets:
                if eid != episode['episode']:
                    continue
                for condition in ('wrong_episode', 'wrong_task'):
                    donor_episode = next(e for e in cache if e['episode'] != eid and
                        (e['task'] == episode['task'] if condition == 'wrong_episode' else e['task'] == {0: 34, 34: 40, 40: 0}[episode['task']]))
                    choices = [j for donor_eid, j in targets if donor_eid == donor_episode['episode']]
                    j = min(choices, key=lambda j: abs(j / len(donor_episode['timesteps']) - i / len(episode['timesteps'])))
                    donors[(eid, i, condition)] = dict(snapshot=snapshots[(donor_episode['episode'], j)], key=donor_episode['episode_key'])
                    provenance.append(dict(episode=eid, index=i, condition=condition, donor_episode=donor_episode['episode'], donor_index=j))
        write(directory / 'donors.json', provenance)
        model.memory.reset()
        rows = evaluate(model, cache, targets, args, label, args.steps, donors=donors)
        # Correct-history replay must reproduce the already saved experiment.
        original = np.load(args.output / arm / f'predictions_step{args.steps:03d}.npz')
        replay = np.load(directory / f'predictions_step{args.steps:03d}.npz')
        differences = [float(np.abs(original[f'{e}_{i}_correct'] - replay[f'{e}_{i}_correct']).max()) for e, i in targets]
        assert max(differences) == 0, differences
        write(directory / 'complete.json', dict(conditions=['correct', 'wrong_episode', 'wrong_task'],
            correct_replay_max_abs_difference=max(differences), samples=len(targets), optimizer_steps_added=0))
        del model, donors, snapshots
        gc.collect()
        torch.cuda.empty_cache()
    # Existing residual option: exact no-history compatibility, no training.
    if not (args.output / 'residual_empty_compatibility.json').exists():
        model = load_model(str(args.checkpoint), True, args.seed, memory_conditioning='residual',
                           memory_length=64, memory_cognitive_qk_norm='layernorm')
        original = np.load(args.output / 'official_plain/predictions_step000.npz')
        rows = []
        for episode in cache:
            i = min(j for eid, j in targets if eid == episode['episode'])
            feat = slice_features(episode['features'], i, i + 1)
            actions = slice_features(episode['actions'], i, i + 1)
            with torch.autocast('cuda', dtype=torch.bfloat16):
                fused = model.memory(feat, episode['input_ids'][i:i + 1].cuda(),
                    dict(mode='empty', episode_keys=[episode['episode_key']], timesteps=[episode['timesteps'][i]]))
                features_equal = all(torch.equal(feat[k], fused[k]) for k in fused)
                seeded(700000 + episode['episode'] * 11 + i + 1)
                pred = model.action_head.get_action(fused, actions)['action_pred'][:, :32, :23].float().cpu().numpy()
            difference = float(np.abs(pred - original[f'{episode["episode"]}_{i}_correct']).max())
            assert features_equal and difference == 0
            rows.append(dict(episode=episode['episode'], index=i, features_bitwise_equal=features_equal, prediction_max_abs_difference=difference))
        write(args.output / 'residual_empty_compatibility.json', dict(conditioning='existing residual option',
            history='empty', optimized_steps=0, observations=rows,
            limit='Verifies original specialist compatibility only; does not establish trained history benefit.'))
        print('RESIDUAL_EMPTY_COMPATIBLE', len(rows), flush=True)


def make_order(cache, steps, seed):
    # One virtual Episode slot per task, consuming all its windows in order.
    queues = {task: [w for e in cache if e['split'] == 'train' and e['task'] == task for w in e['windows']]
              for task in (0, 34, 40)}
    counters = defaultdict(int)
    result = []
    for step in range(steps):
        task = (0, 34, 40)[step % 3]
        seq = queues[task][counters[task] % len(queues[task])]
        result.append(seq)
        counters[task] += 1
    return result


def train(args, cache, targets, arm):
    directory = args.output / arm
    directory.mkdir(exist_ok=True)
    if (directory / 'complete.json').exists():
        return
    model = model_load(args)
    if arm == 'B_frozen':
        model.action_head.requires_grad_(False)
        # Keep training/eval dropout schedule equal to A; only gradients differ.
    initial = {n: p.detach().cpu().clone() for n, p in parameters(model).items()}
    initial_hash = hashlib.sha256()
    for name, value in initial.items():
        initial_hash.update(name.encode())
        initial_hash.update(value.numpy().tobytes())
    digest = initial_hash.hexdigest()
    if not (args.output / 'backup/initialized_memory.pt').exists():
        torch.save({n: p for n, p in initial.items() if n.startswith('memory.')}, args.output / 'backup/initialized_memory.pt')
        model.config.save_pretrained(args.output / 'backup/initialized_config')
    entries = parameters(model)
    memory_params = [p for n, p in entries.items() if n.startswith('memory.') and p.requires_grad]
    head_params = [p for n, p in entries.items() if n.startswith('action_head.') and p.requires_grad]
    groups = [dict(params=memory_params, lr=args.learning_rate, label='memory')]
    if head_params:
        groups.append(dict(params=head_params, lr=args.learning_rate, label='original_action_head'))
    optimizer = torch.optim.AdamW(groups, lr=args.learning_rate, weight_decay=.01, foreach=False)
    config = dict(arm=arm, initialization_sha256=digest, lr=args.learning_rate,
        optimizer_groups=[dict(label=g['label'], lr=g['lr'], tensors=len(g['params']), numel=sum(p.numel() for p in g['params'])) for g in optimizer.param_groups],
        requires_grad={n: p.requires_grad for n, p in model.named_parameters()},
        registered_memory_tensors=sum(n.startswith('memory.') for n in model.state_dict()),
        trainable_memory_tensors=len(memory_params), seed=args.seed, steps=args.steps,
        precision='FP32 optimizer master parameters; BF16 CUDA autocast',
        memory_conditioning='direct', memory_length=64, cognitive_qk_norm='layernorm',
        loss='original masked flow-matching MSE, valid action elements only',
        image_augmentation=False, augmentation_note='Frozen official Eagle eval cache shared exactly by A/B; bounded diagnostic, not production retraining',
        backbone_frozen=not any(p.requires_grad for p in model.backbone.parameters()),
        dropout_note='Both arms use identical action-head train mode and RNG; B only changes requires_grad on the original head',
        checkpoint=str(args.checkpoint))
    write(directory / 'config.json', config)
    other = args.output / ('A_joint' if arm == 'B_frozen' else 'B_frozen') / 'config.json'
    if other.exists():
        assert json.loads(other.read_text())['initialization_sha256'] == digest
    if arm == 'A_joint':
        evaluate(model, cache, targets, args, arm, 0)
    else:
        # Exact same weights, cached features, and eval mode imply same step-0 outputs.
        shutil.copy2(args.output / 'A_joint/predictions_step000.npz', directory / 'predictions_step000.npz')
        rows = [json.loads(line) for line in (args.output / 'A_joint/evaluation.jsonl').read_text().splitlines() if json.loads(line)['step'] == 0]
        for row in rows:
            emit(directory / 'evaluation.jsonl', dict(row, arm=arm))
        shutil.copy2(args.output / 'A_joint/mechanisms_step000.json', directory / 'mechanisms_step000.json')
    order = make_order(cache, args.steps, args.seed)
    write(directory / 'order.json', [dict(episode=s['episode'], task=s['task'], group_id=s['group_id'], frames=s['frame_indices']) for s in order])
    for step, seq in enumerate(order, 1):
        tick = time.monotonic()
        model.train()
        model.backbone.eval()
        optimizer.zero_grad(set_to_none=True)
        loss, parts = train_window(model, seq, args.seed + 1000 * step, args.microbatch)
        gradients = gradient_modules(model)
        if step in (1, 2, 10, 50, args.steps):
            write(directory / f'gradients_step{step:03d}.json', param_audit(model, initial, optimizer, gradients=True))
        params = [p for group in optimizer.param_groups for p in group['params']]
        norm = torch.nn.utils.clip_grad_norm_(params, 1.)
        assert torch.isfinite(norm)
        optimizer.step()
        torch.cuda.synchronize()
        record = dict(step=step, episode=seq['episode'], task=seq['task'], frames=seq['frame_indices'],
                      loss=loss, part_flow_mse=parts, gradients=gradients,
                      global_grad_norm_before_clip=float(norm), seconds=time.monotonic() - tick,
                      peak_allocated_gib=torch.cuda.max_memory_allocated() / 2**30)
        emit(directory / 'training.jsonl', record)
        if step in (1, 10, 50, args.steps):
            write(directory / f'updates_step{step:03d}.json', param_audit(model, initial, optimizer))
        if step == 1 or step % 10 == 0:
            print('TRAIN', arm, step, f'loss={loss:.7f}', f'seconds={record["seconds"]:.2f}', flush=True)
    optimizer.zero_grad(set_to_none=True)
    evaluate(model, cache, targets, args, arm, args.steps)
    model.memory.reset()
    torch.save(dict(memory=model.memory.state_dict(), action_head=model.action_head.state_dict(),
                    optimizer=optimizer.state_dict(), completed_steps=args.steps,
                    backbone_reference=str(args.output / 'backup/official_checkpoint')),
               directory / 'final_training_state.pt')
    model.config.save_pretrained(directory / 'final_config')
    write(directory / 'complete.json', dict(steps=args.steps, initialization_sha256=digest,
          final_parameters_sha256=sha(directory / 'final_training_state.pt')))
    del initial, optimizer, model, entries, memory_params, head_params, groups, params
    gc.collect()
    torch.cuda.empty_cache()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, default=ROOT / 'outputs/task2_memory_training_20261113')
    parser.add_argument('--checkpoint', type=Path, default=ROOT / 'checkpoints/groot_n1d6_behavior1k')
    parser.add_argument('--steps', type=int, default=100)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--microbatch', type=int, default=2)
    parser.add_argument('--learning-rate', type=float, default=1e-5)
    parser.add_argument('--extra-checks', action='store_true')
    args = parser.parse_args()
    assert 100 <= args.steps <= 300
    args.output.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(4)
    if args.extra_checks:
        extra_checks(args)
        return
    write(args.output / 'request.json', {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()})
    backup(args)
    assert torch.cuda.is_available(), 'CUDA required for real GR00T diagnosis'
    write(args.output / 'environment.json', dict(torch=torch.__version__, gpu=torch.cuda.get_device_name(),
          visible_devices=__import__('os').environ.get('CUDA_VISIBLE_DEVICES'),
          total_memory=torch.cuda.get_device_properties(0).total_memory))
    manifest = build_cache(args)
    cache = [torch.load(row['path'], map_location='cpu', weights_only=False) for row in manifest]
    # A BF16 0/1 mask can round its integer sum (e.g. 23000 -> 23040).
    # Keep diagnostic reductions exact without changing any model implementation.
    for episode in cache:
        episode['actions']['action_mask'] = episode['actions']['action_mask'].float()
        for window in episode['windows']:
            window['actions']['action_mask'] = window['actions']['action_mask'].float()
    ambiguous = ambiguous_candidates(cache)
    targets = sorted(set(panel(cache) + [(p['episode'], i) for p in ambiguous for i in (p['i'], p['j'])]))
    write(args.output / 'history_dependency_candidates.json', dict(
        definition='Same validation episode, >=32 decisions apart; current visual pooled cosine>=.98, normalized state RMS<=.05, next8 mean-action RMS difference>=.02. Candidates only: unobserved state and genuine necessity not certified.',
        pairs=ambiguous))
    write(args.output / 'evaluation_panel.json', [dict(episode=e, decision=i) for e, i in targets])
    # Original specialist interface establishes whether adding direct conditioning
    # changes behavior before any history or learning is involved.
    plain_dir = args.output / 'official_plain'
    plain_dir.mkdir(exist_ok=True)
    if not (plain_dir / 'evaluation.jsonl').exists():
        model = model_load(args, memory=False)
        evaluate(model, cache, targets, args, 'official_plain', 0, plain=True)
        del model
        gc.collect()
        torch.cuda.empty_cache()
    train(args, cache, targets, 'A_joint')
    train(args, cache, targets, 'B_frozen')
    write(args.output / 'complete.json', dict(status='complete', steps_per_arm=args.steps,
          unique_episodes=len(cache), evaluation_observations=len(targets), ambiguous_pairs=len(ambiguous)))
    print('ALL_COMPLETE', args.output, flush=True)


if __name__ == '__main__':
    main()
