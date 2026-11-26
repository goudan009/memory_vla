"""One locked evaluation: held-out state scores and 100 fixed-input probes.

Executed only by the existing workbench after the final complete checkpoint.
All prefixes are encoded by that checkpoint with evaluation image transforms.
"""
import argparse
from contextlib import ExitStack
from copy import deepcopy
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch

from v1_groot_memoryvla.data.dataset.episode_windows import EpisodeWindows
from v1_groot_memoryvla.experiment.first_training import cache_sequence, seeded, slice_features
from v1_groot_memoryvla.model.gr00t_n1d6.processing_gr00t_n1d6 import Gr00tN1d6Processor
from .model.gr00t_n1d6 import Gr00tN1d6
from .task40_training import write_json
from .training_smoke import file_hash


def binary_scores(truth, predicted):
    truth, predicted = np.asarray(truth, bool), np.asarray(predicted, bool)
    tp, tn = int((truth & predicted).sum()), int((~truth & ~predicted).sum())
    fp, fn = int((~truth & predicted).sum()), int((truth & ~predicted).sum())
    precision, recall = tp / max(tp + fp, 1), tp / max(tp + fn, 1)
    return dict(tp=tp, tn=tn, fp=fp, fn=fn, positive=int(truth.sum()), negative=int((~truth).sum()),
                precision=precision, recall=recall, f1=2*tp/max(2*tp+fp+fn, 1),
                accuracy=(tp+tn)/max(len(truth), 1))


def matched_wrong(snapshot, target, donor):
    result = deepcopy(snapshot)
    bank = result['cognitive']['bank']
    reference, alternative = bank[target], bank[donor]
    if not reference or len(alternative) < len(reference):
        raise ValueError('Donor has insufficient past slots')
    alternative = alternative[-len(reference):]
    rows, scales = [], []
    for (time, value), (_, source) in zip(reference, alternative):
        a, b = value.float().square().mean().sqrt(), source.float().square().mean().sqrt()
        if b == 0 and a > 0:
            raise ValueError('A zero donor cannot match nonzero target scale')
        scale = a / (b if b > 0 else torch.ones_like(b))
        rows.append((time, (source.float()*scale).to(value.dtype)))
        scales.append(float(scale))
    bank[target] = rows
    return result, dict(target_times=[t for t, _ in reference],
        donor_times=[t for t, _ in alternative], scales=scales,
        rms_max_abs=max(float((a.float().square().mean().sqrt()-b.float().square().mean().sqrt()).abs())
                        for (_, a), (_, b) in zip(reference, rows)))


def off_history(snapshot, target, off):
    result = deepcopy(snapshot)
    for mapping in result['cognitive'].values():
        if isinstance(mapping, dict):
            mapping.pop(target, None)
            if off in mapping:
                mapping[target] = deepcopy(mapping[off])
    # The target perceptual dictionary is kept exactly as in normal.
    return result


@torch.inference_mode()
def forward(model, seq, index, key, *, trace=False, seed=0):
    context = dict(mode='stream', episode_keys=[key], timesteps=[seq['timesteps'][index]],
                   history_mode='normal', memory_diagnostics=trace, memory_action_controls=False)
    features = slice_features(seq['features'], index, index+1)
    ids = seq['input_ids'][index:index+1].cuda()
    with torch.autocast('cuda', dtype=torch.bfloat16), ExitStack() as stack:
        recorder = None
        if trace:
            from task40_memory_instrumentation import _Trace
            recorder = _Trace(SimpleNamespace(model=model), 1., 'eos', 'compact')
            recorder.install(stack)
        fused = model.memory(features, ids, context)
        probability = float(fused['state_feature'].float()[0, 0])
        if not trace:
            return probability
        seeded(seed)
        model.action_head.get_action(fused, slice_features(seq['actions'], index, index+1))
    arrays = recorder.arrays
    arrays['state_feature'] = fused['state_feature'].float().cpu().numpy()
    arrays['memory_gate'] = fused['memory_gate'].float().cpu().numpy()
    arrays['retrieved_memory'] = fused['retrieved_memory'].float().cpu().numpy()
    return arrays, deepcopy(model.memory.last_diagnostics)


def reference_actions(actions, index):
    # Cached normalized actions follow the model's BF16 input protocol.
    # NumPy needs an FP32 diagnostic copy; keep the cached model inputs intact.
    return (actions['action'][index].detach().float().cpu().numpy(),
            actions['action_mask'][index].detach().bool().cpu().numpy())


def compare(normal, alternative, reference, mask, parts):
    result = {}
    for name in ('cognitive_readout', 'cognitive_value_after_gate', 'memory_gate',
                 'cognitive_value_native_scaled', 'expert_input_eos', 'dit_condition_eos', 'normalized_actions'):
        if name in normal and name in alternative:
            delta = normal[name].astype(np.float32)-alternative[name].astype(np.float32)
            if name == 'normalized_actions':
                delta = delta[:, :32, :max(end for _, end in parts.values())]
            result[name] = dict(mae=float(np.abs(delta).mean()), max_abs=float(np.abs(delta).max()))
    result['action_parts'] = {}
    for name, (start, end) in parts.items():
        a, b = normal['normalized_actions'][0, :32, start:end], alternative['normalized_actions'][0, :32, start:end]
        truth, valid = reference[:32, start:end], mask[:32, start:end].astype(bool)
        result['action_parts'][name] = dict(mae=float(np.abs(a-b).mean()),
            mae_first8=float(np.abs(a[:8]-b[:8]).mean()),
            normal_reference_mae=float(np.abs(a-truth)[valid].mean()) if valid.any() else None,
            alternative_reference_mae=float(np.abs(b-truth)[valid].mean()) if valid.any() else None)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('checkpoint', 'dataset', 'protocol', 'labels', 'output'):
        parser.add_argument('--'+name, type=Path, required=True)
    args = parser.parse_args()
    if not (args.checkpoint / 'COMPLETE.json').is_file():
        raise ValueError('Evaluate only the final complete checkpoint')
    protocol = json.loads(args.protocol.read_text())
    if args.checkpoint.name != protocol['v2_checkpoint_id']:
        raise ValueError('Checkpoint identity differs from locked evaluation')
    torch.set_num_threads(4)
    seeded(42)
    processor = Gr00tN1d6Processor.from_pretrained(args.checkpoint)
    processor.eval()
    model = Gr00tN1d6.from_pretrained(args.checkpoint, torch_dtype=torch.float32,
        transformers_loading_kwargs={'trust_remote_code': True, 'local_files_only': True})
    model.backbone.bfloat16()
    model.cuda().eval()
    dataset = EpisodeWindows(args.dataset, processor, protocol['state_evaluation_episodes'], group_size=32)
    labels = {r['episode']: r for r in json.loads(args.labels.read_text())['episodes']}
    output = args.output.parent
    cache = output / 'evaluation_features'
    cache.mkdir(exist_ok=True)
    artifacts = output / 'mechanism_windows'
    artifacts.mkdir(exist_ok=True)
    windows = {(r['episode'], r['decision']): r for r in protocol['windows']}
    if len(windows) != 100:
        raise ValueError('Expected exactly 100 preselected windows')
    # The checkpoint processor defines each normalized action component.
    offset, parts = 0, {}
    # R1Pro action order is verified against the processor below at first batch.
    part_names = processor.modality_configs['behavior_r1_pro']['action'].modality_keys
    dimensions = json.loads((args.dataset / 'meta/modality.json').read_text())['action']
    for name in part_names:
        width = dimensions[name]['end']-dimensions[name]['start']
        parts[name] = (offset, offset+width)
        offset += width

    def get_sequence(eid, window):
        indices = dataset.by_episode[eid]
        index = indices[window]
        path = cache / f'ep{eid}_window{window:03d}.pt'
        if not path.exists():
            seq = cache_sequence(model, processor, dataset, index, microbatch=2)
            torch.save(seq, path)
        return torch.load(path, map_location='cpu', weights_only=False)

    truth_all, prediction_all, completed, per_episode = [], [], [], []
    state_records = output / 'state_predictions.jsonl'
    with state_records.open('x') as state_log:
        for eid in protocol['state_evaluation_episodes']:
            first_known = len(truth_all)
            model.memory.reset()
            targets = [r for r in protocol['windows'] if r['episode'] == eid]
            donor, = {r['donor_episode'] for r in targets}
            last_donor = max(r['decision'] for r in targets)
            key, off_key, donor_key = f'eval/{eid}', f'off/{eid}', f'donor/{donor}'
            label = labels[eid]
            if file_hash(label['artifact']) != label['artifact_sha256']:
                raise ValueError('Certified label artifact changed')
            with np.load(label['artifact']) as archive:
                y, known = archive['labels'][:, 0], archive['known'][:, 0]
            for window in range(len(dataset.by_episode[eid])):
                seq = get_sequence(eid, window)
                donor_seq = get_sequence(donor, window) if seq['timesteps'][0] <= last_donor else None
                for index, (decision, frame) in enumerate(zip(seq['timesteps'], seq['frame_indices'])):
                    item = windows.get((eid, decision))
                    if item:
                        before = model.memory.runtime_state_dict()
                        wrong, donor_evidence = matched_wrong(before, key, donor_key)
                        empty = deepcopy(before)
                        empty['cognitive']['bank'][key] = []
                        variants = dict(normal=before, empty_cognitive=empty, wrong_cognitive=wrong,
                                        state_guidance_off=off_history(before, key, off_key))
                        probes, retrievals = {}, {}
                        for condition, runtime in variants.items():
                            model.memory.load_runtime_state_dict(runtime)
                            model.memory.cognitive.guidance_enabled = condition != 'state_guidance_off'
                            probes[condition], retrievals[condition] = forward(model, seq, index, key,
                                trace=True, seed=600000+eid+decision)
                        model.memory.load_runtime_state_dict(before)
                        model.memory.cognitive.guidance_enabled = True
                        reference, mask = reference_actions(seq['actions'], index)
                        row = dict(item, known=bool(known[frame]), truth=float(y[frame]), donor=donor_evidence,
                            comparisons={condition: compare(probes['normal'], arrays, reference, mask, parts)
                                         for condition, arrays in probes.items() if condition != 'normal'},
                            retrieval=retrievals)
                        for condition, arrays in probes.items():
                            np.savez_compressed(artifacts / f"{item['window_id']}_{condition}.npz", **arrays)
                        write_json(artifacts / f"{item['window_id']}.json", row)
                        completed.append(row)
                        print(json.dumps(dict(window_completed=len(completed), episode=eid, decision=decision)), flush=True)
                    probability = forward(model, seq, index, key)
                    event = bool(model.memory.cognitive.last_record['event'])
                    retained_events = sorted(model.memory.cognitive.event_slots.get(key, set()))
                    if known[frame]:
                        truth_all.append(bool(y[frame]))
                        prediction_all.append(probability >= .5)
                    state_log.write(json.dumps(dict(episode=eid, decision=decision, frame=frame,
                        known=bool(known[frame]), truth=float(y[frame]), probability=probability,
                        predicted_event=event, retained_event_slots=retained_events))+'\n')
                    model.memory.cognitive.guidance_enabled = False
                    forward(model, seq, index, off_key)
                    model.memory.cognitive.guidance_enabled = True
                    if decision <= last_donor:
                        if donor_seq['timesteps'][index] != decision:
                            raise ValueError('Donor prefix clock differs')
                        forward(model, donor_seq, index, donor_key)
                state_log.flush()
            per_episode.append(dict(episode=eid, **binary_scores(truth_all[first_known:], prediction_all[first_known:])))
            write_json(output / 'evaluation_status.json', dict(episodes_completed=len({r['episode'] for r in completed}),
                windows_completed=len(completed), state_known=len(truth_all)))
    result = dict(passed=len(completed) == 100, checkpoint=str(args.checkpoint),
        checkpoint_complete_sha256=file_hash(args.checkpoint / 'COMPLETE.json'),
        protocol_sha256=file_hash(args.protocol), windows_completed=len(completed),
        episodes_completed=len(protocol['state_evaluation_episodes']),
        state_scores=binary_scores(truth_all, prediction_all),
        macro_f1=binary_scores(truth_all, prediction_all)['f1'],
        macro_f1_scope='one enabled label: started', per_episode_state_scores=per_episode,
        constant_zero=binary_scores(truth_all, np.zeros(len(truth_all), bool)),
        constant_one=binary_scores(truth_all, np.ones(len(truth_all), bool)),
        holdout_scope='V2 updates only', metrics_unit='stride-eight decisions; correlated within independent episodes',
        action_units='checkpoint normalized action coordinates; masked reference error is diagnostic',
        mean_effects={condition: {feature: float(np.mean([r['comparisons'][condition][feature]['mae'] for r in completed]))
            for feature in completed[0]['comparisons'][condition] if feature != 'action_parts'}
            for condition in completed[0]['comparisons']})
    write_json(args.output, result)
    print(json.dumps(result), flush=True)


if __name__ == '__main__':
    main()
