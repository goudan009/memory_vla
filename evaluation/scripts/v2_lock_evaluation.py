"""Freeze the single V2 evaluation panel before observing trained results (CPU)."""
import argparse
import hashlib
import json
from pathlib import Path
import time


def digest(path):
    value = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b''):
            value.update(chunk)
    return value.hexdigest()


def build(root):
    experiment = root / 'Output/test_02/experiment_01'
    split_path = experiment / 'evidence/data_split.json'
    split = json.loads(split_path.read_text())
    episodes = {r['episode_index']: r['length'] for r in map(json.loads,
        (root / 'assets/datasets/b1k2025_groot_task40_v1_200/meta/episodes.jsonl').read_text().splitlines())}
    panel_path = root / 'Output/test_01/experiment_11/restore_panel.json'
    pairs = []
    for index, item in enumerate(json.loads(panel_path.read_text())[:12]):
        checkpoint, prefix = root / item['checkpoint'], Path(item['prefix'])
        files = [checkpoint / n for n in ('sim.pt', 'physical.json', 'source.json',
                                          'observation.npz', 'policy_rng_online.pt')]
        for decision in range(int(item['decision']) + 1):
            files.extend(prefix / f'{decision:04d}' / n for n in ('observation.npz', 'decision.json'))
        pairs.append(dict(item, pair_id=f'pair{index+1:03d}',
                          input_sha256={str(p.relative_to(root)): digest(p) for p in files}))
    validation = sorted(split['validation'])
    windows = []
    for eid in validation:
        donor = max((e for e in validation if e != eid), key=lambda e: episodes[e])
        maximum = min((episodes[eid] - 1) // 8, (episodes[donor] - 1) // 8)
        for number, fraction in enumerate((.20, .40, .60, .80, .98)):
            decision = max(2, int(maximum * fraction))
            windows.append(dict(window_id=f'ep{eid}_w{number+1}', episode=eid,
                decision=decision, frame=decision*8, donor_episode=donor,
                prefix_first_decision=0, reference_action_horizon=32,
                model_action_horizon=50, executed_horizon=8))
    return dict(schema_version=1, locked_at=time.strftime('%Y-%m-%d %H:%M:%S %z'),
        model_version='v2', test_id='test02', experiment_id='exp01', run_id='run001',
        v1_checkpoint='checkpoints/v1__test01__exp11__step006000',
        v2_checkpoint_id='v2__test02__exp01__run001__step001000',
        selection='first 12 archived panel entries; five fixed temporal fractions per V2-held-out episode; no performance selection',
        split_sha256=digest(split_path), source_panel_sha256=digest(panel_path),
        pairs=pairs, windows=windows, state_evaluation_episodes=validation,
        closed_loop=dict(cycles=60, n_action_steps=8, restore_repeats=1,
            rebuild_policy_history=True, stop_at_first_open=False,
            post_event_action_chunks=2, validity='original physical/controller checks and approximate restored RGB (camera MAE < 25/255); invalids separate',
            scope='door-stage recovery; continue after opening within 60 decisions; premature toggle terminates after two evidence chunks'),
        diagnostics=dict(conditions=['normal','empty_cognitive','wrong_cognitive','state_guidance_off'],
            state_threshold=.5, state_truth='certified started only; unknown excluded',
            classification_scope='all stride-eight decisions in all 20 V2-update-heldout episodes',
            donor='different held-out episode, own checkpoint prefix to same clock; target slot times/count and per-slot RMS',
            fixed='current encoded RGB/proprio/language, perceptual bank and action noise',
            state_off='all three guidance paths disabled and cognition rebuilt from decision zero; target perceptual bank fixed',
            precision='BF16 backbone and autocast; FP32 memory/expert parameters as in training',
            limits='held out from V2 updates only; initialization saw these demonstrations; action differences do not establish improvement'))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, default=Path.cwd())
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--previous', type=Path)
    args = parser.parse_args()
    protocol = build(args.root.resolve())
    if args.previous:
        previous = json.loads(args.previous.read_text())
        assert protocol['pairs'] == previous['pairs']
        assert [(r['window_id'],r['episode'],r['decision'],r['donor_episode']) for r in protocol['windows']] == [
            (r['window_id'],r['episode'],r['decision'],r['donor_episode']) for r in previous['windows']]
        protocol['previous_protocol_sha256'] = digest(args.previous)
        protocol['correction'] = 'Metadata only: model horizon is the unchanged checkpoint value 50; 32 is the masked reference horizon. Pair/window identities unchanged; no V2 results observed.'
    with args.output.open('x') as stream:
        json.dump(protocol, stream, ensure_ascii=False, indent=2)
        stream.write('\n')
    print(json.dumps(dict(pairs=len(protocol['pairs']), windows=len(protocol['windows']),
                         sha256=digest(args.output))))


if __name__ == '__main__':
    main()
