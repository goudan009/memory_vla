"""Summarize the locked V2 evaluation's measured windows and paired attempts."""
import argparse
import csv
import json
import math
from collections import defaultdict
from pathlib import Path

import numpy as np

from benchmark_dashboard import atomic_json


def csv_rows(path, rows, fieldnames=None):
    if not rows and fieldnames is None:
        return
    with path.open('w', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=fieldnames or list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def attention_change(normal, alternative):
    a, b = normal['retrieval_attention'], alternative['retrieval_attention']
    a = dict(zip(a['slot_timesteps'], a['slot_mass']))
    b = dict(zip(b['slot_timesteps'], b['slot_mass']))
    top_a, top_b = max(a, key=a.get), max(b, key=b.get)
    return dict(attention_slot_l1=sum(abs(a.get(t, 0)-b.get(t, 0)) for t in a.keys() | b.keys()),
                attention_top1_changed=int(top_a != top_b),
                history_slot_sets_changed=int(a.keys() != b.keys()))


def proportion(successes, total):
    if not total:
        return dict(successes=successes, total=0, rate=None, wilson95=None)
    rate, z = successes / total, 1.959963984540054
    center = (rate + z*z/(2*total)) / (1 + z*z/total)
    width = z * math.sqrt(rate*(1-rate)/total + z*z/(4*total*total)) / (1 + z*z/total)
    return dict(successes=successes, total=total, rate=rate,
                wilson95=[max(0., center-width), min(1., center+width)])


def physical_stages(facts):
    """Observed physical predicates, not inferred policy intent."""
    seen_open = closed_after_open = False
    for row in facts:
        closed_after_open |= seen_open and not row['Open']
        seen_open |= row['Open']
    return dict(
        grasped_bag=any(any(obj in row['popcorn'] for obj in row['grasped'].values()) for row in facts),
        bag_inside=any(row['Inside'] for row in facts),
        closed_after_open=closed_after_open,
        started=any(row['ToggledOn'] for row in facts),
        started_with_bag_inside_and_door_closed=any(
            row['Inside'] and not row['Open'] and row['ToggledOn'] for row in facts))


def paired_behavior(results):
    """Count actual valid episodes; do not score invalid restore attempts as losses."""
    rows = []
    for result in results:
        if result['status'] != 'complete':
            continue
        trial, = result['trials']
        rows.append(dict(pair_id=result['pair_id'], version=result['version'],
            instance=result['instance'], seed=result['seed'], decisions=trial['decisions'],
            steps=trial['steps'], stop_reason=trial['stop_reason'],
            opened=trial['first_open_step'] is not None,
            first_open_step=trial['first_open_step'],
            premature_toggle=trial['first_error'] is not None,
            first_premature_toggle_step=(trial['first_error'] or {}).get('step'),
            task_success=trial.get('first_task_success_step') is not None,
            door_contact_steps=len(trial['door_contact_steps']),
            max_hinge_excursion_rad=trial['max_hinge_excursion_rad']))
    lookup = {(r['pair_id'], r['version']): r for r in rows}
    matched = sorted(p for p in {r['pair_id'] for r in rows}
        if all((p, v) in lookup for v in ('v1', 'v2')))
    summary = dict(matched_pairs=len(matched), matched_pair_ids=matched,
        scope='Approximate restored-start comparison; physical/controller and RGB tolerance audited. Not bit-exact same-image causal evidence.',
        intervals='Wilson 95%, episode-level descriptive intervals; repeated instances and selected recovery starts limit generalization')
    for version in ('v1', 'v2'):
        valid = [r for r in rows if r['version'] == version and r['pair_id'] in matched]
        summary[version] = dict(valid_episodes=len(valid),
            total_decisions=sum(r['decisions'] for r in valid),
            **{event: proportion(sum(r[event] for r in valid), len(valid))
               for event in ('opened', 'premature_toggle', 'task_success')})
    summary['opening_pair_outcomes'] = {
        'both_opened': sum(lookup[p, 'v1']['opened'] and lookup[p, 'v2']['opened'] for p in matched),
        'v2_only': sum(not lookup[p, 'v1']['opened'] and lookup[p, 'v2']['opened'] for p in matched),
        'v1_only': sum(lookup[p, 'v1']['opened'] and not lookup[p, 'v2']['opened'] for p in matched),
        'neither_opened': sum(not lookup[p, 'v1']['opened'] and not lookup[p, 'v2']['opened'] for p in matched)}
    return rows, summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run', type=Path, required=True)
    parser.add_argument('--paired-run', type=Path,
                        help='Completed paired continuation; offline diagnostics stay in --run')
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    offline = json.loads((args.run / 'offline_evaluation.json').read_text())
    windows = [json.loads(p.read_text()) for p in sorted((args.run / 'mechanism_windows').glob('*.json'))]
    if not offline['passed'] or len(windows) != 100:
        raise ValueError('Summarize only the completed locked 100-window evaluation')
    args.output.mkdir(parents=True, exist_ok=True)
    detail, action_rows, by_episode = [], [], defaultdict(list)
    for window in windows:
        normal = window['retrieval']['normal']['observations'][0]['cognitive']
        for condition, comparison in window['comparisons'].items():
            row = dict(window_id=window['window_id'], episode=window['episode'],
                decision=window['decision'], condition=condition,
                history_coefficient=1-normal['gate_current_mean'],
                retained_events=len(normal['event_timesteps']),
                event_slot_fraction=len(normal['event_timesteps'])/max(normal['slots_after_write'], 1))
            for feature, values in comparison.items():
                if feature != 'action_parts':
                    row[feature+'_mae'] = values['mae']
            other = window['retrieval'][condition]['observations'][0]['cognitive']
            if normal.get('retrieval_attention') and other.get('retrieval_attention'):
                row.update(attention_change(normal, other))
            detail.append(row)
            by_episode[(window['episode'], condition)].append(row)
            for part, values in comparison['action_parts'].items():
                action_rows.append(dict(window_id=window['window_id'], episode=window['episode'],
                    condition=condition, part=part, **values))
    # Empty history has no readout/attention; retain missing as missing in CSV.
    columns = list(dict.fromkeys(key for row in detail for key in row))
    csv_rows(args.output / 'mechanism_metrics.csv', [{k: row.get(k) for k in columns} for row in detail])
    csv_rows(args.output / 'action_part_metrics.csv', action_rows)
    episode_rows = []
    for (episode, condition), rows in sorted(by_episode.items()):
        episode_rows.append(dict(episode=episode, condition=condition, windows=len(rows),
            **{key: float(np.mean([row[key] for row in rows if key in row]))
               for key in columns if key not in ('window_id', 'episode', 'decision', 'condition')
               and any(key in row for row in rows)}))
    csv_rows(args.output / 'mechanism_per_episode.csv',
             [{k: row.get(k) for k in ['episode','condition','windows',*columns[4:]]} for row in episode_rows])
    predictions = [json.loads(line) for line in (args.run / 'state_predictions.jsonl').open()]
    probability_by_truth = {}
    for truth in (0, 1):
        values = [row['probability'] for row in predictions if row['known'] and row['truth'] == truth]
        probability_by_truth[str(truth)] = dict(decisions=len(values),
            quantile_levels=[0, .25, .5, .75, 1],
            quantiles=np.quantile(values, [0, .25, .5, .75, 1]).tolist() if values else [])
    summary = dict(job_id=args.run.name, checkpoint=offline['checkpoint'],
        protocol_sha256=offline['protocol_sha256'], state_scores=offline['state_scores'],
        constant_zero=offline['constant_zero'], constant_one=offline['constant_one'],
        per_episode_state_scores=offline['per_episode_state_scores'],
        windows=len(windows), independent_episodes=len({w['episode'] for w in windows}),
        probability_by_truth=probability_by_truth,
        predicted_events=sum(r['predicted_event'] for r in predictions),
        decisions_with_retained_event=sum(bool(r['retained_event_slots']) for r in predictions),
        mean_effects=offline['mean_effects'],
        units='100 correlated windows in 20 V2-update-held-out episodes; no end-to-end unseen claim',
        attention_semantics='Recorded FP32 Q/K softmax recomputation, not bit-exact SDPA kernel weights')
    paired_file = (args.paired_run or args.run) / 'paired_evaluation.json'
    if args.paired_run and not paired_file.is_file():
        raise ValueError('Paired continuation has not produced its final report')
    if paired_file.exists():
        paired = json.loads(paired_file.read_text())
        if args.paired_run:
            plan = json.loads((args.paired_run / 'plan.json').read_text())
            if (plan['config']['protocol_sha256'] != offline['protocol_sha256']
                    or plan['config']['checkpoint'] != offline['checkpoint']):
                raise ValueError('Paired continuation and offline diagnostics must share checkpoint and protocol')
        summary['paired_job_id'] = (args.paired_run or args.run).name
        trials, videos = [], []
        for result in paired['results']:
            trials.append({key: result[key] for key in ('pair_id','version','instance','seed','status','returncode','output')})
            # Simulator appdata contains inaccessible Kit directories; recordings
            # belong to videos/, not that runtime cache.
            for video in sorted((Path(result['output']) / 'videos').rglob('*.mp4')):
                videos.append(dict(pair_id=result['pair_id'], version=result['version'],
                                   validity=result['status'], video=str(video)))
        csv_rows(args.output / 'eval_trials.csv', trials)
        csv_rows(args.output / 'video_index.csv', videos,
                 ('pair_id', 'version', 'validity', 'video'))
        behavior_rows, behavior_summary = paired_behavior(paired['results'])
        sources = {(r['pair_id'], r['version']): r for r in paired['results']}
        for row in behavior_rows:
            result = sources[row['pair_id'], row['version']]
            trial, = result['trials']
            truth_file = Path(result['output']) / 'metrics' / trial['branch'] / 'truth.jsonl'
            facts = [json.loads(line) for line in truth_file.read_text().splitlines()]
            if len(facts) != trial['steps']:
                raise ValueError('Incomplete physical truth trace: '+str(truth_file))
            row.update(physical_stages(facts))
        for version in ('v1', 'v2'):
            valid = [r for r in behavior_rows if r['version'] == version
                     and r['pair_id'] in behavior_summary['matched_pair_ids']]
            behavior_summary[version]['physical_stages'] = {
                stage: proportion(sum(r[stage] for r in valid), len(valid))
                for stage in ('grasped_bag', 'bag_inside', 'closed_after_open', 'started',
                              'started_with_bag_inside_and_door_closed')}
        csv_rows(args.output / 'closed_loop_metrics.csv', behavior_rows)
        summary['closed_loop'] = behavior_summary
        summary['paired'] = {key: paired[key] for key in
            ('attempts_completed','valid_pairs','valid_pair_ids','invalid_attempts','acceptance')}
    atomic_json(args.output / 'evaluation_summary.json', summary)
    print(json.dumps(dict(windows=len(windows), episodes=summary['independent_episodes'],
                         state_scores=summary['state_scores'], paired=summary.get('paired'))))


if __name__ == '__main__':
    main()
