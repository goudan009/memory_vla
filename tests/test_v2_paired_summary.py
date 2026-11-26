"""Closed-loop summaries count valid matched episodes, not restore attempts."""
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'evaluation/scripts'))
from v2_evaluation_summary import paired_behavior, proportion, physical_stages


def result(pair, version, opened=False, status='complete'):
    trial = dict(decisions=3, steps=24, stop_reason='decision_limit',
        first_open_step=0 if opened else None, first_error=None,
        first_task_success_step=None, door_contact_steps=[4, 5], max_hinge_excursion_rad=.01)
    return dict(pair_id=pair, version=version, instance=4, seed=10,
                status=status, trials=[trial] if status == 'complete' else [])


def test_invalid_restore_is_excluded_and_unmatched_episode_not_used_in_comparison():
    rows, summary = paired_behavior([result('p1', 'v1'), result('p1', 'v2', True),
        result('p2', 'v1', True), result('p2', 'v2', status='invalid')])
    assert len(rows) == 3
    assert summary['matched_pairs'] == 1
    assert summary['v1']['opened']['total'] == 1
    assert summary['v1']['opened']['successes'] == 0
    assert summary['v2']['opened']['successes'] == 1
    assert summary['opening_pair_outcomes']['v2_only'] == 1


def test_empty_denominator_and_small_sample_intervals_are_explicit():
    _, summary = paired_behavior([result('p1', 'v1', status='invalid')])
    assert summary['v1']['opened']['rate'] is None
    assert summary['v1']['opened']['wilson95'] is None
    low, high = proportion(0, 12)['wilson95']
    assert 0 <= low < 1e-12 and .24 < high < .25


def test_physical_stages_require_observed_events_not_just_closed_door_or_start():
    fact = dict(Open=False, Inside=False, ToggledOn=True,
                popcorn={'bag': {}}, grasped={'left': None})
    metrics = physical_stages([fact])
    assert metrics['started']
    assert not metrics['closed_after_open']
    assert not metrics['started_with_bag_inside_and_door_closed']
    assert not metrics['grasped_bag']
    metrics = physical_stages([dict(fact, Open=True, ToggledOn=False),
        dict(fact, Inside=True, grasped={'left': 'bag'})])
    assert all(metrics.values())
