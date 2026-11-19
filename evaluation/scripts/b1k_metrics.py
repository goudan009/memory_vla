"""Expose B1K terminal metrics separately from GR00T's cumulative progress.

Q is supplied by OmniGibson TaskMetric, never reconstructed from progress.
Legacy success is recovered only from recorded native terminal information.
These are local run statistics over valid completed Episodes, not a 500-Episode
official leaderboard submission (which also has missing-submission rules).
"""
from __future__ import annotations


def terminal_success(episode: dict) -> bool | None:
    native = episode.get('native_termination')
    if isinstance(native, dict) and type(native.get('success')) is bool:
        return native['success']
    value = episode.get('b1k_success')
    return value if type(value) is bool else None


def episode_metrics(episode: dict) -> dict:
    return dict(q_score=episode.get('q_score'), full_success=terminal_success(episode))


def summarize(records: list[dict], mode: str, requested: int | None = None) -> dict:
    valid = [record for record in records if record.get('valid')]
    scores = [record['q_score'] for record in valid if record.get('q_score') is not None]
    successes = [value for record in valid if (value := terminal_success(record)) is not None]
    official = mode == 'official-eval'
    q = sum(scores) / len(scores) if scores and official else None
    rate = sum(successes) / len(successes) if successes and official else None
    return dict(mean_q_score=q, mean_q_score_percent=q * 100 if q is not None else None,
                full_success_rate=rate, successful_episodes=sum(successes),
                q_score_episodes=len(scores), success_episodes=len(successes),
                missing_q_score_episodes=len(valid)-len(scores),
                missing_success_episodes=len(valid)-len(successes),
                valid_episodes=len(valid), requested_episodes=requested,
                metric_coverage_complete=bool(valid) and len(scores)==len(successes)==len(valid),
                scope='valid_completed_episodes', evaluation_mode=mode)
