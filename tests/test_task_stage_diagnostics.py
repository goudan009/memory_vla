import json
from pathlib import Path
import sys

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'evaluation/scripts'))
import task_stage_diagnostics as stages


@pytest.fixture
def run(tmp_path):
    (tmp_path/'results.json').write_text(json.dumps(dict(task_id=40, evaluation_mode='official-eval',
        configuration=dict(video_steps_per_render=24, video_fps=20))))
    return tmp_path


def episode(index=1):
    return dict(episode_index=index,instance_id=0,seed=18000,gpu_index=0,env_idx=0,
                sim_steps=1000,valid=True,success=False,b1k_success=False,q_score=0.,task_progress=50.)


def trace(step, opened=False, held=False, inside=False, on=False):
    return dict(action_steps=step,instance_id=0,seed=18000,env_idx=0,episode_number=1,
        microwaves={'oven':{'Open':opened,'ToggledOn':on}},
        popcorn={'bag':{'inside':{'oven':inside}}},
        grasped={'robot/left':'bag' if held else None},grasp_states={'robot/left/bag':'TRUE' if held else 'FALSE'})


def event(start=80,end=160,**changes):
    return dict(stage_id='S4',action_start=start,action_end=end,
        observed_behavior='pull_door',outcome='failed',error_type='execution_failure',
        confidence=.8,evidence_frames=[1,2],explanation='视频可见拉门尝试后门仍关闭',**changes)


def test_closed_door_is_not_a_failed_open_attempt(run):
    report=stages.analyze_episode(run,episode(),traces=[trace(80),trace(160)],events=[])
    door=next(s for s in report['stages'] if s['id']=='S4')
    assert door['attempted'] is None
    assert door['completed'] is False
    assert report['failures']==[]


def test_frame_index_preserves_recorded_action_offset_and_falls_back_for_old_video(run):
    path = run/'stages'
    path.mkdir()
    rows = [dict(instance_id=0, seed=18000, env_idx=0, episode_number=1,
                 frame_index=i, action_steps=23+24*i, sim_time_s=10+i*.8, video_fps=20)
            for i in range(4)]
    log = path/'video_gpu0_env0.jsonl'
    log.write_text('\n'.join(map(json.dumps, rows)))
    report = stages.analyze_episode(run, episode(), traces=[trace(47, opened=True)], events=[])
    assert report['time_mapping']['frames'] == rows
    assert report['time_mapping']['approximate'] is False
    assert report['timeline'][0]['video_seconds'] == .05
    assert report['timeline'][0]['video_time_approximate'] is False
    # A physical state sampled between saved video frames remains approximate.
    report = stages.analyze_episode(run, episode(), traces=[trace(80)], events=[])
    assert report['timeline'][0]['video_time_approximate'] is True
    # A repeated reset with the same instance+seed cannot be mixed silently.
    log.write_text(log.read_text()+'\n'+json.dumps(dict(rows[0], episode_number=2)))
    report = stages.analyze_episode(run, episode(), traces=[], events=[])
    assert report['time_mapping']['approximate'] is True
    assert report['time_mapping']['warning'] == 'ambiguous_video_identity'
    log.unlink()
    report = stages.analyze_episode(run, episode(), traces=[trace(47)], events=[])
    assert report['time_mapping']['approximate'] is True
    assert report['timeline'][0]['video_seconds'] == 47/480


def test_pull_failure_then_invalid_transition_are_distinct(run):
    report=stages.analyze_episode(run,episode(),traces=[trace(80),trace(160),trace(240,held=True)],events=[event()])
    assert [(f['stage_id'],f['kind']) for f in report['failures']]==[('S4','observed_failure'),('S6','invalid_transition')]
    assert report['failures'][1]['preceding_failure_id']==report['failures'][0]['id']
    assert report['failures'][1]['causal_relationship']=='unknown'


def test_old_physics_does_not_become_current_at_unsampled_video_event(run):
    ev=dict(stage_id='S5',action_start=120,action_end=125,observed_behavior='leave_for_bag',
            outcome='attempted',error_type='none',confidence=.8,evidence_frames=[5],explanation='离开炉门')
    report=stages.analyze_episode(run,episode(),traces=[trace(80),trace(160,opened=True)],events=[ev])
    assert len(report['transitions'])==1
    assert report['transitions'][0]['valid'] is None
    assert report['failures']==[]


def test_earlier_closed_door_does_not_become_outcome_of_later_attempt(run):
    ev=event(start=240,end=320);ev['outcome']='attempted';ev['error_type']='none'
    report=stages.analyze_episode(run,episode(),traces=[trace(80)],events=[ev])
    door=next(s for s in report['stages'] if s['id']=='S4')
    assert door['attempted'] is True and door['completed'] is False
    assert door['attempt_outcome'] is None


def test_history_and_current_state_are_separate(run):
    rows=[trace(80),trace(160,opened=True),trace(240,opened=True,held=True),
          trace(320,opened=True,inside=True),trace(400,inside=True),trace(480,inside=True,on=True)]
    report=stages.analyze_episode(run,episode(),traces=rows,events=[])
    assert report['current_state']['door_open'] is False
    door=next(s for s in report['stages'] if s['id']=='S4')
    assert door['completed'] is True and door['first_completed_step']==160
    assert report['official']['q_score']==0
    assert report['custom_stage_completion']['name']=='Custom Stage Completion'


def test_not_reached_needs_whole_episode_evidence_and_cannot_erase_physics(run):
    ev=dict(stage_id='S4',action_start=0,action_end=1000,observed_behavior='unknown',outcome='not_reached',
            error_type='unknown',confidence=.8,evidence_frames=[0,40],explanation='整集核查')
    report=stages.analyze_episode(run,episode(),traces=[],events=[ev])
    assert next(s for s in report['stages'] if s['id']=='S4')['status']=='not_reached'
    report=stages.analyze_episode(run,episode(),traces=[trace(80,opened=True)],events=[ev])
    assert next(s for s in report['stages'] if s['id']=='S4')['status']=='completed'
    with pytest.raises(ValueError):stages.validate_events([dict(ev,action_end=500)],episode())


def test_trace_matches_gpu_env_instance_and_seed_not_global_episode_number(run):
    path=run/'stages';path.mkdir()
    (path/'task40_gpu0_env0.jsonl').write_text('\n'.join(map(json.dumps,[trace(80),trace(160,opened=True)])))
    (path/'task40_gpu1_env0.jsonl').write_text(json.dumps(trace(80,held=True)))
    report=stages.analyze_episode(run,episode(index=3),events=[])
    assert report['trace_samples']==2
    assert report['failures']==[]


def test_memory_samples_match_episode_without_changing_stage_results(run):
    path=run/'stages';path.mkdir()
    sample=dict(instance_id=0,seed=18000,env_idx=0,session_id='session1',
                episode_id=1,decision_index=10,memory={'read_source':'episode1'},history_mode='normal')
    log=path/'memory_gpu0_env0.jsonl'
    log.write_text('\n'.join(map(json.dumps,[sample,dict(sample,seed=18001)])))
    report=stages.analyze_episode(run,episode(),traces=[],events=[])
    assert report['memory_diagnostics']['sample_count']==1
    assert report['memory_diagnostics']['samples']==[sample]
    assert report['memory_diagnostics']['model_module_cause']=='unknown'
    assert report['official']['q_score']==0 and report['failures']==[]
    log.write_text('\n'.join(map(json.dumps,[sample,dict(sample,session_id='session2')])))
    report=stages.analyze_episode(run,episode(),traces=[],events=[])
    assert report['memory_diagnostics']['sample_count']==0
    assert report['memory_diagnostics']['warning']=='ambiguous_memory_identity'


def test_unknown_grasp_is_not_false_and_multi_target_is_unknown():
    assert stages.physics({'popcorn':{'bag':{}},'grasped':{'left':None}})['bag_held'] is None
    row=trace(80);row['microwaves']['other']={'Open':True}
    assert stages.physics(row)['door_open'] is None
    assert stages.physics(row)['bag_inside'] is None


def test_unknown_review_is_independent_and_suppresses_unconfirmed_events(run):
    raw=run/'stage_diagnostics/vl_events';raw.mkdir(parents=True)
    content=json.dumps({'events':[event()]});(raw/'1.json').write_text(content)
    stages.save_review(run,episode(),dict(decision='unknown',notes='画面遮挡',events=[]))
    report=stages.analyze_episode(run,episode(),traces=[trace(80),trace(160)])
    assert report['events']==[]
    assert (raw/'1.json').read_text()==content
    assert report['human_review']['decision']=='unknown'


def test_unreviewed_vl_candidates_are_visible_but_do_not_change_metrics(run):
    raw=run/'stage_diagnostics/vl_events';raw.mkdir(parents=True)
    (raw/'1.json').write_text(json.dumps({'events':[event()]}))
    report=stages.analyze_episode(run,episode(),traces=[trace(80),trace(160)])
    assert len(report['events'])==1 and report['event_review_required'] is True
    assert report['events_used_for_metrics']==0 and report['failures']==[]
    stages.save_review(run,episode(),dict(decision='confirmed',events=[event()]))
    report=stages.analyze_episode(run,episode(),traces=[trace(80),trace(160)])
    assert report['events_used_for_metrics']==1
    assert report['failures'][0]['kind']=='observed_failure'


def test_batch_unknown_denominators_and_official_values(run):
    (run/'episodes.jsonl').write_text(json.dumps(episode())+'\n')
    report=stages.analyze_run(run)
    metric=report['summary']['stages']['S4']['execution_success_rate']
    assert metric['value'] is None and metric['denominator']==0
    assert report['summary']['stages']['S4']['reach_rate']['unknown_n']==1
    assert report['summary']['official_mean_q']==0.
    assert 'Custom Stage Completion' in stages.export(report)
    assert 'official_q' in stages.export(report,'csv')
    assert '|S4' in stages.export(report,'md')


def test_comparison_rejects_different_protocol_and_keeps_unknown_sample_counts(run):
    (run/'episodes.jsonl').write_text(json.dumps(episode())+'\n')
    report=stages.analyze_run(run)
    other=json.loads(json.dumps(report))
    assert stages.compare(report,other)['matched_n']==0
    report['episodes'][0]['protocol']={'sha256':'same'}
    other['episodes'][0]['protocol']={'sha256':'same'}
    result=stages.compare(report,other)
    assert result['matched_n']==1
    assert result['paired_stage_metrics']['S4']['left_reach']['unknown_n']==1
    assert result['paired_stage_metrics']['S4']['left_execution']['value'] is None


def test_comparison_transition_and_failure_metrics_only_use_paired_episodes(run):
    from copy import deepcopy
    (run/'episodes.jsonl').write_text(json.dumps(episode())+'\n')
    report=stages.analyze_run(run);other=deepcopy(report)
    for item in [report,other]:item['episodes'][0]['protocol']={'sha256':'same'}
    failed=report['episodes'][0]
    failed['transitions']=[{'valid':False}]
    failed['first_observed_failure']={'stage_id':'S6','error_type':'state_order_error'}
    unmatched=deepcopy(failed);unmatched['identity']['seed']=999
    unmatched['transitions']=[{'valid':True}]
    report['episodes'].append(unmatched)
    result=stages.compare(report,other)
    assert result['matched_n']==1
    summary=result['paired_summaries']['left']
    assert summary['valid_n']==1
    assert summary['invalid_transition_rate']['numerator']==1
    assert summary['invalid_transition_rate']['denominator']==1
    assert summary['decision_accuracy']['value']==0
    assert summary['first_observed_failure_distribution']=={'S6:state_order_error':1}
    assert result['paired_summaries']['right']['invalid_transition_rate']['value'] is None


def test_comparison_keeps_official_q_for_the_matched_subset(run):
    from copy import deepcopy
    (run/'episodes.jsonl').write_text(json.dumps(episode())+'\n')
    report=stages.analyze_run(run)
    report['episodes'][0]['protocol']={'sha256':'same'}
    other=deepcopy(report)
    other['episodes'][0]['official']['q_score']=1.
    unmatched=deepcopy(other['episodes'][0])
    unmatched['identity']['seed']=999
    unmatched['official']['q_score']=0.
    other['episodes'].append(unmatched)
    result=stages.compare(report,other)
    assert result['paired_summaries']['left']['official_mean_q']==0.
    assert result['paired_summaries']['right']['official_mean_q']==1.
    # Diagnostic rollouts must never acquire a formal Q through comparison.
    (run/'results.json').write_text(json.dumps(dict(task_id=40,evaluation_mode='debug')))
    diagnostic=stages.analyze_run(run)
    diagnostic['episodes'][0]['protocol']={'sha256':'same'}
    assert stages.compare(diagnostic,other)['paired_summaries']['left']['official_mean_q'] is None


@pytest.mark.parametrize('change',[{'action_start':float('nan')},{'confidence':2},{'stage_id':'S10'},{'action_end':2000}])
def test_reject_untraceable_annotation(change):
    ev=event();ev.update(change)
    with pytest.raises(ValueError):stages.validate_events([ev],episode())
