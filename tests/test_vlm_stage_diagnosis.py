import json
from pathlib import Path
import sys

import pytest

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'evaluation/scripts'))
import vlm_stage_diagnosis as vl
import vlm_stage_protocol


def test_structured_prompt_distinguishes_attempt_and_result():
    prompt=vlm_stage_protocol.prompt('窗口', '证据', 12)
    assert '拉门尝试' in prompt and 'unknown' in prompt and '最大12' in prompt


def test_normalize_preserves_frames_time_and_approximation():
    event=dict(stage_id='S4',frame_start=1,frame_end=3,observed_behavior='pull_door',
        outcome='failed',error_type='execution_failure',confidence=.8,evidence_frames=[1,3],explanation='可见拉门未开')
    report=dict(time_mapping={'video_seconds_per_action_step':1/480})
    result=vl.normalize({'_ok':True,'events':[event]},[20,24,28],report,{'episode_index':1,'sim_steps':1000})[0]
    assert result['evidence_frames']==[20,28]
    assert result['time_start']==1 and result['action_start']==480
    assert result['action_time_approximate'] is True
    with pytest.raises(ValueError):vl.normalize({'_ok':True,'events':[dict(event,evidence_frames=[4])]},[20,24,28],report,{'episode_index':1,'sim_steps':1000})


def test_normalize_uses_recorded_video_steps_and_sim_time_instead_of_zero_origin():
    event = dict(stage_id='S4', frame_start=1, frame_end=2, observed_behavior='pull_door',
                 outcome='attempted', error_type='none', confidence=.8,
                 evidence_frames=[1,2], explanation='可见门附近动作')
    rows = [dict(frame_index=i, action_steps=23+24*i, sim_time_s=10+i*.8, video_fps=20)
            for i in range(3)]
    report = dict(time_mapping={'video_seconds_per_action_step':1/480,
                               'approximate':False, 'frames':rows})
    result = vl.normalize({'_ok':True, 'events':[event]}, [0,2], report,
                          {'episode_index':1, 'sim_steps':1000})[0]
    assert (result['action_start'], result['action_end']) == (23, 71)
    assert (result['time_start'], result['time_end']) == (0, .1)
    assert (result['sim_time_start_s'], result['sim_time_end_s']) == (10, 11.6)
    assert result['action_time_approximate'] is False


def test_legacy_summary_cannot_be_silently_used_as_events():
    with pytest.raises(ValueError):vl.normalize({'_ok':True,'behavior_class':'unclear'},[],{}, {})


def test_offline_service_does_not_create_fake_model_output(tmp_path,monkeypatch):
    monkeypatch.setattr(vl.vlm_diagnosis,'health',lambda endpoint:{'ok':False})
    with pytest.raises(RuntimeError):vl.diagnose(tmp_path,{'episode_index':1},tmp_path/'video.mp4')
    assert not (tmp_path/'stage_diagnostics').exists()


def test_clip_pipeline_uses_existing_service_and_keeps_raw_output(tmp_path,monkeypatch):
    episode={'episode_index':1,'sim_steps':960}
    (tmp_path/'results.json').write_text(json.dumps({'configuration':{'video_steps_per_render':24,'video_fps':20}}))
    monkeypatch.setattr(vl.stages,'analyze_episode',lambda *args:{'timeline':[],'failures':[],'time_mapping':{'video_seconds_per_action_step':1/480}})
    monkeypatch.setattr(vl.vlm_diagnosis,'health',lambda endpoint:{'ok':True})
    monkeypatch.setattr(vl,'_extract',lambda video,frames:['image']*len(frames))
    requests=[]
    def analyze(endpoint,payload):
        requests.append(payload)
        return {'_ok':True,'events':[]}
    monkeypatch.setattr(vl.vlm_diagnosis,'_analyze',analyze)
    result=vl.diagnose(tmp_path,episode,tmp_path/'video.mp4')
    assert requests and all(r['mode']=='task40_stage' for r in requests)
    assert result['windows'][0]['raw_output']=={'_ok':True,'events':[]}
    assert (tmp_path/'stage_diagnostics/vl_events/1.json').is_file()


def test_bad_model_event_is_archived_and_not_silently_accepted(tmp_path,monkeypatch):
    episode={'episode_index':1,'sim_steps':960}
    (tmp_path/'results.json').write_text(json.dumps({'configuration':{}}))
    monkeypatch.setattr(vl.stages,'analyze_episode',lambda *args:{'timeline':[],'failures':[],'time_mapping':{'video_seconds_per_action_step':1/480}})
    monkeypatch.setattr(vl.vlm_diagnosis,'health',lambda endpoint:{'ok':True})
    monkeypatch.setattr(vl,'_extract',lambda video,frames:['image']*len(frames))
    raw={'_ok':True,'events':[{'stage_id':'made-up'}]}
    monkeypatch.setattr(vl.vlm_diagnosis,'_analyze',lambda *args:raw)
    result=vl.diagnose(tmp_path,episode,tmp_path/'video.mp4')
    assert result['events']==[] and result['failed_windows']>0
    assert result['windows'][0]['raw_output']==raw
    assert result['windows'][0]['error']


def test_placement_claim_needs_microwave_destination_and_visible_release():
    # Real calibration failure: Qwen called placement at the stovetop S7 success.
    event=dict(stage_id='S7',frame_start=1,frame_end=3,observed_behavior='place_bag',
        outcome='completed',error_type='none',confidence=1.,evidence_frames=[1,3],
        explanation='机器人持续握持袋子并将其放置在灶台边缘')
    report=dict(time_mapping={'video_seconds_per_action_step':1/480})
    episode={'episode_index':7,'sim_steps':6476}
    for evidence in [None, {'destination':'stovetop','bag_inside_visible':False,'release_visible':False},
                     {'destination':'microwave_interior','bag_inside_visible':True,'release_visible':'true'}]:
        raw=dict(event)
        if evidence is not None: raw['stage_evidence']=evidence
        result=vl.normalize({'_ok':True,'events':[raw]},[60,65,73],report,episode)[0]
        assert result['outcome']=='unknown'
        assert result['model_outcome']=='completed' and result['semantic_validation']['accepted'] is False
        assert result['explanation']==event['explanation']
    raw=dict(event,stage_evidence={'destination':'microwave_interior','bag_inside_visible':True,'release_visible':True})
    result=vl.normalize({'_ok':True,'events':[raw]},[60,65,73],report,episode)[0]
    assert result['outcome']=='completed'
    assert result['semantic_validation']['accepted'] is True


def test_force_reanalysis_preserves_prior_raw_model_output(tmp_path,monkeypatch):
    episode={'episode_index':1,'sim_steps':960}
    (tmp_path/'results.json').write_text(json.dumps({'configuration':{}}))
    path=tmp_path/'stage_diagnostics/vl_events/1.json'
    path.parent.mkdir(parents=True)
    original={'source':'vl','events':[],'windows':[{'raw_output':{'events':[{'explanation':'old raw'}]}}]}
    path.write_text(json.dumps(original))
    monkeypatch.setattr(vl.stages,'analyze_episode',lambda *args:{'timeline':[],'failures':[],'time_mapping':{'video_seconds_per_action_step':1/480}})
    monkeypatch.setattr(vl.vlm_diagnosis,'health',lambda endpoint:{'ok':True})
    monkeypatch.setattr(vl,'_extract',lambda video,frames:['image']*len(frames))
    monkeypatch.setattr(vl.vlm_diagnosis,'_analyze',lambda *args:{'_ok':True,'events':[]})
    result=vl.diagnose(tmp_path,episode,tmp_path/'video.mp4',force=True)
    archived=[json.loads(p.read_text()) for p in path.parent.glob('history/1/*.json')]
    assert original in archived
    assert result['prompt_version']==vlm_stage_protocol.PROMPT_VERSION
