"""Keep official terminal outcomes separate from cumulative GR00T progress."""
import json
from pathlib import Path
import sys

import pytest

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'evaluation/scripts'))
import b1k_metrics
import groot_eval_results
import groot_eval_output
import view_results


def test_native_terminal_success_is_not_q_threshold_or_ever_success():
    records=[dict(valid=True,success=True,q_score=.5,task_progress=100,native_termination={'success':False}),
             dict(valid=True,success=True,q_score=1,task_progress=25,b1k_success=True),
             dict(valid=True,success=False,q_score=0,task_progress=80),
             dict(valid=False,success=True,q_score=1,b1k_success=True)]
    summary=b1k_metrics.summarize(records,'official-eval',10)
    assert summary['mean_q_score']==.5 and summary['mean_q_score_percent']==50
    assert summary['full_success_rate']==.5 and summary['success_episodes']==2
    assert summary['missing_success_episodes']==1 and summary['valid_episodes']==3
    assert summary['metric_coverage_complete'] is False
    assert b1k_metrics.terminal_success(records[0]) is False
    assert b1k_metrics.terminal_success(records[2]) is None
    assert b1k_metrics.terminal_success(dict(q_score=1,success=True)) is None
    assert b1k_metrics.terminal_success(dict(b1k_success=True,native_termination={'success':False})) is False


@pytest.mark.parametrize('mode',['fast-debug',None])
def test_debug_and_unknown_protocol_never_produce_official_results(mode):
    result=b1k_metrics.summarize([dict(valid=True,q_score=1,b1k_success=True)],mode)
    assert result['mean_q_score'] is None and result['full_success_rate'] is None


def test_native_false_and_zero_survive_writer_public_merge_and_readonly_viewer(tmp_path):
    run=tmp_path/'run';run.mkdir()
    (run/'run_meta.json').write_text(json.dumps(dict(task_id=0,task_name='turning_on_radio',episodes=10)))
    writer=groot_eval_results.EpisodeWriter(run)
    writer.append(instance_id=0,success=True,task_progress=100,valid=True,sim_steps=8,
                  q_score=0,native_termination={'success':False})
    summary=json.loads((run/'results.json').read_text())
    assert summary['success_rate']==1 and summary['b1k_metrics']['full_success_rate']==0
    assert summary['b1k_metrics']['mean_q_score']==0
    record=json.loads((run/'episodes.jsonl').read_text())
    assert record['b1k_success'] is False
    before={file:file.read_bytes() for file in run.iterdir() if file.is_file()}
    viewed=view_results.load_runs(tmp_path)[0]
    assert viewed['b1k_metrics']['full_success_rate']==0
    assert viewed['_episodes'][0]['b1k_metrics']==dict(q_score=0,full_success=False)
    assert all(file.read_bytes()==data for file,data in before.items())
    updated=groot_eval_output.update_results(run,updates={'status':'complete'})
    assert updated['b1k_metrics']['full_success_rate']==0
    assert updated['success_rate']==1 and updated['mean_task_progress']==100


def test_legacy_native_flags_recovered_without_rewriting_or_inventing(tmp_path):
    run=tmp_path/'run';run.mkdir()
    (run/'results.json').write_text(json.dumps(dict(evaluation_mode='official-eval',completed_episodes=2)))
    (run/'episodes.jsonl').write_text('\n'.join(json.dumps(e) for e in [
        dict(valid=True,success=True,q_score=1,native_termination={'success':True}),
        dict(valid=True,success=True,task_progress=100)]))
    before=(run/'episodes.jsonl').read_bytes()
    result=view_results.load_runs(tmp_path)[0]
    assert result['b1k_metrics']['full_success_rate']==1
    assert result['b1k_metrics']['success_episodes']==1
    assert result['b1k_metrics']['missing_success_episodes']==1
    assert result['_episodes'][1]['b1k_metrics']['full_success'] is None
    assert (run/'episodes.jsonl').read_bytes()==before
