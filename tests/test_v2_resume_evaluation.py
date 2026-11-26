"""A previous physical-boundary rejection is recorded invalid, never retried silently."""
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'evaluation/scripts'))
from v2_task40_paired_evaluation import load_resumable_results


def test_resume_preserves_completed_trials_and_classifies_boundary_rejection(tmp_path):
    output = tmp_path/'pair005_v1'
    folder = output/'metrics/restore_seed'
    folder.mkdir(parents=True)
    evidence = folder/'initial_correspondence.json'
    evidence.write_text(json.dumps(dict(language_equal=True, truth_correspondence=dict(
        step=True,decision=True,Open=True,ToggledOn=True,Inside=True,hinge_abs_rad=.0016))))
    (output/'logs').mkdir()
    (output/'logs/failure.txt').write_text(
        'RuntimeError: Restored initial truth or language disagrees with the verified parent boundary\n')
    status = tmp_path/'paired_status.json'
    status.write_text(json.dumps(dict(results=[
        dict(pair_id='pair001',version='v1',status='complete',output=str(tmp_path/'old')),
        dict(pair_id='pair005',version='v1',status='failed',output=str(output),trials=[])])))
    rows = load_resumable_results(status)
    assert rows[0]['status'] == 'complete'
    assert rows[1]['status'] == 'invalid'
    assert rows[1]['validity']['reason'] == 'INVALID_PARENT_BOUNDARY'
    assert 'not rerun' in rows[1]['resume_classification']


def test_zero_exit_parent_boundary_rejection_is_invalid_not_unresolved(tmp_path):
    output = tmp_path/'pair006_v1'
    folder = output/'metrics/restore_seed'
    folder.mkdir(parents=True)
    (folder/'initial_correspondence.json').write_text(json.dumps(dict(language_equal=True,
        truth_correspondence=dict(step=True,decision=True,Open=True,ToggledOn=True,Inside=True,
                                  hinge_abs_rad=.00056))))
    (output/'logs').mkdir()
    # The simulator wrapper writes this exception but currently exits with status 0.
    (output/'logs/failure.txt').write_text(
        'RuntimeError: Restored initial truth or language disagrees with the verified parent boundary\n')
    status = tmp_path/'paired_status.json'
    status.write_text(json.dumps(dict(results=[dict(pair_id='pair006',version='v1',status='failed',
        returncode=0,output=str(output),trials=[])])))

    row, = load_resumable_results(status)

    assert row['status'] == 'invalid'
    assert row['validity']['reason'] == 'INVALID_PARENT_BOUNDARY'


def test_unrelated_failure_with_boundary_mismatch_still_blocks_resume(tmp_path):
    output = tmp_path/'pair007_v1'
    folder = output/'metrics/restore_seed'
    folder.mkdir(parents=True)
    (folder/'initial_correspondence.json').write_text(json.dumps(dict(language_equal=True,
        truth_correspondence=dict(step=True,decision=True,Open=True,ToggledOn=True,Inside=True,
                                  hinge_abs_rad=.00056))))
    (output/'logs').mkdir()
    (output/'logs/failure.txt').write_text('RuntimeError: unrelated controller error\n')
    status = tmp_path/'paired_status.json'
    status.write_text(json.dumps(dict(results=[dict(pair_id='pair007',version='v1',status='failed',
        returncode=1,output=str(output),trials=[])])))
    import pytest
    with pytest.raises(ValueError,match='Cannot safely resume'):
        load_resumable_results(status)


def test_unknown_failed_attempt_blocks_resume_instead_of_being_hidden(tmp_path):
    output = tmp_path/'pair1'
    (output/'metrics').mkdir(parents=True)
    status = tmp_path/'paired_status.json'
    status.write_text(json.dumps(dict(results=[dict(pair_id='pair001',version='v1',
        status='failed',output=str(output))])))
    import pytest
    with pytest.raises(ValueError,match='Cannot safely resume'):
        load_resumable_results(status)
