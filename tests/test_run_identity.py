"""Scientific identity stays separate from experiment and process numbering."""
import json
from pathlib import Path
import sys

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'train'))
import run_identity as identity


def test_unknown_version_cannot_create_an_experiment_version():
    with pytest.raises(ValueError, match='登记'):
        identity.training_base('V_12', 40, 'memory')


@pytest.mark.parametrize('alias', ['2.1', 'v2.1', 'V2_1', 'V_2_1'])
def test_minor_version_and_run_fields_are_independent(alias):
    fields = identity.validate_run_identity('v2_1__test02__exp12__run003', alias)
    assert fields['model_version'] == 'V_2_1'
    assert fields['test_id'] == 'test02'
    assert fields['experiment_id'] == 'exp12'
    assert fields['training_run'] == 'run003'


def test_version_mismatch_and_undeclared_patch_are_rejected():
    with pytest.raises(ValueError, match='版本'):
        identity.validate_run_identity('v1__test02__exp12__run003', 'V_2_1')
    with pytest.raises(ValueError, match='登记'):
        identity.validate_run_identity('v2_1_1__test02__exp12__run003', 'V_2_1_1')
    assert identity.canonical_version('v2.1.1') == 'V_2_1_1'


def test_known_wrong_legacy_version_is_corrected_without_renaming(tmp_path):
    root = tmp_path / 'V_12_memory_task40_train01'
    config = root / 'config/web_launch.json'
    config.parent.mkdir(parents=True)
    original = dict(version='V_12', method='memory', task=40)
    config.write_text(json.dumps(original))
    fields = identity.training_identity(root, 'memory_v1')
    assert fields['version'] == 'V_1'
    assert fields['experiment_id'] == 'exp12'
    assert fields['training_id'] == root.name
    assert fields['legacy_version'] == 'V_12'
    assert json.loads(config.read_text()) == original


def test_cumulative_step_supports_minor_versions_and_large_counts():
    run = 'v2_1__test02__exp12__run003'
    assert identity.checkpoint_step(run + '__step1000000', run) == 1000000
    assert identity.checkpoint_step('step-002000', 'V_1_bas_task40') == 2000


def test_allocator_requires_explicit_namespace_and_does_not_infer_from_legacy_names(tmp_path):
    training, deployments = tmp_path/'training', tmp_path/'deployments'
    legacy = training/'V_12_memory_task40_train99'
    legacy.mkdir(parents=True)

    first, number = identity.reserve_training(training, deployments, 'V_1',
        test_id='test01', experiment_id='exp12')
    assert (first, number) == ('v1__test01__exp12__run001', 1)
    second, number = identity.reserve_training(training, deployments, 'V_1',
        test_id='test01', experiment_id='exp12')
    assert (second, number) == ('v1__test01__exp12__run002', 2)
    other, number = identity.reserve_training(training, deployments, 'V_1',
        test_id='test02', experiment_id='exp12')
    assert (other, number) == ('v1__test02__exp12__run001', 1)

    with pytest.raises(ValueError, match='test_id'):
        identity.reserve_training(training, deployments, 'V_1', test_id='test1', experiment_id='exp12')
