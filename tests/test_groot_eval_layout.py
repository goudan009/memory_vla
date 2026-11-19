"""Version allocation, legacy migration, and viewer discovery."""

import importlib.util
import json
from pathlib import Path
import subprocess

import pytest


ROOT = Path(__file__).resolve().parents[1]


def module(name, relative):
    spec = importlib.util.spec_from_file_location(name, ROOT / relative)
    result = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(result)
    return result


output = module('layout_output', 'evaluation/scripts/groot_eval_output.py')
viewer = module('layout_viewer', 'evaluation/scripts/view_results.py')
organizer = module('layout_organizer', 'evaluation/scripts/organize_results.py')


def bundle(path, status='complete'):
    path.mkdir(parents=True)
    result = {'status': status, 'evaluation_mode': 'official-eval', 'task_id': 22,
              'model_name': 'groot', 'requested_episodes': 10,
              'configuration': {'resume_count': 0}}
    (path / 'results.json').write_text(json.dumps(result))
    (path / 'episodes.jsonl').write_text(json.dumps({
        'valid': True, 'success': False, 'q_score': 0.0, 'task_progress': 20,
        'gpu_index': 0, 'env_idx': 0, 'instance_id': 0, 'seed': 0,
    }) + '\n')
    return result


def test_version_selection_and_protocol_are_in_plan_without_writes():
    for version, group in [('V_1', 'baseline'), ('V_12345', 'experment_2'), ('memory_v1', 'ablation_1')]:
        run = subprocess.run(['bash', str(ROOT / 'evaluation/scripts/run_groot_eval.sh'),
                              '--version', version, '--group', group, '--dry-run'], capture_output=True, text=True)
        assert run.returncode == 0, run.stderr
        plan, _ = json.JSONDecoder().raw_decode(run.stdout)
        assert Path(plan['output']).parent == ROOT / 'evaluation' / version / group
        assert plan['version'] == version
        assert plan['group'] == group
        assert plan['protocol']['id'] == 'b1k-groot-v1'
        assert len(plan['protocol']['sha256']) == 64
    assert not (ROOT / 'evaluation/V_12345').exists()
    assert not (ROOT / 'evaluation/memory_v1').exists()
    for flag in ('--version', '--group'):
        bad = subprocess.run(['bash', str(ROOT / 'evaluation/scripts/run_groot_eval.sh'),
                              flag, '../escape', '--dry-run'], capture_output=True, text=True)
        assert bad.returncode != 0 and f'{flag} must be' in bad.stderr


@pytest.mark.parametrize('legacy,group', [('baseline', 'baseline'), ('experiment_2', 'experment_2')])
def test_old_version_arguments_select_compatible_new_group(legacy, group):
    run = subprocess.run(['bash', str(ROOT / 'evaluation/scripts/run_groot_eval.sh'),
                          '--version', legacy, '--dry-run'], capture_output=True, text=True)
    assert run.returncode == 0, run.stderr
    plan, _ = json.JSONDecoder().raw_decode(run.stdout)
    assert plan['version'] == 'V_1' and plan['group'] == group
    assert Path(plan['output']).parent == ROOT / 'evaluation/V_1' / group


def test_legacy_migration_keeps_live_writer_then_finalizes(tmp_path):
    completed = tmp_path / 'groot_task22_1'
    live = tmp_path / 'groot_task26_1'
    bundle(completed)
    live_result = bundle(live, 'running')
    ledger = tmp_path / 'baseline/results.md'
    ledger.parent.mkdir()
    ledger.write_text('Historical results\n')
    original = (completed / 'episodes.jsonl').read_bytes()
    organizer.organize(tmp_path)
    assert not completed.exists()
    assert (tmp_path / 'V_1/baseline/groot_task22_1/episodes.jsonl').read_bytes() == original
    target = tmp_path / 'V_1/baseline/groot_task26_1'
    assert not target.exists()
    assert live.exists()
    assert len(viewer.load_runs(tmp_path)) == 2
    ledger = tmp_path / 'V_1/baseline/results.md'
    assert not (tmp_path / 'V_1/experment_1/results.md').exists()
    before = ledger.read_bytes()
    organizer.organize(tmp_path)
    assert ledger.read_bytes() == before
    live_result['status'] = 'complete'
    (live / 'results.json').write_text(json.dumps(live_result))
    organizer.organize(tmp_path)
    assert target.is_dir() and not live.exists()
    assert ledger.read_bytes() == before == b'Historical results\n'


def test_viewer_separates_same_run_names_by_version(tmp_path):
    bundle(tmp_path / 'baseline/groot_task22_1')
    bundle(tmp_path / 'experiment_1/groot_task22_1')
    diagnostics = tmp_path / 'baseline/groot_task22_1/diagnostics/gpu0'
    bundle(diagnostics)
    runs = viewer.load_runs(tmp_path)
    assert {run['_folder'] for run in runs} == {'baseline/groot_task22_1', 'experiment_1/groot_task22_1'}


def test_migration_moves_old_groups_without_rewriting_results(tmp_path):
    old = tmp_path / 'experiment_1/groot_task22_1'
    bundle(old)
    original = {p.name: p.read_bytes() for p in old.iterdir()}
    organizer.organize(tmp_path)
    new = tmp_path / 'V_1/experment_1/groot_task22_1'
    assert not old.parent.exists()
    assert {p.name: p.read_bytes() for p in new.iterdir()} == original
    organizer.organize(tmp_path)
    run = viewer.load_runs(tmp_path)[0]
    assert (run['version'], run['group']) == ('V_1', 'experment_1')


def test_nested_discovery_supports_custom_names_and_narrow_roots(tmp_path):
    run_dir = tmp_path / 'custom_version/custom_group/groot_task22_1'
    result = bundle(run_dir)
    result.update(version='custom_version', group='custom_group')
    (run_dir / 'results.json').write_text(json.dumps(result))
    bundle(run_dir / 'diagnostics/gpu0')
    bundle(run_dir / '.runtime/gpu1')
    for root in (tmp_path, tmp_path / 'custom_version', run_dir.parent, run_dir):
        runs = viewer.load_runs(root)
        assert len(runs) == 1
        assert (runs[0]['version'], runs[0]['group']) == ('custom_version', 'custom_group')


def test_migration_refuses_to_overwrite_an_existing_run(tmp_path):
    bundle(tmp_path / 'baseline/groot_task22_1')
    bundle(tmp_path / 'V_1/baseline/groot_task22_1')
    with pytest.raises(FileExistsError):
        organizer.organize(tmp_path)
    assert (tmp_path / 'baseline/groot_task22_1/results.json').is_file()
    assert (tmp_path / 'V_1/baseline/groot_task22_1/results.json').is_file()
