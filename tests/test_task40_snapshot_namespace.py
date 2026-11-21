import sys
from pathlib import Path


sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'evaluation' / 'scripts'))
from task40_paired_sim import remote_snapshot_name


def test_collection_reuses_seed_on_same_service_without_snapshot_collision(tmp_path):
    branch = 'episode_i1_s76212_A_normal'
    failure = branch + '_failure'
    for name, suffix in ((branch, '_online_prefix'), (failure, '')):
        legacy = tmp_path / (name + suffix)
        legacy.mkdir()
        for run in ('12d637feb663', 'd19e545e8558'):
            remote_name = remote_snapshot_name(Path('runs') / run, name, True)
            # Match the legacy endpoint's Path(branch).name behavior.
            path = tmp_path / (Path(remote_name).name + suffix)
            path.mkdir()
            assert path != legacy
        assert remote_snapshot_name(Path('legacy'), name, False) == name
