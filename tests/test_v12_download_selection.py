import hashlib
import importlib.util
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location('v12_download', ROOT / 'scripts/download_v12_resources.py')
download = importlib.util.module_from_spec(spec)
spec.loader.exec_module(download)


def test_pinned_scope_excludes_depth_and_other_tasks():
    manifest = json.loads(download.MANIFEST.read_text())
    demo = next(r for r in manifest['resources'] if r['name'] == 'demos')
    paths = [f['path'] for f in demo['files']]
    assert len(demo['revision']) == 40
    assert len([p for p in paths if p.startswith('annotations/task-0000/')]) == 200
    assert len([p for p in paths if p.startswith('annotations/task-0040/')]) == 200
    assert not any('depth' in p for p in paths)
    assert all(download.select_demo(p) for p in paths)
    assert not download.select_demo('data/chunk-041/file-000.parquet')
    assert not download.select_demo('videos/observation.depth_linear.zed_link_camera_0/chunk-040/file-000.mp4')
    assert not download.select_demo('annotations/task-0007/episode_00070010.json')


def test_integrity_rejects_same_size_corruption(tmp_path):
    path = tmp_path / 'payload'
    path.write_bytes(b'good')
    entry = {'bytes': 4, 'sha256': hashlib.sha256(b'good').hexdigest()}
    assert download.verify(path, entry) == entry['sha256']
    path.write_bytes(b'evil')
    import pytest
    with pytest.raises(AssertionError):
        download.verify(path, entry)
