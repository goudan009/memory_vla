import importlib.util
from pathlib import Path
import os
import pytest

spec=importlib.util.spec_from_file_location('storage',Path(__file__).resolve().parents[1]/'evaluation/scripts/registry_storage.py')
storage=importlib.util.module_from_spec(spec)
spec.loader.exec_module(storage)

def test_protected_ancestor_and_symlink(tmp_path):
    root=tmp_path/'resources';root.mkdir()
    target=root/'group';target.mkdir()
    protected=target/'managed';protected.mkdir()
    link=root/'alias';link.symlink_to(target,target_is_directory=True)
    req=dict(roots=[str(root)],protected=[str(protected)],path=str(target))
    assert storage.operate(dict(req,action='stat'))['can_delete'] is False
    with pytest.raises(ValueError):
        storage.operate(dict(req,action='trash'))
    with pytest.raises(ValueError):
        storage.operate(dict(req,action='trash',path=str(link/'managed')))

def test_private_staging_and_fifo(tmp_path):
    root=tmp_path/'resources';root.mkdir()
    stage=root/'.workbench-downloads';stage.mkdir()
    link=root/'alias';link.symlink_to(stage,target_is_directory=True)
    with pytest.raises(ValueError):storage.operate(dict(roots=[str(root)],path=str(link)))
    fifo=root/'pipe';os.mkfifo(fifo)
    with pytest.raises(ValueError,match='普通文件'):storage.operate(dict(roots=[str(root)],path=str(fifo)))

def test_training_root_protection(tmp_path):
    root=tmp_path/'resources';root.mkdir()
    training=root/'experiment-output';training.mkdir()
    run=training/'run';run.mkdir()
    req=dict(roots=[str(root)],training_roots=[str(training)],path=str(run))
    assert not storage.operate(dict(req,action='stat'))['can_delete']
    with pytest.raises(ValueError):storage.operate(dict(req,action='trash'))
