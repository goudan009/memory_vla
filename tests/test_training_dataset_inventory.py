"""Readiness requires every Episode's converted data and all three videos."""

import importlib.util
import json
from pathlib import Path


def inventory_module():
    spec=importlib.util.spec_from_file_location('dataset_inventory_test',
        Path(__file__).resolve().parents[1]/'train/dataset_inventory.py')
    module=importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def create_task(root, task=0):
    data=root/f'b1k2025_groot_task{task}_v1_200'
    meta=data/'meta'
    meta.mkdir(parents=True)
    for name in ['modality.json','stats.json','relative_stats.json']:
        (meta/name).write_text('{}')
    (meta/'tasks.jsonl').write_text('{}\n')
    (meta/'episodes.jsonl').write_text(''.join(json.dumps(dict(episode_index=e,length=1000))+'\n' for e in range(200)))
    (meta/'info.json').write_text(json.dumps(dict(fps=30,total_episodes=200,
        features={'observation.state':dict(shape=[82]),'action':dict(shape=[23])})))
    for e in range(200):
        paths=[data/f'data/task-{task:04d}/episode_{e:08d}.parquet']
        paths += [data/f'videos/task-{task:04d}/observation.images.rgb.{cam}_256_256/episode_{e:08d}.mp4'
                  for cam in ('head','left_wrist','right_wrist')]
        for path in paths:
            path.parent.mkdir(parents=True,exist_ok=True)
            path.write_bytes(b'converted')
    return data


def test_old_ready_dataset_without_local_preparation_can_be_registered(tmp_path):
    module=inventory_module()
    data=create_task(tmp_path/'datasets')
    training=tmp_path/'training'
    split=training/'V_1_bas_task0/config/data_split.json'
    split.parent.mkdir(parents=True)
    value=dict(task=0,seed=42,train=list(range(180)),validation=list(range(180,200)))
    split.write_text(json.dumps(value))
    rows=module.remote_inventory(data.parent,training,register=True)
    assert rows[0]['ready'] and rows[0]['video_files']==600 and rows[0]['parquet_files']==200
    assert json.loads((data/'preparation/data_split.json').read_text())==value
    assert (data/'preparation/ready.json').is_file()


def test_missing_camera_file_prevents_ready_marker(tmp_path):
    module=inventory_module()
    data=create_task(tmp_path/'datasets')
    split=data/'preparation/data_split.json'
    split.parent.mkdir()
    split.write_text(json.dumps(dict(task=0,seed=42,train=list(range(180)),validation=list(range(180,200)))))
    missing=data/'videos/task-0000/observation.images.rgb.head_256_256/episode_00000199.mp4'
    missing.unlink()
    rows=module.remote_inventory(data.parent,tmp_path/'training',register=True)
    assert not rows[0]['ready'] and rows[0]['missing']==[str(missing)]
    assert not (data/'preparation/ready.json').exists()
