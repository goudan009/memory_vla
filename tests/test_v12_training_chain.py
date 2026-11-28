"""Native CPU contract tests; run with the installed N1.7 Python."""
import json
from pathlib import Path
import random
import sys
import tempfile
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / 'src'), str(ROOT / 'train')]
NATIVE = Path('/nvme3/projects/GR00T/Isaac-GR00T-n1d7-ace36d9')


class CursorTest(unittest.TestCase):
    def setUp(self):
        if Path(sys.prefix).resolve() != (NATIVE / '.venv').resolve():
            self.skipTest('run the sampler tests in the pinned N1.7 environment')
        try:
            import numpy as np
            import torch
        except ImportError:
            self.skipTest('requires installed native torch/numpy')
        sys.path.insert(0, str(NATIVE))
        from gr00t.data.dataset.sharded_mixture_dataset import ShardedMixtureDataset
        from v1_2_groot_baseline.b1k2026_resume import resumable_type

        class Shard:
            def __len__(self):
                return 2

            def get_shard_length(self, index):
                return 3

            def get_shard(self, index):
                return [(index, i, random.random(), float(np.random.rand()), float(torch.rand(())))
                        for i in range(3)]

        class Dataset(resumable_type(ShardedMixtureDataset)):
            def merge_statistics(self):
                pass

        self.make = lambda: Dataset([Shard()], [1.], None, seed=13, num_shards_per_epoch=2)

    def test_resume_inside_shard_at_boundary_and_epoch(self):
        uninterrupted = self.make()
        stream = iter(uninterrupted)
        expected = [next(stream) for _ in range(19)]
        for cut in (1, 3, 6, 8):
            before = self.make()
            iterator = iter(before)
            for _ in range(cut):
                next(iterator)
            state = json.loads(json.dumps(before.state_dict()))
            resumed = self.make()
            resumed.load_state_dict(state)
            actual = iter(resumed)
            self.assertEqual([next(actual) for _ in range(19-cut)], expected[cut:])
            self.assertEqual(resumed.cursor['samples'], 19)

    def test_preprocessing_does_not_advance_training_rng(self):
        import numpy as np
        import torch
        data = self.make()
        python_state, numpy_state, torch_state = random.getstate(), np.random.get_state(), torch.get_rng_state().clone()
        next(iter(data))
        self.assertEqual(random.getstate(), python_state)
        self.assertTrue(np.array_equal(np.random.get_state()[1], numpy_state[1]))
        self.assertTrue(torch.equal(torch.get_rng_state(), torch_state))

    def test_native_training_config_validates_before_model_loading(self):
        import importlib.util
        spec=importlib.util.spec_from_file_location('v12_worker',ROOT/'scripts/train_v12_b1k2026.py')
        worker=importlib.util.module_from_spec(spec);spec.loader.exec_module(worker)
        from gr00t.configs.finetune_config import FinetuneConfig
        from v12_workbench import FINETUNE_REQUIRED
        from dataclasses import asdict
        ft=asdict(FinetuneConfig(base_model_path='fixture',dataset_path='fixture',embodiment_tag='NEW_EMBODIMENT'))
        ft={key:ft[key] for key in FINETUNE_REQUIRED}
        ft.update(global_batch_size=1,gradient_accumulation_steps=1,max_steps=4,save_steps=2,
                  dataloader_num_workers=0,shard_size=4,num_shards_per_epoch=2)
        plan=dict(upstream=str(NATIVE),request=dict(finetune=ft,base_checkpoint='/nvme3/models/GR00T-N1.7-3B',
                                                   seed=42,gradient_checkpointing=False))
        manifest=dict(views=dict(train=dict(path='/fixture/train')))
        config,_=worker.native_config(Path('/fixture/job'),plan,manifest)
        config.validate()
        self.assertEqual(config.data.datasets[0].embodiment_tag,'new_embodiment')
        self.assertEqual(config.data.modality_configs['new_embodiment']['action'].delta_indices,list(range(16)))

    def test_single_gpu_unwrap_does_not_import_unused_deepspeed(self):
        import importlib.util
        import torch
        import accelerate.utils.other as other
        spec=importlib.util.spec_from_file_location('v12_worker',ROOT/'scripts/train_v12_b1k2026.py')
        worker=importlib.util.module_from_spec(spec);spec.loader.exec_module(worker)
        original=other.is_deepspeed_available
        model=torch.nn.Linear(2,2)
        with worker.single_gpu_unwrap():
            self.assertIs(other.extract_model_from_parallel(model),model)
        self.assertIs(other.is_deepspeed_available,original)


class SplitTest(unittest.TestCase):
    def setUp(self):
        try:
            import pyarrow as pa
            import pyarrow.parquet as pq
        except ImportError:
            self.skipTest('requires installed native pyarrow')
        from v1_2_groot_baseline.b1k2026_data import CAMERAS
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = root = Path(self.temp.name)
        self.source = source = root / 'source'
        (source / 'meta/episodes/chunk-000').mkdir(parents=True)
        (source / 'data/chunk-000').mkdir(parents=True)
        features = {'observation.state': {'dtype': 'float32', 'shape': [61]},
                    'action': {'dtype': 'float32', 'shape': [23]}}
        for camera in CAMERAS.values():
            features[camera] = {'dtype': 'video', 'shape': [16,16,3]}
            video = source / 'videos' / camera / 'chunk-000/file-000.mp4'
            video.parent.mkdir(parents=True)
            video.write_bytes(b'video-not-decoded-in-statistics-test')
        info = dict(codebase_version='v3.0', fps=30, features=features, chunks_size=1000,
                    data_path='data/chunk-{chunk_index:03d}/file-{file_index:03d}.parquet',
                    video_path='videos/{video_key}/chunk-{chunk_index:03d}/file-{file_index:03d}.mp4')
        (source / 'meta/info.json').write_text(json.dumps(info))
        tasks = [dict(task_index=0, task='Radio full instruction', task_name='turning_on_radio'),
                 dict(task_index=40, task='Popcorn full instruction', task_name='make_microwave_popcorn')]
        (source / 'meta/tasks.jsonl').write_text('\n'.join(map(json.dumps, tasks))+'\n')
        pq.write_table(pa.Table.from_pylist([dict(task_index=r['task_index'], task=r['task_name']) for r in tasks]), source/'meta/tasks.parquet')
        episodes, rows = [], []
        for ordinal, eid in enumerate((0, 1, 8000, 8001)):
            task = 0 if eid < 8000 else 40
            ep = dict(episode_index=eid, task_index=task, task_instance_id=ordinal % 2, length=20,
                      dataset_from_index=ordinal*20, dataset_to_index=(ordinal+1)*20,
                      **{'data/chunk_index':0, 'data/file_index':0})
            for camera in CAMERAS.values():
                ep.update({f'videos/{camera}/{k}':v for k,v in dict(chunk_index=0,file_index=0,
                           from_timestamp=ordinal*20/30, to_timestamp=(ordinal+1)*20/30).items()})
            episodes.append(ep)
            for frame in range(20):
                value = float(10 if ordinal % 2 else 1)
                rows.append(dict(episode_index=eid, frame_index=frame, index=ordinal*20+frame,
                    task_index=task, timestamp=frame/30, **{'observation.state':[value]*61, 'action':[value]*23}))
        pq.write_table(pa.Table.from_pylist(episodes), source/'meta/episodes/chunk-000/file-000.parquet')
        pq.write_table(pa.Table.from_pylist(rows), source/'data/chunk-000/file-000.parquet')
        (source/'meta/stats.json').write_text('{"invalid_global_statistics":true}')
        self.split = dict(train=[0,8000],validation=[1,8001])

    def test_native_materialization_statistics_language_and_file_leak(self):
        import pyarrow.parquet as pq
        from v1_2_groot_baseline.b1k2026_training_data import materialize, verify_manifest
        for mode in ('native_tasks','task_description'):
            destination = self.root/mode
            destination.mkdir()
            manifest = materialize(self.source,destination,self.split,NATIVE,mode)
            verify_manifest(destination/'data_manifest.json')
            for name, expected in [('train',1.),('validation',10.)]:
                view = destination/name
                actual = pq.read_table(view/'data/chunk-000/file-000.parquet')
                self.assertEqual(set(actual['episode_index'].to_pylist()),set(self.split[name]))
                stats = json.loads((view/'meta/stats.json').read_text())
                self.assertEqual(stats['action']['mean'],[expected]*23)
                self.assertEqual(manifest['views'][name]['rows'],40)
            text = pq.read_table(destination/'train/meta/tasks.parquet')['task'][0].as_py()
            self.assertEqual(text,'turning_on_radio' if mode=='native_tasks' else 'Radio full instruction')
            leak = destination/'train/data/chunk-000/file-999.parquet'
            leak.write_bytes((self.source/'data/chunk-000/file-000.parquet').read_bytes())
            with self.assertRaisesRegex(ValueError,'unregistered'):
                verify_manifest(destination/'data_manifest.json')

    def test_reject_scene_instance_overlap(self):
        from v1_2_groot_baseline.b1k2026_training_data import validate_split
        import pyarrow as pa
        import pyarrow.parquet as pq
        path = self.source/'meta/episodes/chunk-000/file-000.parquet'
        rows = pq.read_table(path).to_pylist()
        rows[1]['task_instance_id']=0
        pq.write_table(pa.Table.from_pylist(rows),path)
        with self.assertRaisesRegex(ValueError,'scene-instance overlap'):
            validate_split(self.source,self.split)


class ManagerTest(unittest.TestCase):
    def test_initialization_retry_preserves_failed_evidence_and_refuses_training_state(self):
        from v12_workbench import V12TrainingManager, atomic
        with tempfile.TemporaryDirectory() as directory:
            manager=V12TrainingManager(directory)
            run='v0_0__test99__exp99__run999'
            job=manager.training/run;job.mkdir(parents=True)
            plan=dict(run_id=run,adapter_source_sha256={'old':'hash'},recipe_sha256='same-recipe',
                      request=dict(gpu_index=1,minimum_free_mib=30000,notes='V0_0 engineering'))
            status=dict(state='failed',worker_alive=False,error='CUDA_HOME missing')
            request=dict(preview=True,reason='single-GPU optional import fix',expected_adapter_source_sha256={})
            with patch('v12_workbench.ADAPTER_FILES',()),patch('v12_workbench.gpu_info',return_value=[dict(index=1,total_mb=49140,used_mb=6000)]),patch.object(manager,'_spawn',return_value=dict(run_id=run)):
                self.assertEqual(manager._retry_failed(job,plan,status,request)['optimizer_step'],0)
                result=manager._retry_failed(job,plan,status,dict(request,preview=False))
                self.assertEqual(result['run_id'],run)
                event=json.loads((job/'retry_events.jsonl').read_text())
                self.assertEqual(event['previous_state']['error'],'CUDA_HOME missing')
                self.assertEqual(json.loads((job/'job.json').read_text())['recipe_sha256'],'same-recipe')
                (job/'metrics.jsonl').write_text('{"step":1}\n')
                with self.assertRaisesRegex(ValueError,'training state'):
                    manager._retry_failed(job,plan,status,request)

    def test_current_training_identity_rejects_other_benchmarks_before_resource_checks(self):
        from v12_workbench import V12TrainingManager
        with tempfile.TemporaryDirectory() as directory:
            manager=V12TrainingManager(directory)
            request={key:None for key in ('base_checkpoint','base_sha256','data_manifest','finetune',
                'gpu_index','seed','gradient_checkpointing','minimum_free_mib')}
            request.update(version='V0_0',run_id='v0_0__test99__exp99__run999',benchmark='b1k',
                           engineering_only=True,initiator='agent',notes='V0_0 engineering')
            with self.assertRaisesRegex(ValueError,'exclusively'):
                manager.prepare(request)

    def test_missing_request_is_a_validation_error(self):
        from v12_workbench import V12TrainingManager
        with tempfile.TemporaryDirectory() as directory:
            manager=V12TrainingManager(directory)
            for request in ({},dict(version='V_1_2',run_id='v1_2__test99__exp99__run999',
                                     engineering_only=True,initiator='agent',notes='V1.2 engineering')):
                with self.assertRaises(ValueError):
                    manager.prepare(request)

    def test_restart_detects_dead_worker_without_overwriting_terminal_status(self):
        from v12_workbench import V12TrainingManager, atomic
        with tempfile.TemporaryDirectory() as directory:
            manager=V12TrainingManager(directory)
            run='v1_2__test99__exp99__run999'
            job=manager.training/run
            job.mkdir(parents=True)
            atomic(job/'job.json',dict(run_id=run,request=dict(gpu_index=0,notes='V1.2 engineering')))
            atomic(job/'status.json',dict(state='running',pid=999999999,step=2))
            self.assertEqual(manager.snapshot()['jobs'][0]['state'],'failed')
            atomic(job/'status.json',dict(state='paused',step=2))
            self.assertEqual(manager.snapshot()['jobs'][0]['state'],'paused')

    def test_spawn_does_not_overwrite_a_fast_workers_terminal_state(self):
        from types import SimpleNamespace
        from v12_workbench import V12TrainingManager, atomic
        with tempfile.TemporaryDirectory() as directory:
            manager=V12TrainingManager(directory)
            job=Path(directory)/'job';job.mkdir()
            plan=dict(run_id='fixture',request=dict(gpu_index=1,notes='V1.2 engineering'))
            def popen(*args,**kwargs):
                atomic(job/'status.json',dict(state='complete',step=4))
                return SimpleNamespace(pid=999999999,poll=lambda:0)
            with patch('v12_workbench.subprocess.Popen',side_effect=popen):
                manager._spawn(job,plan)
            self.assertEqual(json.loads((job/'status.json').read_text())['state'],'complete')


if __name__=='__main__':
    unittest.main()
