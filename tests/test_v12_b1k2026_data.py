"""Run with the pinned N1.7 Python (unittest; no environment installation)."""
import json
from pathlib import Path
import sys
import tempfile
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))
from v1_2_groot_baseline.b1k2026_data import CAMERAS, TASKS, audit_2026, validate_annotation


class AnnotationTest(unittest.TestCase):
    def test_raw_frame_coordinates_with_nonzero_valid_start(self):
        annotation = dict(meta_data=dict(task_duration=2529, valid_duration=[576, 3105]),
                          skill_annotation=[dict(frame_duration=[2644, 3105])])
        self.assertEqual(validate_annotation(annotation), 2529)
        annotation['skill_annotation'][0]['frame_duration'] = [575, 1443]
        with self.assertRaisesRegex(ValueError, 'outside'):
            validate_annotation(annotation)


class PhysicalDataTest(unittest.TestCase):
    def setUp(self):
        try:
            import av
            import numpy as np
            import pyarrow as pa
            import pyarrow.parquet as pq
        except ImportError:
            self.skipTest('physical audit requires the existing N1.7 environment')
        self.pa, self.pq = pa, pq
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = root = Path(self.temp.name)
        (root / 'meta/episodes/chunk-000').mkdir(parents=True)
        (root / 'data/chunk-000').mkdir(parents=True)
        features = {'observation.state': {'shape': [61]}, 'action': {'shape': [23]}}
        for key in CAMERAS.values():
            features[key] = {'shape': [16, 16, 3]}
            path = root / 'videos' / key / 'chunk-000/file-000.mp4'
            path.parent.mkdir(parents=True)
            with av.open(str(path), 'w') as container:
                stream = container.add_stream('mpeg4', rate=30)
                stream.width = stream.height = 16
                stream.pix_fmt = 'yuv420p'
                for _ in range(4):
                    for packet in stream.encode(av.VideoFrame.from_ndarray(np.zeros((16,16,3), dtype='uint8'), format='rgb24')):
                        container.mux(packet)
                for packet in stream.encode():
                    container.mux(packet)
        (root / 'meta/info.json').write_text(json.dumps(dict(codebase_version='v3.0', fps=30,
            features=features, data_path='data/chunk-{chunk_index:03d}/file-{file_index:03d}.parquet',
            video_path='videos/{video_key}/chunk-{chunk_index:03d}/file-{file_index:03d}.mp4')))
        tasks = [dict(task_index=t, task_name=n, task=n) for t,n in TASKS.items()]
        (root / 'meta/tasks.jsonl').write_text('\n'.join(map(json.dumps, tasks)))
        pq.write_table(pa.Table.from_pylist(tasks), root / 'meta/tasks.parquet')
        (root / 'meta/stats.json').write_text(json.dumps({'action': {'count': [999]}}))
        rows, physical = [], []
        for i,task in enumerate(TASKS):
            annotation = f'annotation-{i}.json'
            (root / annotation).write_text(json.dumps(dict(meta_data=dict(task_duration=3, valid_duration=[10,13]),
                skill_annotation=[dict(frame_duration=[11,13])])))
            row = dict(episode_index=i, task_index=task, length=2, task_instance_id=i, annotation_path=annotation)
            row.update({'data/chunk_index': 0, 'data/file_index': 0, 'dataset_from_index': i*2, 'dataset_to_index': i*2+2})
            for key in CAMERAS.values():
                row.update({f'videos/{key}/chunk_index': 0, f'videos/{key}/file_index': 0,
                    f'videos/{key}/from_timestamp': i*2/30, f'videos/{key}/to_timestamp': (i*2+2)/30})
            rows.append(row)
            physical.extend(dict(episode_index=i, task_index=task, frame_index=f, index=i*2+f,
                timestamp=f/30, **{'observation.state': [0.0]*61, 'action': [float(i)]*23}) for f in range(2))
        self.rows, self.physical = rows, physical
        self.data = root / 'data/chunk-000/file-000.parquet'
        self.metadata = root / 'meta/episodes/chunk-000/file-000.parquet'
        pq.write_table(pa.Table.from_pylist(rows), self.metadata)
        pq.write_table(pa.Table.from_pylist(physical), self.data)

    def test_aggregated_v3_file_and_unaligned_annotations(self):
        result = audit_2026(self.root)
        self.assertEqual((result['episodes'], result['rows'], result['rgb_files']), (2,4,3))
        self.assertEqual(result['annotation_length_mismatches'], 2)
        self.assertFalse(result['native_train_ready'])
        self.assertEqual(result['action_audit_statistics']['mean'], [0.5]*23)
        self.assertFalse((self.root / 'meta/modality.json').exists())

    def test_corrupt_indices_and_tensor_width_are_rejected(self):
        self.physical[0]['index'] = 100
        self.pq.write_table(self.pa.Table.from_pylist(self.physical), self.data)
        with self.assertRaisesRegex(ValueError, 'global index'):
            audit_2026(self.root)
        self.physical[0]['index'] = 0
        self.physical[0]['action'] = [0.0]*22
        self.pq.write_table(self.pa.Table.from_pylist(self.physical), self.data)
        with self.assertRaisesRegex(ValueError, 'tensor width'):
            audit_2026(self.root)

    def test_duplicate_episode_metadata_is_rejected(self):
        self.pq.write_table(self.pa.Table.from_pylist(self.rows + self.rows[:1]), self.metadata)
        with self.assertRaisesRegex(ValueError, 'duplicate'):
            audit_2026(self.root)


if __name__ == '__main__':
    unittest.main()
