import importlib.util
from pathlib import Path
import tempfile
import unittest
import hashlib
import sys

spec=importlib.util.spec_from_file_location('resource_downloads',Path(__file__).resolve().parents[1]/'evaluation/scripts/resource_downloads.py')
m=importlib.util.module_from_spec(spec)
spec.loader.exec_module(m)

class DownloadTests(unittest.TestCase):
    def recipe(self,root):
        return dict(id='test',host='4090',kind='weights',repo_type='model',repo='org/model',
                    revision='a'*40,root=str(root),target=str(root/'target'),python=sys.executable,
                    files=[dict(path='model.bin',bytes=3,sha256=hashlib.sha256(b'abc').hexdigest())])
    def test_manifest_and_hash(self):
        with tempfile.TemporaryDirectory() as d:
            root=Path(d);r=self.recipe(root)
            self.assertEqual(m.validate(r),3)
            file=root/'model.bin';file.write_bytes(b'abc');m.verify(file,r['files'][0])
            file.write_bytes(b'bad')
            with self.assertRaises(ValueError):m.verify(file,r['files'][0])
    def test_traversal(self):
        with tempfile.TemporaryDirectory() as d:
            r=self.recipe(Path(d));r['files'][0]['path']='../escape'
            with self.assertRaises(ValueError):m.validate(r)
    def test_existing_target_no_overwrite(self):
        with tempfile.TemporaryDirectory() as d:
            r=self.recipe(Path(d));Path(r['target']).mkdir()
            with self.assertRaises(ValueError):m.operation(r,'preview')
    def test_status_no_side_effect(self):
        with tempfile.TemporaryDirectory() as d:
            r=self.recipe(Path(d));self.assertEqual(m.operation(r,'status')['status'],'not_started')
            self.assertFalse((Path(d)/'.workbench-downloads').exists())

    def test_start_rejects_stale_preview(self):
        with tempfile.TemporaryDirectory() as d:
            r=self.recipe(Path(d))
            with self.assertRaisesRegex(ValueError,'预检'):
                m.operation(r,'start',expected_job_id='different')
            self.assertFalse((Path(d)/'.workbench-downloads').exists())

    def test_existing_status(self):
        with tempfile.TemporaryDirectory() as d:
            r=self.recipe(Path(d));Path(r['target']).mkdir()
            self.assertEqual(m.operation(r,'status')['status'],'existing')

    def test_file_directory_collision(self):
        with tempfile.TemporaryDirectory() as d:
            r=self.recipe(Path(d));r['files'].append(dict(r['files'][0],path='model.bin/other'))
            with self.assertRaisesRegex(ValueError,'冲突'):m.validate(r)

if __name__=='__main__':unittest.main()
