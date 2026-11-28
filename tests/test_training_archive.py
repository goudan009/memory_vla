import importlib.util
import json
from pathlib import Path
import subprocess
import tempfile
import unittest
from unittest.mock import patch


def module(name):
    spec = importlib.util.spec_from_file_location(name, Path(__file__).resolve().parents[1]/'scripts'/f'{name}.py')
    value = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(value)
    return value


archive = module('training_archive')
remote = module('training_archive_remote')


class ArchiveTests(unittest.TestCase):
    def test_statistics_keep_missing_gradients_and_resumed_duplicates(self):
        rows = [dict(kind='train', step=1, loss=4, component_grad_norms={'memory':0}),
                dict(kind='train', step=2, loss=2),
                dict(kind='train', step=2, loss=1, component_grad_norms={'memory':3}),
                dict(kind='eval', step=2, loss=99)]
        stats = archive.statistics(b''.join(archive.encoded(r).replace(b'\n',b'')+b'\n' for r in rows)+b'{', {})
        self.assertEqual(stats['records'], 3)
        self.assertEqual(stats['duplicate_steps'], 1)
        self.assertEqual(stats['incomplete_tail_records'], 1)
        self.assertEqual(stats['fields']['component_grad_norms.memory']['missing_or_nonfinite'], 1)
        self.assertEqual(stats['fields']['component_grad_norms.memory']['zero'], 1)
        self.assertAlmostEqual(stats['loss_last100'], 7/3)

    def test_v21_objective_uses_recorded_weight(self):
        data = b'{"step":1,"fm":2,"state":3,"lr":0.001,"grad_norm":0.5}\n'
        self.assertEqual(archive.statistics(data, {'state_weight':0.25})['loss_last100'], 2.75)
        self.assertIsNone(archive.statistics(data, {})['loss_last100'])

    def test_native_report_loss_and_part_metrics(self):
        data = b'{"step":10,"fm":0.2,"state":0,"lr":0.0001,"part_loss":{"left_arm":0.3}}\n'
        stats = archive.statistics(data, {'recipe':'native_random_v2'})
        self.assertEqual(stats['loss_last100'],0.2)
        self.assertEqual(stats['fields']['part_loss.left_arm']['mean'],0.3)
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp)
            job=dict(run='run1',arm='baseline',status='completed',step=10)
            detail=dict(training_parameters=dict(values={'recipe':'native_random_v2'}),validation=[{'step':10,'fm':0.4,'samples':100}])
            with patch.object(archive,'remote_snapshot',return_value=({'metrics.jsonl':data},{'metrics.jsonl':{'sha256':archive.sha(data)}})):
                archive.archive(root,'a100',{},job,detail)
            archive.render(root)
            report=(root/archive.REPORT/'runs/a100/run1/baseline/training_report.md').read_text()
            self.assertIn('0.4 | 100',report)
            self.assertIn('part_loss.left_arm',report)
            self.assertIn('0.2',report)
            self.assertIn('尚未归档',report)

    def test_malformed_interior_record_is_an_error(self):
        with self.assertRaisesRegex(ValueError, 'line 1'):
            archive.statistics(b'{\n{"step":1,"loss":1}\n', {})

    def test_collector_excludes_tensors_and_records_log_tail(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            (root/'memory_v1/metrics').mkdir(parents=True)
            (root/'logs').mkdir()
            (root/'memory_v1/metrics/metrics.jsonl').write_bytes(b'{"step":1,"loss":2}\n')
            (root/'logs/controller.log').write_bytes(b'x'*300000)
            (root/'model.safetensors').write_bytes(b'never export')
            value = remote.collect(dict(output=str(root), arm='memory_v1'))
            self.assertIn('metrics.jsonl', value['files'])
            self.assertFalse(any('safetensors' in key for key in value['files']))
            origin = value['origins']['recorded/logs/controller.log']
            self.assertEqual(origin['source_bytes'], 300000)
            self.assertEqual(origin['bytes'], 256*1024)
            self.assertEqual(origin['offset'], 300000-256*1024)

    def test_archive_roundtrip_and_content_addressed_metrics(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            metric = b'{"kind":"train","step":1,"loss":2}\n'
            job = dict(run='v1__test01__exp13__run001', arm='memory_v1', status='stopped', step=1)
            detail = dict(training_parameters=dict(values={'seed':42}, sources=['actual.json']))
            origin = dict(path='/recorded/metrics.jsonl', sha256=archive.sha(metric))
            with patch.object(archive, 'remote_snapshot', side_effect=lambda *_:({'metrics.jsonl':metric},{'metrics.jsonl':origin})):
                archive.archive(root, 'a100', {}, job, detail)
                job['status'] = 'completed'
                archive.archive(root, 'a100', {}, job, detail)
            manifests = list((root/archive.REPORT).rglob('MANIFEST.json'))
            self.assertEqual(len(manifests), 2)
            self.assertEqual(len(list((root/archive.REPORT).rglob('raw_metrics/*.jsonl'))), 1)
            for path in manifests:
                value = archive.load(path)
                run_dir = path.parents[2]
                original = b''.join((run_dir/p['path']).read_bytes() for p in value['metrics']['parts'])
                self.assertEqual(original, metric)
                self.assertEqual(archive.sha(original), value['metrics']['sha256'])
            archive.verify(root)
            archive.render(root)
            self.assertIn('completed', (root/archive.REPORT/'README.md').read_text())

    def test_scan_deduplicates_and_retries_a_failed_export(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            (root/'configs').mkdir()
            (root/'configs/workbench_devices.json').write_text('{"a100":{}}')
            args = type('Args', (), dict(root=root, devices=['a100'], backfill=['run1'], publish=False, workbench='unused'))()
            job = dict(run='run1', arm='baseline', status='completed', step=4)
            def api(*_, **kw):
                return {'jobs':[job]} if 'run' not in kw else {}
            real_export = archive.archive
            with patch.object(archive, 'api', side_effect=api), patch.object(archive, 'remote_snapshot', return_value=({},{})), patch.object(archive, 'archive') as export:
                export.side_effect = RuntimeError('connection lost')
                self.assertEqual(archive.scan(args), 1)
                export.side_effect = real_export
                self.assertEqual(archive.scan(args), 0)
                self.assertEqual(archive.scan(args), 0)
                self.assertEqual(export.call_count, 2)

    def test_evaluations_require_run_and_checkpoint_and_terminal_evidence(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            directory = root/archive.REPORT/'runs/a100/run1/baseline'
            archive.write(directory/'latest.json', archive.encoded(dict(run='run1')))
            result = root/'evaluation/V_1/memory/eval1/results.json'
            value = dict(status='running', policy_identity=dict(training_id='run1', checkpoint='/exact/step1000'))
            archive.write(result, archive.encoded(value))
            archive.evaluations(root)
            self.assertFalse((directory/'evaluations').exists())
            archive.evaluations(root, [dict(output=str(result.parent), state='failed', id='real-job')])
            copies = list((directory/'evaluations').glob('*/results.json'))
            self.assertEqual(len(copies), 1)
            self.assertEqual(copies[0].read_bytes(), result.read_bytes())
            self.assertEqual(archive.load(copies[0].parent/'origin.json')['workbench_state'], 'failed')
            archive.verify(root)

    def test_offline_secondary_device_does_not_fail_successful_scan(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            args = type('Args', (), dict(root=root, devices=['a100','a6000'], backfill=[], publish=True, workbench='unused'))()
            def api(*_, **kw):
                if kw.get('device') == 'a6000':
                    raise RuntimeError('SSH unavailable')
                return {'jobs':[]}
            with patch.object(archive,'api',side_effect=api), patch.object(archive,'publish') as publish:
                self.assertEqual(archive.scan(args),0)
                publish.assert_called_once_with(root)
            state = archive.load(root/archive.STATE)
            self.assertEqual(state['scanned_devices'], ['a100'])
            self.assertEqual(state['status'], 'degraded')
            self.assertIn('SSH unavailable', state['errors']['a6000'])

    def test_interrupted_evaluation_without_training_archive_is_reported(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            result = root/'evaluation/V_2_1/baseline/task40_eval01/results.json'
            value = dict(status='interrupted', completed_episodes=10, requested_episodes=50,
                         successful_episodes=0, official_episodes=10, mean_task_progress=45,
                         policy_identity=dict(training_id='historical-stage1', checkpoint='/exact/step6000'))
            archive.write(result, archive.encoded(value))
            archive.write(result.parent/'episodes.jsonl', b'{"success":false}\n')
            archive.evaluations(root)
            copies = list((root/archive.REPORT/'evaluations').glob('*/results.json'))
            self.assertEqual(len(copies),1)
            self.assertEqual(copies[0].read_bytes(),result.read_bytes())
            archive.render(root)
            report = (root/archive.REPORT/'README.md').read_text()
            self.assertIn('historical-stage1',report)
            self.assertIn('interrupted',report)
            self.assertIn('10 / 50',report)
            first = {str(p):p.read_bytes() for p in (root/archive.REPORT).rglob('*') if p.is_file()}
            archive.evaluations(root)
            archive.render(root)
            self.assertEqual(first,{str(p):p.read_bytes() for p in (root/archive.REPORT).rglob('*') if p.is_file()})
            archive.verify(root)
            copies[0].write_bytes(b'changed')
            with self.assertRaisesRegex(ValueError,'Evaluation checksum'):
                archive.verify(root)

    def test_running_evaluation_waits_and_missing_training_identity_is_explicit(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            result = root/'evaluation/V_1/official/task40_eval01/results.json'
            archive.write(result,archive.encoded(dict(status='interrupted')))
            archive.evaluations(root,[dict(output=str(result.parent),state='running')])
            self.assertFalse(list((root/archive.REPORT).rglob('results.json')))
            archive.evaluations(root,[dict(output=str(result.parent),state='failed',id='eval-job')])
            archive.render(root)
            report = (root/archive.REPORT/'README.md').read_text()
            self.assertIn('eval-job',report)
            self.assertIn('未记录',report)

    def test_git_publishing_preserves_unrelated_staged_and_unstaged_edits(self):
        with tempfile.TemporaryDirectory() as temp, tempfile.TemporaryDirectory() as origin:
            root = Path(temp)
            subprocess.run(['git','init','--bare',origin], check=True, capture_output=True)
            def git(*args):
                return archive.git(root,*args)
            git('init','-b','main')
            git('config','user.email','test@example.com')
            git('config','user.name','archive-test')
            git('remote','add','origin',origin)
            (root/'unrelated.txt').write_text('initial\n')
            git('add','unrelated.txt')
            git('commit','-m','initial')
            (root/'unrelated.txt').write_text('staged\n')
            git('add','unrelated.txt')
            (root/'unrelated.txt').write_text('unstaged\n')
            archive.write(root/archive.REPORT/'README.md', b'automatic\n')
            archive.publish(root)
            self.assertEqual(git('show','HEAD:unrelated.txt'), 'initial')
            self.assertEqual(git('show',':unrelated.txt'), 'staged')
            self.assertEqual((root/'unrelated.txt').read_text(), 'unstaged\n')
            first = git('rev-parse','HEAD')
            archive.publish(root)
            self.assertEqual(git('rev-parse','HEAD'), first)


if __name__ == '__main__':
    unittest.main()
