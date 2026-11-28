"""B1K-2026 local backend for the existing /api/training routes.

This exposes engineering submissions only. Formal training stays disabled
until a real end-to-end acceptance exists. Scientific values and identities
are explicit request fields; none are borrowed from the 2025 task recipes.
"""
import datetime
import hashlib
import io
import json
import os
from pathlib import Path
import subprocess
import sys
import threading
import zipfile

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))
from run_identity import validate_run_identity, MODEL_VERSIONS
from v1_2_groot_baseline.contracts import (inspect_upstream, inspect_checkpoint, checkpoint_digest,
                                         inspect_training_checkpoint, B1K2026_VERSION,
                                         B1K2026_DISPLAY_VERSION, B1K2026_FAMILY)
from v1_2_groot_baseline.b1k2026_training_data import digest
from v1_2_groot_baseline.b1k2026_capabilities import verified_engineering_run

FINETUNE_REQUIRED = {'tune_llm', 'tune_visual', 'tune_projector', 'tune_diffusion_model',
    'state_dropout_prob', 'global_batch_size', 'gradient_accumulation_steps', 'dataloader_num_workers',
    'learning_rate', 'weight_decay', 'warmup_ratio', 'max_steps', 'save_steps',
    'shard_size', 'episode_sampling_rate', 'num_shards_per_epoch'}
ADAPTER_FILES = ('scripts/train_v12_b1k2026.py',
    'src/v1_2_groot_baseline/b1k2026_resume.py',
    'src/v1_2_groot_baseline/b1k2026_training_data.py',
    'src/v1_2_groot_baseline/contracts.py')


def atomic(path, value):
    temp = path.with_suffix('.pending')
    temp.write_text(json.dumps(value, ensure_ascii=False, indent=2))
    temp.replace(path)


def gpu_info():
    output = subprocess.check_output(['nvidia-smi', '--query-gpu=index,uuid,memory.total,memory.used',
                                     '--format=csv,noheader,nounits'], text=True, timeout=10)
    return [dict(index=int(fields[0]), uuid=fields[1].strip(), total_mb=int(fields[2]), used_mb=int(fields[3]))
            for fields in (line.split(',') for line in output.strip().splitlines())]


def process_token(pid):
    try:
        # comm may contain whitespace or parentheses; fields after ')' are stable.
        fields = (Path('/proc') / str(pid) / 'stat').read_text().rsplit(')', 1)[1].split()
        return fields[19] if fields[0] != 'Z' else None
    except (OSError, IndexError):
        return None


class V12TrainingManager:
    def __init__(self, root):
        self.root = Path(root).resolve()
        self.device_id, self.host, self.label = 'local', '127.0.0.1', '4090 · B1K-2026'
        self.training = self.root / 'training/b1k2026'
        self.python = Path(os.environ.get('V12_GROOT_PYTHON', '/nvme3/projects/GR00T/Isaac-GR00T-n1d7-ace36d9/.venv/bin/python'))
        self.upstream = Path(os.environ.get('V12_GROOT_ROOT', '/nvme3/projects/GR00T/Isaac-GR00T-n1d7-ace36d9'))
        self.lock = threading.Lock()
        self.processes = {}

    def options(self):
        verified = verified_engineering_run(self.root, self.upstream, self.python)
        return dict(device_id='local', device_label=self.label, benchmark='b1k2026',
            model_versions={}, registered_model_versions={B1K2026_VERSION: MODEL_VERSIONS[B1K2026_VERSION]},
            version=B1K2026_VERSION, display_version=B1K2026_DISPLAY_VERSION, method_family=B1K2026_FAMILY,
            formal_training_available=False, engineering_training_verified=bool(verified),
            verified_engineering_run=verified['run_id'] if verified else None,
            reason=('Task40 四步工程训练、严格恢复及自训练闭环已验收；正式训练未开放' if verified else
                    '工程训练、保存重载、严格恢复及自训练闭环尚待真实验收；仅接受显式 engineering 请求'),
            python_path=str(self.python), upstream=str(self.upstream), gpus=gpu_info(),
            datasets=[dict(path='/nvme3/datasets/b1k2026-task0-task40', tasks=[0,40],
                           ready=False, reason='需显式 split、语言选择和独立 train-only statistics')])

    def sync_state(self):
        return dict(state='local', enabled=False, host=self.host, live_source=str(self.root))

    def prepare(self, request):
        allowed = {'version','run_id','initiator','notes','engineering_only','base_checkpoint','base_sha256',
                   'data_manifest','finetune','gpu_index','seed','gradient_checkpointing','minimum_free_mib',
                   'pause_after_steps','benchmark'}
        if not isinstance(request, dict) or set(request) - allowed:
            raise ValueError('unsupported B1K-2026 training fields')
        if (allowed - {'pause_after_steps','benchmark'}) - request.keys():
            raise ValueError('explicit B1K-2026 training fields required: ' +
                             ','.join(sorted((allowed - {'pause_after_steps','benchmark'}) - request.keys())))
        identity = validate_run_identity(request.get('run_id'), request.get('version'))
        if identity['model_version'] != B1K2026_VERSION or request.get('engineering_only') is not True:
            raise ValueError('only B1K-2026 V0_0 engineering training is enabled; no formal training')
        if request.get('benchmark', 'b1k2026') != 'b1k2026':
            raise ValueError('V0_0 belongs exclusively to benchmark b1k2026')
        request = dict(request, version=B1K2026_VERSION, benchmark='b1k2026')
        if request.get('initiator') not in ('agent', 'user') or not isinstance(request.get('notes'), str) or not request['notes'].strip():
            raise ValueError('explicit initiator and concise version notes required')
        if not 1 <= len(request['notes']) <= 200:
            raise ValueError('notes length must be 1–200')
        experiment = self.root / f"Output/test_{identity['test_id'][4:]}/experiment_{identity['experiment_id'][3:]}"
        if not experiment.is_dir() or not (experiment / 'task.md').is_file():
            raise ValueError('explicit experiment task/plan must exist before training')
        job = self.training / request['run_id']
        if job.exists():
            raise ValueError('run already exists; resume the original job')
        ft = request.get('finetune')
        if not isinstance(ft, dict) or set(ft) != FINETUNE_REQUIRED:
            raise ValueError('all native finetune fields must be explicit: ' + ','.join(sorted(FINETUNE_REQUIRED)))
        for key in ('tune_llm','tune_visual','tune_projector','tune_diffusion_model'):
            if type(ft[key]) is not bool:
                raise ValueError('explicit boolean required: ' + key)
        for key in ('global_batch_size','gradient_accumulation_steps','max_steps','save_steps','shard_size','num_shards_per_epoch'):
            if type(ft[key]) is not int or ft[key] < 1:
                raise ValueError('positive integer required: ' + key)
        if ft['max_steps'] > 100 or ft['save_steps'] > ft['max_steps']:
            raise ValueError('engineering budget is at most 100 optimizer steps; save within budget')
        if ft['dataloader_num_workers'] != 0:
            raise ValueError('strict cursor currently requires dataloader_num_workers=0')
        import math
        for key in ('learning_rate','weight_decay','warmup_ratio','state_dropout_prob','episode_sampling_rate'):
            if type(ft[key]) not in (int,float) or not math.isfinite(ft[key]):
                raise ValueError('finite numeric value required: ' + key)
        if not (ft['learning_rate'] > 0 and ft['weight_decay'] >= 0 and 0 <= ft['warmup_ratio'] < 1
                and 0 <= ft['state_dropout_prob'] <= 1 and 0 < ft['episode_sampling_rate'] <= 1):
            raise ValueError('native optimizer/data sampling values out of range')
        for key in ('gpu_index','seed','minimum_free_mib'):
            if type(request.get(key)) is not int or request[key] < 0:
                raise ValueError('explicit nonnegative integer required: ' + key)
        if type(request.get('gradient_checkpointing')) is not bool:
            raise ValueError('explicit gradient_checkpointing required')
        if request.get('pause_after_steps') is not None and (type(request['pause_after_steps']) is not int
                or not 0 < request['pause_after_steps'] < ft['max_steps']):
            raise ValueError('pause_after_steps must be inside the engineering budget')
        if any(row.get('gpu_index') == request['gpu_index'] and row['state'] in ('running','starting') for row in self.snapshot()['jobs']):
            raise ValueError('selected GPU already has a live B1K-2026 training job')
        gpu = next((g for g in gpu_info() if g['index'] == request['gpu_index']), None)
        if gpu is None or gpu['total_mb'] - gpu['used_mb'] < request['minimum_free_mib']:
            raise ValueError('selected GPU has insufficient currently free memory; existing jobs retained')
        source = inspect_upstream(self.upstream)
        base = Path(request['base_checkpoint']).resolve(strict=True)
        if base != Path('/nvme3/models/GR00T-N1.7-3B').resolve():
            raise ValueError('V1.2 engineering training must start from the verified generic N1.7-3B')
        inspect_checkpoint(base, role='generic')
        if checkpoint_digest(base) != request['base_sha256']:
            raise ValueError('generic base checkpoint hash mismatch')
        manifest_path = Path(request['data_manifest']).resolve(strict=True)
        manifest = json.loads(manifest_path.read_text())
        if manifest.get('benchmark') != 'b1k2026' or manifest.get('schema') != 'native-v3-split-v1':
            raise ValueError('native v3 training split manifest required')
        if manifest['upstream_revision'] != source['revision']:
            raise ValueError('data preparation source revision mismatch')
        if not manifest['views']['train']['tasks'] or not set(manifest['views']['train']['tasks']) <= {0,40}:
            raise ValueError('only scoped Task0/40 engineering data supported')
        subprocess.check_call([str(self.python), '-c',
            'import sys;from v1_2_groot_baseline.b1k2026_training_data import verify_manifest;verify_manifest(sys.argv[1])',
            str(manifest_path)], env={**os.environ,'PYTHONPATH':str(self.root / 'src')}, timeout=120)
        recipe_hash = hashlib.sha256(json.dumps(request, sort_keys=True).encode()).hexdigest()
        return dict(version=B1K2026_VERSION, display_version=B1K2026_DISPLAY_VERSION,
            method_family=B1K2026_FAMILY, benchmark='b1k2026', run_id=request['run_id'], request=request,
            output=str(job), python=str(self.python), upstream=str(self.upstream), source_revision=source['revision'],
            gpu=gpu, data_manifest_sha256=digest(manifest_path), recipe_sha256=recipe_hash,
            tasks=manifest['views']['train']['tasks'], optimizer_steps=ft['max_steps'],
            adapter_source_sha256={name:digest(self.root / name) for name in ADAPTER_FILES},
            batch=dict(world_size=1, microbatch=ft['global_batch_size'], accumulation=ft['gradient_accumulation_steps'],
                       effective=ft['global_batch_size'] * ft['gradient_accumulation_steps']),
            training_verified=False, formal_training=False)

    def _spawn(self, job, plan, resume_checkpoint=None):
        env = os.environ.copy()
        env.update(CUDA_VISIBLE_DEVICES=str(plan['request']['gpu_index']), PYTHONPATH=str(self.root / 'src') + os.pathsep + str(self.upstream),
                   PYTHONNOUSERSITE='1', HF_HUB_CACHE='/nvme3/models/.hf-cache/hub',
                   HF_HUB_OFFLINE='1', TRANSFORMERS_OFFLINE='1', GROOT_HF_LOCAL_FIRST='1',
                   GROOT_PATCH_MISTRAL='1', WANDB_MODE='disabled')
        argv = [str(self.python), '-u', str(self.root / 'scripts/train_v12_b1k2026.py'), '--job', str(job)]
        if resume_checkpoint:
            argv += ['--resume-checkpoint', str(resume_checkpoint)]
        atomic(job / 'status.json', dict(state='starting', step=0 if not resume_checkpoint else
            json.loads((resume_checkpoint / 'trainer_state.json').read_text())['global_step']))
        with (job / 'train.log').open('ab') as log:
            process = subprocess.Popen(argv, cwd=self.root, env=env, stdin=subprocess.DEVNULL,
                stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
        self.processes[plan['run_id']] = process
        atomic(job / 'worker.json', dict(pid=process.pid, process_start=process_token(process.pid)))
        return dict(job_id=plan['run_id'], run_id=plan['run_id'], pid=process.pid, state='starting', output=str(job), notes=plan['request']['notes'])

    def start(self, request):
        with self.lock:
            plan = self.prepare(request)
            job = Path(plan['output'])
            job.mkdir(parents=True)
            atomic(job / 'job.json', plan)
            return self._spawn(job, plan)

    def snapshot(self):
        jobs = []
        for marker in sorted(self.training.glob('*/job.json')):
            plan = json.loads(marker.read_text())
            state = json.loads((marker.parent / 'status.json').read_text()) if (marker.parent / 'status.json').exists() else dict(state='failed', error='missing worker state')
            process = self.processes.get(plan['run_id'])
            worker = json.loads((marker.parent / 'worker.json').read_text()) if (marker.parent / 'worker.json').exists() else state
            live = (process.poll() is None if process is not None else
                    bool(worker.get('process_start')) and process_token(worker.get('pid')) == worker['process_start'])
            if not live and state['state'] in ('starting','running'):
                state = dict(state, state='failed', error='worker exited before final state')
            jobs.append(dict(id=plan['run_id'], run=plan['run_id'], benchmark='b1k2026',
                version=plan.get('version', 'V_1_2'), display_version=plan.get('display_version'),
                method_family=plan.get('method_family'), arm='baseline',
                output=str(marker.parent), gpu_index=plan['request']['gpu_index'], notes=plan['request']['notes'],
                worker_alive=live, **state))
        return dict(jobs=jobs)

    def _job(self, request):
        run = request.get('run', request.get('run_id'))
        validate_run_identity(run, B1K2026_VERSION)
        root = self.training / run
        plan = json.loads((root / 'job.json').read_text())
        if plan['run_id'] != run:
            raise ValueError('training identity mismatch')
        return root, plan

    def detail(self, request):
        job, plan = self._job(request)
        state = next(row for row in self.snapshot()['jobs'] if row['run'] == plan['run_id'])
        losses = {row['step']:row for row in map(json.loads, (job / 'loss.jsonl').read_text().splitlines())} if (job / 'loss.jsonl').is_file() else {}
        metrics = [dict(row, loss=losses.get(row['step'],{}).get('loss')) for row in
                   map(json.loads,(job / 'metrics.jsonl').read_text().splitlines())] if (job / 'metrics.jsonl').is_file() else []
        return dict(job=state, configuration=plan, metrics=metrics,
            log_tail=(job / 'train.log').read_text(errors='replace')[-16000:] if (job / 'train.log').is_file() else '',
            checkpoint_index=json.loads((job / 'checkpoint_index.json').read_text()) if (job / 'checkpoint_index.json').is_file() else None)

    def evidence(self, request):
        job, plan = self._job(request)
        stream = io.BytesIO()
        with zipfile.ZipFile(stream,'w',zipfile.ZIP_DEFLATED) as archive:
            for name in ('job.json','status.json','effective_config.yaml','trainable_parameters.json',
                         'metrics.jsonl','loss.jsonl','checkpoint_index.json','training_report.md','resume_events.jsonl',
                         'runtime_environment.json','retry_events.jsonl','resume_attempts.jsonl','train.log'):
                if (job / name).is_file():
                    archive.write(job / name,name)
        return plan['run_id']+'-evidence.zip', stream.getvalue()

    def stop(self, request):
        with self.lock:
            job, plan = self._job(request)
            if next(row for row in self.snapshot()['jobs'] if row['run'] == plan['run_id'])['state'] not in ('running','starting'):
                raise ValueError('only a live job can pause')
            atomic(job / 'pause.request', dict(requested_at=datetime.datetime.now().isoformat()))
            return dict(run=plan['run_id'], state='pause_requested', notes=plan['request']['notes'])

    def resume(self, request):
        with self.lock:
            job, plan = self._job(request)
            status = next(row for row in self.snapshot()['jobs'] if row['run'] == plan['run_id'])
            if request.get('retry_failed') is True:
                return self._retry_failed(job, plan, status, request)
            if status['state'] not in ('paused','failed') or status['worker_alive']:
                raise ValueError('resume requires an exited worker and complete optimizer-boundary checkpoint')
            checkpoint = Path(json.loads((job / 'checkpoint_index.json').read_text())['checkpoint']).resolve(strict=True)
            identity = inspect_training_checkpoint(checkpoint)
            steps = [json.loads(line)['step'] for line in (job / 'metrics.jsonl').read_text().splitlines()]
            if not steps or max(steps) != identity['optimizer_step']:
                raise ValueError('training updates beyond published checkpoint; retain failed job for review')
            if (identity['recipe_sha256'] != plan['recipe_sha256'] or identity['run_id'] != plan['run_id']
                    or identity['data_manifest_sha256'] != plan['data_manifest_sha256']):
                raise ValueError('resume configuration changed')
            current = {name:digest(self.root / name) for name in ADAPTER_FILES}
            changed = current != plan['adapter_source_sha256']
            if changed or status['state'] == 'failed':
                if (request.get('expected_adapter_source_sha256') != current
                        or request.get('expected_checkpoint') != str(checkpoint)
                        or not isinstance(request.get('reason'),str) or not request['reason'].strip()):
                    raise ValueError('engineering recovery requires explicit checkpoint, source hashes and reason')
            gpu = next(g for g in gpu_info() if g['index'] == plan['request']['gpu_index'])
            if gpu['total_mb'] - gpu['used_mb'] < plan['request']['minimum_free_mib']:
                raise ValueError('resume GPU capacity unavailable; retain paused job')
            if request.get('preview') is True:
                return dict(state='preview',run=plan['run_id'],checkpoint=str(checkpoint),
                            optimizer_step=identity['optimizer_step'],notes=plan['request']['notes'],
                            adapter_source_sha256=current,engineering_revision=changed)
            if request.get('preview', False) is not False:
                raise ValueError('preview must be boolean')
            with (job / 'resume_attempts.jsonl').open('a') as stream:
                stream.write(json.dumps(dict(recorded_at=datetime.datetime.now().isoformat(),
                    previous_state=status,checkpoint=str(checkpoint),optimizer_step=identity['optimizer_step'],
                    training_state_sha256=identity['training_state_sha256'],
                    reason=request.get('reason','strict checkpoint resume'),
                    previous_adapter_source_sha256=plan['adapter_source_sha256'],adapter_source_sha256=current))+'\n')
            if changed:
                plan = dict(plan,adapter_source_sha256=current)
                atomic(job / 'job.json',plan)
            if (job / 'pause.request').exists():
                (job / 'pause.request').rename(job / f'pause-event-{datetime.datetime.now().strftime("%Y%m%d%H%M%S%f")}.json')
            return self._spawn(job, plan, checkpoint)

    def _retry_failed(self, job, plan, status, request):
        """Retry a failed initialization, preserving its run and original error.

        A run with any training evidence must use its full checkpoint instead.
        Engineering source changes are pinned in the preview and archived.
        """
        if status['state'] != 'failed' or status['worker_alive']:
            raise ValueError('initialization retry requires a failed, exited worker')
        if (status.get('step', 0) != 0 or any((job / name).exists() for name in
                ('metrics.jsonl', 'loss.jsonl', 'checkpoint_index.json'))
                or any((job / 'native').glob('checkpoint-*'))):
            raise ValueError('run has training state; use its complete checkpoint, not initialization retry')
        if not isinstance(request.get('reason'), str) or not request['reason'].strip():
            raise ValueError('initialization retry requires an engineering reason')
        current = {name:digest(self.root / name) for name in ADAPTER_FILES}
        if request.get('expected_adapter_source_sha256') != current:
            raise ValueError('retry adapter source differs from the previewed engineering revision')
        gpu = next(g for g in gpu_info() if g['index'] == plan['request']['gpu_index'])
        if gpu['total_mb'] - gpu['used_mb'] < plan['request']['minimum_free_mib']:
            raise ValueError('retry GPU capacity unavailable; retain failed job')
        if request.get('preview') is True:
            return dict(state='preview', action='retry_failed', run=plan['run_id'],
                optimizer_step=0, previous_error=status.get('error'), gpu=gpu,
                adapter_source_sha256=current, notes=plan['request']['notes'])
        if request.get('preview') is not False:
            raise ValueError('retry requires explicit boolean preview')
        record = job / 'retry_events.jsonl'
        attempt = len(record.read_text().splitlines()) + 2 if record.exists() else 2
        with record.open('a') as stream:
            stream.write(json.dumps(dict(attempt=attempt, reason=request['reason'],
                previous_state=status, previous_adapter_source_sha256=plan['adapter_source_sha256'],
                adapter_source_sha256=current, run_id=plan['run_id'],
                recipe_sha256=plan['recipe_sha256'], recorded_at=datetime.datetime.now().isoformat())) + '\n')
        plan = dict(plan, adapter_source_sha256=current)
        atomic(job / 'job.json', plan)
        return dict(self._spawn(job, plan), attempt=attempt)

    def prepare_dataset(self, request):
        with self.lock:
            return self._prepare_dataset(request)

    def _prepare_dataset(self, request):
        allowed = {'source','destination','split','language_mode','preview'}
        if set(request) - allowed or not allowed <= request.keys():
            raise ValueError('explicit source/destination/split/language_mode/preview required')
        if request['language_mode'] not in ('native_tasks','task_description'):
            raise ValueError('explicit native_tasks or task_description language_mode required')
        destination = Path(request['destination']).resolve(strict=True)
        if not destination.is_relative_to(Path('/nvme3/datasets')) or any(destination.iterdir()):
            raise ValueError('allocate an empty /nvme3/datasets resource using the storage API')
        # Parse the physical metadata in the installed N1.7 Python, not the web interpreter.
        code = ('import json,sys;from v1_2_groot_baseline.b1k2026_training_data import validate_split;'
                'r=json.loads(sys.argv[1]);s=validate_split(r["source"],r["split"]);'
                'print(json.dumps({k:len(v) for k,v in s.items()}))')
        preview = subprocess.check_output([str(self.python), '-c', code, json.dumps(request)], text=True,
            env={**os.environ, 'PYTHONPATH': str(self.root / 'src')}, timeout=30)
        if request['preview'] is True:
            return dict(state='preview', episodes=json.loads(preview), destination=str(destination), language_mode=request['language_mode'])
        if request['preview'] is not False:
            raise ValueError('preview must be a boolean')
        work = self.root / 'outputs/shared/training_dashboard/b1k2026_datasets'
        work.mkdir(parents=True, exist_ok=True)
        name = hashlib.sha256(str(destination).encode()).hexdigest()[:20]
        status = work / (name + '.json')
        if status.exists() and json.loads(status.read_text())['state'] != 'failed':
            raise ValueError('dataset preparation already exists; query its status file')
        payload = dict(request, upstream=str(self.upstream))
        atomic(status, dict(state='starting', destination=str(destination)))
        with (work / (name + '.log')).open('ab') as log:
            process = subprocess.Popen([str(self.python), '-u', str(self.root / 'scripts/prepare_v12_b1k2026.py'),
                '--request', json.dumps(payload), '--status', str(status)], stdin=subprocess.DEVNULL,
                stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
        atomic(work / (name + '.worker.json'), dict(pid=process.pid, process_start=process_token(process.pid)))
        return dict(state='starting', pid=process.pid, status=str(status), log=str(work / (name + '.log')))

    def delete(self, request):
        raise ValueError('preserve B1K-2026 engineering jobs and checkpoints during acceptance')

    delete_checkpoint = delete
    cancel = stop
    pause = stop

    def sync(self, request):
        raise ValueError('local B1K-2026 uses the pinned installed environment; remote sync is unsupported')

    def close(self):
        # Workbench reload must leave detached training workers running.
        pass
