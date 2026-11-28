"""Evidence-backed capabilities for the existing B1K-2026 launch adapter."""
import hashlib
import json
from pathlib import Path
import subprocess

from .contracts import (B1K2026, B1K2026_VERSION, B1K2026_DISPLAY_VERSION,
                        B1K2026_FAMILY, inspect_upstream)


def read(path):
    return json.loads(Path(path).read_text())


def verified_radio_jobs(repo, evaluation, config):
    """Match accepted native artifacts to the currently configured runtime.

    This rechecks source identity and the small result files. The full weight
    digest remains checked by the existing preview/start contract.
    """
    try:
        evidence = read(repo / 'docs/v1_2/evidence/native-acceptance-2026-11-28.json')
        if not evidence['native_runtime_accepted']:
            return []
        if inspect_upstream(config['upstream'])['revision'] != evidence['groot_revision']:
            return []
        behavior_revision = subprocess.check_output(
            ['git', '-C', str(config['behavior_root']), 'rev-parse', 'HEAD'], text=True, timeout=10).strip()
        changes = subprocess.check_output(
            ['git', '-C', str(config['behavior_root']), 'status', '--porcelain', '--untracked-files=no'],
            text=True, timeout=10).strip()
        if behavior_revision != evidence['behavior_revision'] or changes:
            return []
        jobs = []
        for accepted in evidence['runs']:
            output = (repo / accepted['output']).resolve()
            if not output.is_relative_to(evaluation.resolve() / 'V_1_2/official'):
                continue
            job, runner = read(output / 'job.json'), read(output / 'runner.json')
            policy = job['policy_identity']
            if (job['job_id'] != accepted['job_id'] or job['kind'] != B1K2026
                    or job['task_id'] != 'turning_on_radio'
                    or job['checkpoint_sha256'] != evidence['checkpoint_sha256']
                    or job['checkpoint_sha256'] != config['checkpoint_sha256']
                    or Path(policy['checkpoint']['path']).resolve() != config['checkpoint'].resolve()
                    or Path(policy['source']['root']).resolve() != config['upstream'].resolve()
                    or policy['argv'][0] != str(config['python'])
                    or job['evaluator']['argv'][0] != str(config['behavior_python'])
                    or Path(job['evaluator']['cwd']).resolve() != config['behavior_root'].resolve()
                    or runner['state'] != 'complete' or runner['evaluator_returncode'] != 0):
                continue
            results = list(output.glob('json/*.json'))
            if len(results) != 1 or not any(output.glob('videos/*.mp4')):
                continue
            if hashlib.sha256(results[0].read_bytes()).hexdigest() != accepted['native_json_sha256']:
                continue
            jobs.append(job['job_id'])
        return jobs
    except (OSError, ValueError, KeyError, TypeError, subprocess.SubprocessError):
        return []


def verified_engineering_run(repo, upstream, python):
    """Expose the accepted Task40 engineering chain only while its evidence matches.

    Small archived records are checked here; preview/start still hash the actual
    weights and full optimizer state before any new inference or recovery.
    """
    try:
        evidence = read(repo / 'docs/v1_2/evidence/training-acceptance-v031.json')
        if (evidence['accepted'] is not True or evidence['benchmark'] != 'b1k2026'
                or evidence['version'] != B1K2026_VERSION or evidence['tasks'] != [40]
                or evidence['optimizer_steps'] != 4
                or Path(evidence['upstream']).resolve() != Path(upstream).resolve()
                or evidence['python'] != str(python)):
            return None
        run = evidence['run_id']
        training = repo / 'training/b1k2026' / run
        output = (repo / evidence['evaluation_output']).resolve()
        if not output.is_relative_to((repo / 'evaluation/b1k2026/V_0_0/self_trained').resolve()):
            return None
        checkpoint = Path(evidence['checkpoint']).resolve()
        if not checkpoint.is_relative_to(training.resolve()):
            return None
        required = [training / name for name in ('job.json', 'status.json', 'metrics.jsonl',
                    'loss.jsonl', 'resume_events.jsonl')]
        required += [checkpoint / 'b1k2026_training.json', output / 'job.json',
                     output / 'runner.json', output / evidence['native_json']]
        hashes = evidence['files_sha256']
        for path in required:
            relative = str(path.relative_to(repo))
            if hashlib.sha256(path.read_bytes()).hexdigest() != hashes.get(relative):
                return None
        job, status = read(training / 'job.json'), read(training / 'status.json')
        identity, evaluation = read(checkpoint / 'b1k2026_training.json'), read(output / 'job.json')
        runner = read(output / 'runner.json')
        if (job['run_id'] != run or job['tasks'] != [40] or job['formal_training'] is not False
                or status['state'] != 'complete' or status['step'] != 4
                or identity['run_id'] != run or identity['optimizer_step'] != 4
                or identity['checkpoint_sha256'] != evidence['checkpoint_sha256']
                or identity['source_revision'] != inspect_upstream(upstream)['revision']
                or evaluation['job_id'] != evidence['evaluation_job_id']
                or evaluation['checkpoint_sha256'] != identity['checkpoint_sha256']
                or evaluation['task_id'] != 'make_microwave_popcorn'
                or evaluation['policy_identity']['argv'][0] != str(python)
                or runner['state'] != 'complete' or runner['evaluator_returncode'] != 0
                or not any(output.glob('videos/*.mp4'))):
            return None
        return evidence
    except (OSError, ValueError, KeyError, TypeError, subprocess.SubprocessError):
        return None


def options(repo, evaluation, config, paths_ready):
    jobs = verified_radio_jobs(repo, evaluation, config) if paths_ready else []
    ready = bool(jobs)
    reason = None if ready else '当前环境/源码/权重配置没有匹配的已验收原生回合；需重新验收'
    catalog_path = config['behavior_root'] / 'docs/challenge/task_data.json'
    try:
        catalog = read(catalog_path)['tasks']
        tasks = [dict(id=row['id'], name=row['name'], instruction=row.get('instruction'),
                      launch_ready=ready and row['id'] == 'turning_on_radio',
                      reason=reason if row['id'] == 'turning_on_radio' else '没有匹配且已闭环验收的任务权重')
                 for row in catalog]
    except (OSError, ValueError, KeyError, TypeError):
        tasks = []
    capabilities = {
        'official_inference': dict(available=ready, reason=reason),
        'official_evaluation': dict(available=ready, reason=reason),
        'training': dict(available=False, reason='尚无已锁定的训练身份、划分和冻结配置；真实前后向及保存重载未验收'),
        'self_trained_inference': dict(available=False, reason='尚无真实 B1K-2026 自训练 checkpoint 重载证据'),
        'self_trained_evaluation': dict(available=False, reason='尚无自训练权重原生闭环证据'),
        'strict_training_resume': dict(available=False, reason='optimizer、scheduler、RNG 和采样位置严格恢复未实跑'),
        'evaluation_resume': dict(available=False, reason='原生回合暂停恢复未验收'),
        'multi_environment': dict(available=False, reason='当前只验收单环境、单回合'),
        'live_camera': dict(available=False, reason='没有实时采集端验收；现有视频为回合录像'),
    }
    engineering = verified_engineering_run(repo, config['upstream'], config['python']) if ready else None
    whitelist = [dict(path=str(config['checkpoint']), sha256=config['checkpoint_sha256'],
                      tasks=['turning_on_radio'], model_role='official', runtime_verified=True)] if ready else []
    if engineering:
        scope = dict(available=True, engineering_only=True, tasks=['make_microwave_popcorn'],
                     run_id=engineering['run_id'], optimizer_steps=4,
                     evaluation_job_id=engineering['evaluation_job_id'],
                     reason='Task40 四步工程链路已实跑；正式训练未开放')
        for name in ('training', 'self_trained_inference', 'self_trained_evaluation', 'strict_training_resume'):
            capabilities[name] = dict(scope)
        whitelist.append(dict(path=engineering['checkpoint'], sha256=engineering['checkpoint_sha256'],
                              tasks=['make_microwave_popcorn'], model_role='self_trained',
                              engineering_only=True, runtime_verified=True))
        for task in tasks:
            if task['id'] == 'make_microwave_popcorn':
                task.update(launch_ready=True, model_roles=['self_trained'], engineering_only=True, reason=None)
    return dict(benchmark='b1k2026', version=B1K2026_VERSION,
                display_version=B1K2026_DISPLAY_VERSION, method_family=B1K2026_FAMILY,
                protocol=B1K2026, task='turning_on_radio', launch_ready=ready,
                environment_present=paths_ready, reason=reason, capabilities=capabilities,
                verified_job_ids=jobs, task_performance_accepted=False,
                task_catalog_source=str(catalog_path), task_catalog=tasks,
                formal_training_available=False, checkpoint_whitelist=whitelist)
