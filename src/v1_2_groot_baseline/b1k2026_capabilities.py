"""Evidence-backed capabilities for the existing B1K-2026 launch adapter."""
import hashlib
import json
from pathlib import Path
import subprocess

from .contracts import B1K2026, inspect_upstream


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
    return dict(protocol=B1K2026, task='turning_on_radio', launch_ready=ready,
                environment_present=paths_ready, reason=reason, capabilities=capabilities,
                verified_job_ids=jobs, task_performance_accepted=False,
                task_catalog_source=str(catalog_path), task_catalog=tasks,
                checkpoint_whitelist=[dict(path=str(config['checkpoint']), sha256=config['checkpoint_sha256'],
                    tasks=['turning_on_radio'], runtime_verified=ready)] if ready else [])
