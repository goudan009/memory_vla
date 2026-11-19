#!/usr/bin/env python3
"""Local dashboard for launching GR00T evaluations and viewing results."""

from __future__ import annotations

import argparse
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
import os
from pathlib import Path
import re
import subprocess
import sys
import threading
import time
from urllib.parse import parse_qs, urlparse
import webbrowser

sys.path.insert(0, str(Path(__file__).resolve().parent))
from groot_live_preview import read_preview, subscribe
from groot_web_launch import LaunchManager
from groot_inference_launch import InferenceManager
import vlm_diagnosis
import experiment_registry
import b1k_metrics
import task_stage_diagnostics
import vlm_stage_diagnosis
import human_review


EVALUATION_ROOT = Path(__file__).resolve().parents[1]
REGISTRY_ANNOTATIONS = EVALUATION_ROOT.parent / "outputs" / "registry_annotations.json"
REGISTRY_ANNOTATIONS_LOCK = threading.Lock()
TASK40_VIDEOS = Path("/nvme3/projects/memory_vla/outputs/overnight_memory_task40_20261114/verification/task40_vlm_blind_review/full_episodes")
TASK40_ANNOTATIONS_FILE = EVALUATION_ROOT.parent / "outputs" / "task40_annotations.json"
TASK40_ANNOTATIONS_LOCK = threading.Lock()
PAGE = Path(__file__).with_name("result_viewer.html")
TRAIN_TOOLS = EVALUATION_ROOT.parent / "train"
sys.path.insert(0, str(TRAIN_TOOLS))
from dashboard import TrainingManager
from benchmark_dashboard import BenchmarkManager


class DetachedEvaluation:
    """Observe an evaluation that remains alive during a dashboard reload."""
    def __init__(self, job):
        self.pid=job['pid']
        self.start=job.get('process_start')
        self.output=Path(job['output'])
        self.returncode=job.get('returncode')

    def poll(self):
        path=Path('/proc')/str(self.pid)/'stat'
        if path.exists():
            fields=path.read_text().split()
            if fields[2]!='Z' and self.start==fields[21]:
                return None
        if self.returncode is not None:
            return self.returncode
        result=self.output/'results.json'
        status=json.loads(result.read_text()).get('status') if result.exists() else None
        return 0 if status in ['completed','complete'] else 1


def restore_evaluation_jobs(manager, records):
    for row in reversed(records):
        manager.jobs.append(dict(id=row['id'],process=DetachedEvaluation(row),gpus=set(row['gpus']),
            log_path=Path(row['log']),command=row['command'],started_at=row['started_at'],
            plan={k:row.get(k) for k in ['output','task_id','version','group']}))


def video_files(run_dir: Path) -> dict[str, Path]:
    """Expose sealed episodes under the same URL before and after archiving.

    The recorder flushes/closes the MP4 before reset renames it with _s0/_s1.
    Plain UUID files in .runtime are still being recorded and stay hidden.
    This is read-only: the running launcher retains ownership of all files.
    """
    videos = {}
    for path in (run_dir / ".runtime").glob("gpu*/*/videos/*.mp4"):
        attempt = path.parents[2].name
        if (re.fullmatch(r"gpu[0-9]+(?:_resume[0-9]+)?", attempt)
                and re.search(r"_s[01](?:_|$)", path.stem) and path.is_file()):
            videos[f"videos/{attempt}/{path.name}"] = path
    # Archived files take precedence as the launcher moves finished artifacts.
    for path in (run_dir / "videos").rglob("*.mp4"):
        if path.is_file():
            videos[path.relative_to(run_dir).as_posix()] = path
    return videos


def load_runs(root: Path) -> list[dict]:
    runs = []
    # Bounded discovery supports evaluation/version/group/run and narrower
    # --root choices; never scan runtime/diagnostic summaries as public runs.
    candidates = list(root.glob("*/*/*/results.json"))
    candidates += list(root.glob("*/*/results.json"))
    candidates += list(root.glob("*/results.json"))
    if (root / "results.json").is_file():
        candidates.append(root / "results.json")
    seen = set()
    for result_path in sorted(candidates, key=lambda p: p.stat().st_mtime, reverse=True):
        parts = result_path.parent.relative_to(root).parts
        if any(part.startswith('.') or part in ('diagnostics', 'videos', 'scripts') for part in parts):
            continue
        identity = result_path.resolve()
        if identity in seen:
            continue
        seen.add(identity)
        try:
            result = json.loads(result_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        result["_folder"] = result_path.parent.relative_to(root).as_posix()
        container = result_path.parent.parent.name
        config = result.get("configuration", {})
        version = result.get("version") or config.get("version")
        group = result.get("group") or config.get("group")
        if version == 'baseline' or (version and re.fullmatch(r'experiment_[1-9][0-9]*', version)):
            group, version = version.replace('experiment_', 'experment_', 1), 'V_1'
        if len(parts) == 3:
            version, group = parts[:2]
        result["version"] = version or 'V_1'
        result["group"] = group or container.replace('experiment_', 'experment_', 1)
        result["_episodes"] = []
        journal = result_path.parent / "episodes.jsonl"
        if journal.exists():
            for line in journal.read_text(encoding="utf-8").splitlines():
                try:
                    result["_episodes"].append(json.loads(line))
                except json.JSONDecodeError:
                    continue
        review_records = human_review.all_reviews(result_path.parent)
        result['_human_reviews'] = review_records
        task_id = result.get('task_id')
        if task_id is None:
            task_id = config.get('task_id')
        try:
            review_stages = human_review.stage_options(task_id) if task_id is not None else []
        except (ValueError, TypeError):
            review_stages = []
        for episode in result['_episodes']:
            review_record = review_records.get(f"episode:{episode.get('episode_index')}", {})
            annotated = human_review.is_annotated(review_record)
            episode['human_review_needs_update'] = (
                annotated and human_review.needs_update(review_record, review_stages))
            episode['human_reviewed'] = annotated and not episode['human_review_needs_update']
        # Recover only recorded native terminal facts, without rewriting old runs.
        for episode in result['_episodes']:
            episode['b1k_metrics'] = b1k_metrics.episode_metrics(episode)
        result['b1k_metrics'] = b1k_metrics.summarize(
            result['_episodes'], result.get('evaluation_mode'), result.get('requested_episodes'))
        # Include legacy/resumed runs without modifying their result files.
        # Missing Q is unknown, rather than a zero or an inferred task progress.
        valid = [episode for episode in result["_episodes"] if episode.get("valid")]
        q_scores = [episode["q_score"] for episode in valid if episode.get("q_score") is not None]
        result["q_score_episodes"] = len(q_scores)
        result["missing_q_score_episodes"] = len(valid) - len(q_scores)
        result["mean_q_score"] = (
            sum(q_scores) / len(q_scores)
            if q_scores and result.get("evaluation_mode") == "official-eval" else None
        )
        result["_video_episodes"] = {
            episode["video_file"]: episode for episode in result["_episodes"]
            if episode.get("video_file")
        }
        result["_videos"] = sorted(video_files(result_path.parent))
        result["_logs"] = {}
        for log_path in result_path.parent.glob("gpu*.log"):
            try:
                result["_logs"][log_path.name] = "".join(log_path.read_text(encoding="utf-8", errors="replace").splitlines(keepends=True)[-100:])
            except OSError:
                pass
        runs.append(result)
    return runs


def percent(value):
    return "-" if value is None else f"{float(value) * 100:.1f}%"


def number(value, digits=1):
    return "-" if value is None else f"{float(value):.{digits}f}"


def elapsed(run: dict) -> str:
    seconds = run.get("elapsed_wall_s")
    if seconds is None:
        return "运行中"
    seconds = int(seconds)
    hours, remainder = divmod(seconds, 3600)
    minutes, seconds = divmod(remainder, 60)
    return f"{hours}h {minutes:02d}m" if hours else f"{minutes}m {seconds:02d}s"


def load_workbench_devices(project_root: Path) -> dict:
    path = Path(project_root) / "configs" / "workbench_devices.json"
    try:
        devices = json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError:
        return {}
    if not isinstance(devices, dict):
        raise ValueError(f"设备配置必须是 JSON 对象: {path}")
    return devices


def print_report(root: Path) -> int:
    runs = load_runs(root)
    if not runs:
        print(f"未找到结果: {root} 下的 results.json（默认 V_1/baseline/运行目录）")
        return 0
    print("评估结果")
    print("=" * 116)
    print(f"{'目录':28} {'状态':10} {'回合':10} {'成功率':9} {'平均Q':9} {'Q记录/有效':10} {'进度':9} {'耗时':10}")
    for run in runs:
        completed = run.get("completed_episodes", run.get("recorded_episodes", 0))
        requested = run.get("requested_episodes", "?")
        print(f"{run['_folder']:28} {run.get('status', '-'):10} {str(completed) + '/' + str(requested):10} "
              f"{percent(run.get('success_rate')):9} {number(run.get('mean_q_score'), 4):9} "
              f"{str(run['q_score_episodes']) + '/' + str(completed):10} "
              f"{number(run.get('mean_task_progress')):>7}% {elapsed(run):10}")
    latest = runs[0]
    config = latest.get("configuration", {}) or {}
    print("\n最新运行:")
    print(f"  模式: {latest.get('evaluation_mode', '-')}  Task: {latest.get('task_id', '-')} {latest.get('task_name', '')}")
    print(f"  成功: {latest.get('successful_episodes', 0)}/{latest.get('completed_episodes', 0)} ({percent(latest.get('success_rate'))})  平均进度: {number(latest.get('mean_task_progress'))}%")
    print(f"  平均Q (0–1): {number(latest.get('mean_q_score'), 4)}  Q记录: {latest['q_score_episodes']}/{latest.get('completed_episodes', 0)} 有效回合；缺失不按0计算，fast-debug不汇总正式Q")
    print(f"  环境: {config.get('mode', '-')}  seed: {config.get('base_seed', '-')} ({config.get('seed_mode', '-')})  Policy: {config.get('policy_server', '-')}")
    print("\n最近回合: GPU/env  instance  seed  success  valid  Q(0–1)  progress  actions")
    for episode in latest.get("_episodes", [])[-8:]:
        print(f"  {episode.get('gpu_index', '-')}/{episode.get('env_idx', '-'):>1}       {episode.get('instance_id', '-'):>3}     "
              f"{episode.get('seed', '-'):>4}  {str(episode.get('success', '-')):<7}  {str(episode.get('valid', '-')):<5}  "
              f"{number(episode.get('q_score'), 4):>7}  "
              f"{number(episode.get('task_progress')):>7}%  {episode.get('sim_steps', '-')}")
    return 0


class Handler(BaseHTTPRequestHandler):
    root = EVALUATION_ROOT
    service_monitor = None
    launch_manager = None
    inference_manager = None
    training_manager = None
    training_devices = None
    evaluation_devices = None
    inference_devices = None
    benchmark_manager = None

    def _evaluation_for(self, device='local', run=None):
        if run and self.evaluation_devices:
            path=self.launch_manager.control.run({'run':run})
            marker=path/'.device.json'
            device=json.loads(marker.read_text())['device'] if marker.exists() else 'local'
        if device=='local': return self.launch_manager
        if not self.evaluation_devices or device not in self.evaluation_devices:
            raise ValueError('未知评估设备')
        return self.evaluation_devices[device]

    def _training_for(self, device='a100'):
        if self.training_devices is None:
            if device != 'a100':
                raise ValueError('训练设备未配置')
            return self.training_manager
        if device not in self.training_devices:
            raise ValueError('未知训练设备')
        return self.training_devices[device]

    def _json(self, value, status=200):
        body = json.dumps(value, ensure_ascii=False).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(body)

    def _save_registry_annotation(self):
        origin = self.headers.get("Origin")
        if origin and urlparse(origin).netloc != self.headers.get("Host"):
            raise ValueError("请从本评估页面提交修改")
        length = int(self.headers.get("Content-Length", "0"))
        if not 0 < length <= 4096:
            raise ValueError("台账修改内容长度不正确")
        request = json.loads(self.rfile.read(length))
        if not isinstance(request, dict) or set(request) - {"key", "label", "notes", "hidden"}:
            raise ValueError("台账修改字段不正确")
        key = request.get("key")
        if not isinstance(key, str) or len(key) > 2048:
            raise ValueError("评估记录标识不正确")
        current = experiment_registry.cached()
        if not any(i.get("kind") == "eval" and i.get("id") == key for i in current.get("items", [])):
            raise ValueError("找不到对应评估记录，请先重新扫描")
        label, notes, hidden = request.get("label", ""), request.get("notes", ""), request.get("hidden", False)
        if not isinstance(label, str) or len(label) > 48:
            raise ValueError("标记最多 48 个字符")
        if not isinstance(notes, str) or len(notes) > 500:
            raise ValueError("备注最多 500 个字符")
        if not isinstance(hidden, bool):
            raise ValueError("归档状态不正确")
        with REGISTRY_ANNOTATIONS_LOCK:
            try:
                annotations = json.loads(REGISTRY_ANNOTATIONS.read_text())
                if not isinstance(annotations, dict):
                    annotations = {}
            except (OSError, ValueError):
                annotations = {}
            entry = {"label": label.strip(), "notes": notes.strip(), "hidden": hidden}
            if not any(entry.values()):
                annotations.pop(key, None)
            else:
                annotations[key] = entry
            REGISTRY_ANNOTATIONS.parent.mkdir(parents=True, exist_ok=True)
            temp = REGISTRY_ANNOTATIONS.with_suffix(".json.tmp")
            temp.write_text(json.dumps(annotations, ensure_ascii=False, indent=2))
            os.replace(temp, REGISTRY_ANNOTATIONS)
        return annotations

    def _launch_post(self, route):
        manager = (self.benchmark_manager if route.startswith('/api/benchmarks/') else
                   self.training_manager if route.startswith('/api/training/') else
                   self.inference_manager if route.startswith('/api/inference/') else self.launch_manager)
        if manager is None:
            self._json({"error": "启动服务未初始化"}, 503)
            return
        try:
            if self.headers.get("Content-Type", "").split(";")[0] != "application/json":
                raise ValueError("请使用 JSON 启动参数")
            origin = self.headers.get("Origin")
            if origin and urlparse(origin).netloc != self.headers.get("Host"):
                raise ValueError("请从本评估页面提交启动请求")
            length = int(self.headers.get("Content-Length", "0"))
            if not 0 < length <= 8192:
                raise ValueError("启动参数长度不正确")
            request = json.loads(self.rfile.read(length))
            if route.startswith('/api/training/'):
                manager = self._training_for(request.pop('device', 'a100'))
            elif route.startswith('/api/launch/'):
                device=request.pop('device','local')
                manager=self._evaluation_for(device,request.get('run') or request.get('resume_from'))
            elif route.startswith('/api/inference/'):
                device=request.pop('device','a100')
                if self.inference_devices:
                    if device not in self.inference_devices: raise ValueError('未知推理设备')
                    manager=self.inference_devices[device]
            action = route.rsplit('/', 1)[-1]
            operation = {'preview': 'prepare', 'start': 'start', 'sync': 'sync', 'stop': 'stop', 'dataset': 'prepare_dataset',
                         'pause':'pause','delete':'delete','cancel':'cancel','resume':'resume'}[action]
            value = getattr(manager, operation)(request)
        except (ValueError, OSError, subprocess.SubprocessError) as error:
            self._json({"error": str(error)}, 400)
            return
        self._json(value, 200 if route.endswith("/preview") else 202)

    def _preview_run(self, name):
        root = self.root.resolve()
        run = (root / name).resolve()
        if not name or not run.is_relative_to(root) or not (run / "results.json").is_file():
            return None
        return run

    def _vlm_diagnose(self):
        try:
            length = int(self.headers.get("Content-Length", "0"))
            if not 0 < length <= 65536:
                raise ValueError("诊断参数长度不正确")
            request = json.loads(self.rfile.read(length))
            run = self._preview_run(request.get("run", ""))
            if run is None:
                raise ValueError("找不到该运行目录")
            video = video_files(run).get(request.get("path", ""))
            if video is None or not video.is_file():
                raise ValueError("找不到该回合录像")
            value = vlm_diagnosis.diagnose(
                run, video,
                endpoint=request.get("endpoint") or vlm_diagnosis.DEFAULT_ENDPOINT,
                task=request.get("task", ""),
                episode=str(request.get("episode", "")),
                instruction=request.get("instruction", ""),
                frames=int(request.get("frames") or 0),
                notes=request.get("notes", ""),
                task_name=request.get("task_name", ""),
                force=bool(request.get("force")),
            )
        except (ValueError, OSError, subprocess.SubprocessError, RuntimeError) as error:
            self._json({"error": str(error)}, 400)
            return
        except Exception as error:  # noqa: BLE001 - 远端服务异常直接回报界面
            self._json({"error": f"{type(error).__name__}: {error}"}, 502)
            return
        self._json(value, 200)

    def _vlm_batch(self):
        try:
            length = int(self.headers.get("Content-Length", "0"))
            request = json.loads(self.rfile.read(length)) if 0 < length <= 65536 else {}
            run = self._preview_run(request.get("run", ""))
            if run is None:
                raise ValueError("找不到该运行目录")
            value = vlm_diagnosis.start_batch(
                run,
                endpoint=request.get("endpoint") or vlm_diagnosis.DEFAULT_ENDPOINT,
                frames=int(request.get("frames") or 0),
                task_name=request.get("task_name", ""),
                force=bool(request.get("force")),
            )
        except (ValueError, OSError) as error:
            self._json({"error": str(error)}, 400)
            return
        self._json(value, 200)

    def _stage_review(self):
        try:
            length = int(self.headers.get('Content-Length', '0'))
            if not 0 < length <= 262144:
                raise ValueError('复核参数长度不正确')
            request = json.loads(self.rfile.read(length))
            run = self._preview_run(request.get('run', ''))
            if run is None:
                raise ValueError('找不到该运行目录')
            if urlparse(self.path).path.endswith('/batch'):
                self._json(task_stage_diagnostics.archive_run(run))
                return
            report = task_stage_diagnostics.analyze_run(run)
            matches = [r for r in report['episodes'] if r['identity']['episode_index'] == request.get('episode_index')]
            if len(matches) != 1:
                raise ValueError('找不到唯一回合')
            episode = next(row for row in task_stage_diagnostics.read_jsonl(run/'episodes.jsonl')
                           if row.get('episode_index') == request['episode_index'])
            if urlparse(self.path).path.endswith('/diagnose'):
                video = video_files(run).get(episode.get('video_file'))
                if video is None:
                    raise ValueError('找不到回合录像')
                self._json(vlm_stage_diagnosis.diagnose(run,episode,video,force=bool(request.get('force'))))
                return
            self._json(task_stage_diagnostics.save_review(run, episode, request))
        except (ValueError, OSError, TypeError, RuntimeError, subprocess.SubprocessError) as error:
            self._json({'error':str(error)}, 400)

    def _human_episode_review(self, request):
        run = self._preview_run(request.get('run', ''))
        if run is None:
            raise ValueError('找不到该运行目录')
        episode_index = request.get('episode_index')
        identity = human_review.target(run, episode_index=episode_index)
        if identity.get('task_id') is None:
            result = json.loads((run / 'results.json').read_text(encoding='utf-8'))
            identity['task_id'] = result.get('task_id', result.get('configuration', {}).get('task_id'))
        if request.get('action') == 'delete':
            return {'deleted': human_review.delete(run, identity)}
        options = human_review.stage_options(identity.get('task_id'))
        review = request.get('review', {})
        if not isinstance(review, dict):
            raise ValueError('标注内容格式不正确')
        previous = human_review.read(run, identity) or {}
        review = {name: previous.get(name) for name in human_review.FIELDS}
        review.update(request.get('review', {}))
        review['reviewed'] = True
        selected_stage = review.get('reached_stage') if isinstance(review, dict) else None
        if not selected_stage or not review.get('root_cause') or not review.get('specific_reason'):
            raise ValueError('请依次选择失败阶段、主要根因和根因分类')
        stage = next((item for item in options if item['id'] == selected_stage), None)
        if selected_stage and stage is None:
            raise ValueError('请刷新后重新选择当前 Task 的失败阶段')
        if stage:
            review['stage_label'] = stage['label']
        causes = stage.get('causes', []) if stage else []
        selected_cause = review.get('root_cause')
        cause = next((item for item in causes if item['id'] == selected_cause), None)
        if selected_cause and cause is None:
            raise ValueError('所选主要根因不属于当前阶段，请重新选择')
        if cause:
            review['root_cause_label'] = cause['label']
        specific_reason = next((item for item in cause.get('notes', [])
                                if item['id'] == review.get('specific_reason')), None) if cause else None
        if specific_reason is None:
            raise ValueError('所选根因分类不属于当前根因，请重新选择')
        review['specific_reason_label'] = specific_reason['label']
        secondary_cause_id = review.get('secondary_root_cause')
        secondary_specific_id = review.get('secondary_specific_reason')
        secondary_cause = None
        secondary_specific = None
        if secondary_cause_id and not secondary_specific_id:
            raise ValueError('旧版补充原因缺少分类')
        if secondary_cause_id:
            secondary_cause = next((item for item in causes if item['id'] == secondary_cause_id), None)
            if secondary_cause is None:
                raise ValueError('旧版补充原因根因不属于当前阶段')
            secondary_specific = next((item for item in secondary_cause.get('notes', [])
                                       if item['id'] == secondary_specific_id), None)
            if secondary_specific is None:
                raise ValueError('旧版补充原因不属于所选根因分类')
            review['secondary_root_cause_label'] = secondary_cause['label']
            review['secondary_specific_reason_label'] = secondary_specific['label']
        else:
            review['secondary_root_cause'] = None
            review['secondary_root_cause_label'] = ''
            if secondary_specific_id:
                secondary_specific = next((item for item in stage.get('supplements', [])
                                           if item['id'] == secondary_specific_id), None)
                if secondary_specific is None:
                    raise ValueError('补充原因不属于当前阶段，请刷新后重新选择')
                review['secondary_specific_reason_label'] = secondary_specific['label']
            else:
                review['secondary_specific_reason_label'] = ''
        return human_review.save(
            run, identity, review, {item['id'] for item in options},
            {item['id'] for item in causes}, {item['id'] for item in cause.get('notes', [])},
            {item['id'] for item in causes},
            ({item['id'] for item in stage.get('supplements', [])}
             | {item['id'] for category in causes for item in category.get('notes', [])}))

    def do_POST(self):
        route = urlparse(self.path).path
        if route == '/api/human-review/options':
            try:
                length = int(self.headers.get('Content-Length', '0'))
                if not 0 < length <= 4096:
                    raise ValueError('选项参数长度不正确')
                request = json.loads(self.rfile.read(length))
                options = human_review.update_annotation_options(
                    request.get('task_id'), request.get('kind'), request.get('action'),
                    request.get('stage_id'), request.get('cause_id'), request.get('label'),
                    request.get('option_id'))
                self._json({'stages': options})
            except (ValueError, OSError, TypeError, json.JSONDecodeError) as error:
                self._json({'error': str(error)}, 400)
            return
        if route == '/api/human-review/stages':
            try:
                length = int(self.headers.get('Content-Length', '0'))
                if not 0 < length <= 4096:
                    raise ValueError('阶段参数长度不正确')
                request = json.loads(self.rfile.read(length))
                options = human_review.update_stage_options(
                    request.get('task_id'), request.get('action'),
                    request.get('label'), request.get('option_id'), request.get('before_id'))
                self._json({'stages': options})
            except (ValueError, OSError, TypeError, json.JSONDecodeError) as error:
                self._json({'error': str(error)}, 400)
            return
        if route == '/api/human-review':
            try:
                length = int(self.headers.get('Content-Length', '0'))
                if not 0 < length <= 16384:
                    raise ValueError('标注参数长度不正确')
                request = json.loads(self.rfile.read(length))
                self._json(self._human_episode_review(request))
            except (ValueError, OSError, TypeError, json.JSONDecodeError) as error:
                self._json({'error': str(error)}, 400)
            return
        if route in ('/api/vlm/stages/review', '/api/vlm/stages/batch', '/api/vlm/stages/diagnose'):
            self._stage_review()
            return
        if route == "/api/registry/annotations":
            try:
                self._json(self._save_registry_annotation())
            except (ValueError, OSError, json.JSONDecodeError) as error:
                self._json({"error": str(error)}, 400)
            return
        if route == "/api/task40/annotations":
            try:
                length = int(self.headers.get('Content-Length', '0'))
                if length < 1 or length > 8192:
                    raise ValueError('invalid request size')
                payload = json.loads(self.rfile.read(length))
                case_id = payload.get('case_id', '')
                if not re.fullmatch(r'C\d{2}', case_id):
                    raise ValueError('invalid case_id')
                with TASK40_ANNOTATIONS_LOCK:
                    data = json.loads(TASK40_ANNOTATIONS_FILE.read_text()) if TASK40_ANNOTATIONS_FILE.is_file() else {}
                    data[case_id] = {k: v for k, v in payload.items()}
                    TASK40_ANNOTATIONS_FILE.parent.mkdir(parents=True, exist_ok=True)
                    TASK40_ANNOTATIONS_FILE.write_text(json.dumps(data, ensure_ascii=False, indent=2))
                self._json(data)
            except (ValueError, OSError, json.JSONDecodeError) as error:
                self._json({'error': str(error)}, 400)
            return
        if route == "/api/vlm/diagnose":
            self._vlm_diagnose()
            return
        if route == "/api/vlm/batch":
            self._vlm_batch()
            return
        if route in ('/api/benchmarks/preview', '/api/benchmarks/start', '/api/benchmarks/stop'):
            self._launch_post(route)
            return
        if route in ("/api/launch/preview", "/api/launch/start", "/api/launch/pause", "/api/launch/delete", "/api/inference/preview", "/api/inference/start", "/api/inference/stop",
                     "/api/training/preview", "/api/training/start", "/api/training/sync", "/api/training/stop", "/api/training/cancel", "/api/training/delete", "/api/training/resume", "/api/training/dataset"):
            self._launch_post(route)
            return
        if route != "/api/preview":
            self.send_error(404)
            return
        try:
            length = int(self.headers.get("Content-Length", "0"))
            if length < 1 or length > 4096:
                raise ValueError("Invalid preview request size")
            request = json.loads(self.rfile.read(length))
            run = self._preview_run(request.get("run", ""))
            if run is None:
                self.send_error(404, "Run not found")
                return
            if not isinstance(request.get("enabled"), bool):
                raise ValueError("enabled must be boolean")
            if request["enabled"] and json.loads((run / "results.json").read_text()).get("status") != "running":
                raise ValueError("Preview requires a running evaluation")
            subscribe(run, request.get("client"), request["enabled"])
        except (ValueError, TypeError, AttributeError) as error:
            self.send_error(400, str(error))
            return
        self.send_response(204)
        self.send_header("Cache-Control", "no-store")
        self.send_header("Content-Length", "0")
        self.end_headers()

    def _video_file(self):
        query = parse_qs(urlparse(self.path).query)
        run_name = query.get("run", [""])[0]
        relative_name = query.get("path", [""])[0]
        root = self.root.resolve()
        run_dir = (root / run_name).resolve()
        if not run_name or not relative_name or not run_dir.is_relative_to(root):
            return None
        candidate = video_files(run_dir).get(relative_name)
        if candidate is None:
            return None
        video_path = candidate.resolve()
        if not video_path.is_relative_to(run_dir) or not video_path.is_file():
            return None
        return video_path

    def _serve_video(self, video_path=None):
        video_path = video_path or self._video_file()
        if video_path is None:
            self.send_error(404, "Video not found")
            return
        size = video_path.stat().st_size
        start, end, status = 0, size - 1, 200
        range_header = self.headers.get("Range", "")
        if range_header.startswith("bytes="):
            requested = range_header[6:].split(",", 1)[0].strip()
            try:
                first, last = requested.split("-", 1)
                if first:
                    start = int(first)
                    end = int(last) if last else size - 1
                else:
                    length = int(last)
                    start = max(0, size - length)
                if start < 0 or start >= size or end < start:
                    raise ValueError
                end = min(end, size - 1)
                status = 206
            except ValueError:
                self.send_error(416, "Invalid video range")
                return
        length = end - start + 1
        self.send_response(status)
        self.send_header("Content-Type", "video/mp4")
        self.send_header("Accept-Ranges", "bytes")
        self.send_header("Content-Length", str(length))
        if status == 206:
            self.send_header("Content-Range", f"bytes {start}-{end}/{size}")
        self.send_header("Cache-Control", "no-cache")
        self.end_headers()
        with video_path.open("rb") as handle:
            handle.seek(start)
            remaining = length
            while remaining:
                chunk = handle.read(min(1024 * 1024, remaining))
                if not chunk:
                    break
                self.wfile.write(chunk)
                remaining -= len(chunk)

    def do_GET(self):
        route = urlparse(self.path).path
        if route == '/api/human-review/stages':
            query = {k: v[0] for k, v in parse_qs(urlparse(self.path).query).items()}
            try:
                task_id = int(query.get('task_id', ''))
                self._json({'task_id': task_id, 'stages': human_review.stage_options(task_id)})
            except (ValueError, OSError, TypeError) as error:
                self._json({'error': str(error)}, 400)
            return
        if route == '/api/human-review':
            query = {k: v[0] for k, v in parse_qs(urlparse(self.path).query).items()}
            try:
                run = self._preview_run(query.get('run', ''))
                if run is None:
                    raise ValueError('找不到该运行目录')
                try:
                    episode_index = int(query.get('episode_index', ''))
                except ValueError as error:
                    raise ValueError('回合编号无效') from error
                identity = human_review.target(run, episode_index=episode_index)
                if identity.get('task_id') is None:
                    result = json.loads((run / 'results.json').read_text(encoding='utf-8'))
                    identity['task_id'] = result.get('task_id', result.get('configuration', {}).get('task_id'))
                review = human_review.read(run, identity)
                options = human_review.stage_options(identity.get('task_id'))
                if review:
                    stage = next((item for item in options if item['id'] == review.get('reached_stage')), None)
                    cause = next((item for item in (stage or {}).get('causes', [])
                                  if item['id'] == review.get('root_cause')), None)
                    review['root_cause_label'] = review.get('root_cause_label') or (cause or {}).get('label')
                self._json({'review': review})
            except (ValueError, OSError, TypeError) as error:
                self._json({'error': str(error)}, 400)
            return
        if route in ('/api/vlm/stages', '/api/vlm/stages/export', '/api/vlm/stages/compare'):
            query = {k:v[0] for k,v in parse_qs(urlparse(self.path).query).items()}
            try:
                run = self._preview_run(query.get('run', ''))
                if run is None:
                    raise ValueError('找不到该运行目录')
                report = task_stage_diagnostics.analyze_run(run)
                report['recommendations'] = task_stage_diagnostics.recommendations(report)
                if route.endswith('/compare'):
                    other = self._preview_run(query.get('other', ''))
                    if other is None:
                        raise ValueError('找不到对比运行目录')
                    self._json(task_stage_diagnostics.compare(report, task_stage_diagnostics.analyze_run(other)))
                elif route.endswith('/export'):
                    format = query.get('format', 'json')
                    body = task_stage_diagnostics.export(report, format).encode('utf-8')
                    self.send_response(200)
                    self.send_header('Content-Type', {'json':'application/json','csv':'text/csv','md':'text/markdown'}[format]+'; charset=utf-8')
                    self.send_header('Content-Disposition', f'attachment; filename="task40_stages.{format}"')
                    self.send_header('Content-Length', str(len(body)))
                    self.end_headers()
                    self.wfile.write(body)
                else:
                    self._json(report)
            except (ValueError, OSError, TypeError) as error:
                self._json({'error':str(error)}, 400)
            return
        if route == '/task40_diagnostics.js':
            body = Path(__file__).with_name('task40_diagnostics.js').read_bytes()
            self.send_response(200)
            self.send_header('Content-Type', 'application/javascript; charset=utf-8')
            self.send_header('Content-Length', str(len(body)))
            self.send_header('Cache-Control', 'no-store')
            self.end_headers()
            self.wfile.write(body)
            return
        ai_assets = {
            '/ai': ('ai_workbench.html', 'text/html'),
            '/ai/': ('ai_workbench.html', 'text/html'),
            '/ai_workbench.html': ('ai_workbench.html', 'text/html'),
            '/ai_workbench.css': ('ai_workbench.css', 'text/css'),
            '/ai_workbench.js': ('ai_workbench.js', 'application/javascript'),
            '/ai/architecture': ('ai_architecture.html', 'text/html'),
            '/ai/architecture/': ('ai_architecture.html', 'text/html'),
            '/ai_architecture.html': ('ai_architecture.html', 'text/html'),
            '/ai_architecture.css': ('ai_architecture.css', 'text/css'),
            '/ai_architecture.js': ('ai_architecture.js', 'application/javascript'),
        }
        if route in ai_assets:
            name, content_type = ai_assets[route]
            body = Path(__file__).with_name(name).read_bytes()
            self.send_response(200)
            self.send_header('Content-Type', content_type + '; charset=utf-8')
            self.send_header('Content-Length', str(len(body)))
            self.send_header('Cache-Control', 'no-store')
            self.end_headers()
            self.wfile.write(body)
            return
        if route == "/api/registry/annotations":
            try:
                self._json(json.loads(REGISTRY_ANNOTATIONS.read_text()))
            except (OSError, ValueError):
                self._json({})
            return
        if route in ("/registry", "/api/registry"):
            if route == "/registry":
                try:
                    body = Path(__file__).with_name("registry_view.html").read_bytes()
                except OSError as error:
                    self.send_error(500, f"Cannot read registry page: {error}")
                    return
                self.send_response(200)
                self.send_header("Content-Type", "text/html; charset=utf-8")
                self.send_header("Content-Length", str(len(body)))
                self.send_header("Cache-Control", "no-store")
                self.end_headers()
                self.wfile.write(body)
                return
            query = {k: v[0] for k, v in parse_qs(urlparse(self.path).query).items()}
            try:
                self._json(experiment_registry.cached(force=query.get("force") == "1"))
            except Exception as error:
                self._json({"error": str(error)}, 500)
            return
        if route in ("/api/vlm/health", "/api/vlm/cached", "/api/vlm/batch"):
            query = {k: v[0] for k, v in parse_qs(urlparse(self.path).query).items()}
            try:
                if route.endswith("health"):
                    value = vlm_diagnosis.health(
                        query.get("endpoint") or vlm_diagnosis.DEFAULT_ENDPOINT)
                elif route.endswith("batch"):
                    run = self._preview_run(query.get("run", ""))
                    if run is None:
                        raise ValueError("找不到该运行目录")
                    value = vlm_diagnosis.batch_state(run)
                else:
                    run = self._preview_run(query.get("run", ""))
                    if run is None:
                        raise ValueError("找不到该运行目录")
                    video = video_files(run).get(query.get("path", ""))
                    if video is None or not video.is_file():
                        raise ValueError("找不到该回合录像")
                    value = {"diagnosis": vlm_diagnosis.cached(run, video)}
                self._json(value)
            except (ValueError, OSError) as error:
                self._json({"error": str(error)}, 400)
            return
        if route == '/api/benchmarks/video':
            try:
                query={k:v[0] for k,v in parse_qs(urlparse(self.path).query).items()}
                if self.benchmark_manager is None: raise ValueError('Benchmark 工作台未初始化')
                self._serve_video(self.benchmark_manager.video_path(query))
            except (ValueError,OSError) as error:
                self._json({'error':str(error)},404)
            return
        if route == '/benchmarks.js':
            code = Path(__file__).with_name('benchmark_workbench.js').read_bytes()
            code += b'\n'+Path(__file__).with_name('libero_workflow.js').read_bytes()
            template = Path(__file__).with_name('benchmark_workbench.html').read_text()
            body = ('window.__benchmarkHTML = '+json.dumps(template)+';\n').encode()+code
            self.send_response(200)
            self.send_header('Content-Type', 'application/javascript; charset=utf-8')
            self.send_header('Content-Length', str(len(body)))
            self.send_header('Cache-Control', 'no-store')
            self.end_headers()
            self.wfile.write(body)
            return
        if route == '/api/benchmarks/image':
            try:
                if self.benchmark_manager is None:
                    raise ValueError('Benchmark 工作台未初始化')
                query = {k: v[0] for k, v in parse_qs(urlparse(self.path).query).items()}
                body = self.benchmark_manager.image_path(query).read_bytes()
                self.send_response(200)
                self.send_header('Content-Type', 'image/png')
                self.send_header('Content-Length', str(len(body)))
                self.end_headers()
                self.wfile.write(body)
            except (ValueError, OSError) as error:
                self._json({'error': str(error)}, 404)
            return
        if route in ('/api/benchmarks/options', '/api/benchmarks/jobs', '/api/benchmarks/detail'):
            if self.benchmark_manager is None:
                self._json({'error': 'Benchmark 工作台未初始化'}, 503)
                return
            try:
                query = {k: v[0] for k, v in parse_qs(urlparse(self.path).query).items()}
                value = (self.benchmark_manager.options() if route.endswith('/options') else
                         self.benchmark_manager.snapshot() if route.endswith('/jobs') else
                         self.benchmark_manager.detail(query))
                self._json(value)
            except (ValueError, OSError, subprocess.SubprocessError) as error:
                self._json({'error': str(error)}, 400)
            return
        if route in ('/training.js', '/training.html'):
            file = TRAIN_TOOLS / ('dashboard.js' if route.endswith('.js') else 'dashboard.html')
            body = file.read_bytes()
            if route.endswith('.js'):
                # Keep a tab's training markup and handlers from the same load.
                # Fetching markup later can mix a new form with an older script.
                template = (TRAIN_TOOLS / 'dashboard.html').read_text()
                body = ('window.__grootTrainingPageHTML = ' + json.dumps(template) + ';\n').encode() + body
            self.send_response(200)
            self.send_header('Content-Type', 'application/javascript; charset=utf-8' if route.endswith('.js') else 'text/html; charset=utf-8')
            self.send_header('Content-Length', str(len(body)))
            self.send_header('Cache-Control', 'no-store')
            self.end_headers()
            self.wfile.write(body)
            return
        if route.startswith('/api/training/'):
            if self.training_manager is None:
                self._json({'error': '训练工作台未初始化'}, 503)
                return
            try:
                action = route.rsplit('/', 1)[-1]
                query = {k: v[0] for k,v in parse_qs(urlparse(self.path).query).items()}
                manager=self._training_for(query.pop('device','a100'))
                if action == 'options': value = manager.options()
                elif action == 'jobs': value = manager.snapshot()
                elif action == 'detail': value = manager.detail(query)
                elif action == 'sync': value = manager.sync_state()
                else:
                    self.send_error(404)
                    return
                self._json(value)
            except (ValueError, OSError, subprocess.SubprocessError) as error:
                self._json({'error': str(error)}, 400)
            return
        if route in ("/api/launch/options", "/api/launch/jobs", "/api/inference/options", "/api/inference/jobs"):
            inference = route.startswith('/api/inference/')
            manager = self.inference_manager if inference else self.launch_manager
            if manager is None:
                self._json({"error": "启动服务未初始化"}, 503)
            else:
                try:
                    if inference and self.inference_devices:
                        query=parse_qs(urlparse(self.path).query)
                        device=query.get('device',['a100'])[0]
                        if device not in self.inference_devices: raise ValueError('未知推理设备')
                        manager=self.inference_devices[device]
                    if not inference and route.endswith('/options'):
                        query=parse_qs(urlparse(self.path).query)
                        manager=self._evaluation_for(query.get('device',['local'])[0])
                    value = manager.options() if route.endswith('/options') else manager.snapshot()
                    if not inference and route.endswith('/jobs'):
                        errors={}
                        for key,remote in (self.evaluation_devices or {}).items():
                            try:
                                value.extend(remote.snapshot())
                                if remote.error: errors[key]=remote.error
                            except (ValueError,OSError,subprocess.SubprocessError) as error: errors[key]=str(error)
                        self._json({'jobs':value,'device_errors':errors})
                        return
                    if not inference and route.endswith('/options') and manager is self.launch_manager:
                        from groot_service_status import collect_snapshot
                        value['gpus']=collect_snapshot().get('gpus',[])
                        value['ready']=True
                    self._json(value if inference or route.endswith('/options') else {"jobs": value})
                except (ValueError, OSError, subprocess.SubprocessError) as error:
                    self._json({"error": str(error)}, 400)
            return
        if route == "/api/results":
            body = json.dumps({"updated_at": time.strftime("%H:%M:%S"), "runs": load_runs(self.root)}, ensure_ascii=False).encode("utf-8")
            self.send_response(200)
            self.send_header("Content-Type", "application/json; charset=utf-8")
        elif route == "/api/services":
            value = self.service_monitor.snapshot() if self.service_monitor else {
                "local": {"ok": None}, "remote": {"ok": None}}
            body = json.dumps(value, ensure_ascii=False).encode("utf-8")
            self.send_response(200)
            self.send_header("Content-Type", "application/json; charset=utf-8")
        elif route == "/api/preview":
            query = parse_qs(urlparse(self.path).query)
            run = self._preview_run(query.get("run", [""])[0])
            if run is None:
                self.send_error(404, "Run not found")
                return
            body = json.dumps(read_preview(run), ensure_ascii=False).encode("utf-8")
            self.send_response(200)
            self.send_header("Content-Type", "application/json; charset=utf-8")
        elif route == "/video":
            self._serve_video()
            return
        elif route in ("/", "/index.html"):
            try:
                body = PAGE.read_bytes()
            except OSError as error:
                self.send_error(500, f"Cannot read viewer page: {error}")
                return
            self.send_response(200)
            self.send_header("Content-Type", "text/html; charset=utf-8")
        elif route == '/task40':
            try:
                body = Path(__file__).with_name('task40_annotate.html').read_bytes()
            except OSError as error:
                self.send_error(500, f"Cannot read annotation page: {error}")
                return
            self.send_response(200)
            self.send_header('Content-Type', 'text/html; charset=utf-8')
            self.send_header('Content-Length', str(len(body)))
            self.send_header('Cache-Control', 'no-store')
            self.end_headers()
            self.wfile.write(body)
            return
        if route == '/api/task40/annotations':
            try:
                data = json.loads(TASK40_ANNOTATIONS_FILE.read_text()) if TASK40_ANNOTATIONS_FILE.is_file() else {}
            except (OSError, ValueError):
                data = {}
            self._json(data)
            return
        if route == '/api/task40/video':
            query = {k: v[0] for k, v in parse_qs(urlparse(self.path).query).items()}
            case = query.get('case', '')
            if not re.fullmatch(r'C\d{2}', case):
                self.send_error(400, 'invalid case')
                return
            vp = TASK40_VIDEOS / f'sequence_{case}.mp4'
            if not vp.is_file():
                self.send_error(404, 'video not found')
                return
            self._serve_video(vp)
            return
        elif route not in ('/api/results', '/api/services', '/api/preview', '/', '/index.html'):
            self.send_error(404)
            return
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(body)

    def log_message(self, _format, *_args):
        pass


def main() -> int:
    parser = argparse.ArgumentParser(description="查看 evaluation 下的 GR00T 评估结果")
    parser.add_argument("--root", type=Path, default=EVALUATION_ROOT, help="结果根目录")
    parser.add_argument("--once", action="store_true", help="只输出终端汇总")
    parser.add_argument("--host", default="127.0.0.1", help="本地监听地址")
    parser.add_argument("--port", type=int, default=8765, help="本地网页端口")
    parser.add_argument("--no-browser", action="store_true", help="只启动服务，不自动打开浏览器")
    parser.add_argument("--policy-monitor-host", default="172.28.62.17", help="通过现有SSH检查推理服务的A100主机")
    args = parser.parse_args()
    root = args.root.expanduser().resolve()
    if args.once:
        return print_report(root)
    if not PAGE.is_file():
        parser.error(f"viewer page not found: {PAGE}")
    Handler.root = root
    Handler.benchmark_manager = BenchmarkManager(EVALUATION_ROOT.parent)
    Handler.launch_manager = LaunchManager(EVALUATION_ROOT.parent)
    evaluation_record=EVALUATION_ROOT.parent/'outputs/shared/web_launch/evaluation_jobs.json'
    if evaluation_record.exists():
        restore_evaluation_jobs(Handler.launch_manager,json.loads(evaluation_record.read_text()))
    Handler.inference_manager = InferenceManager(EVALUATION_ROOT.parent, args.policy_monitor_host)
    Handler.launch_manager.policy_resolver = Handler.inference_manager.resolve
    inference_record=EVALUATION_ROOT.parent/'outputs/shared/web_launch/inference_jobs.json'
    if inference_record.exists():
        saved=json.loads(inference_record.read_text())
        if saved.get('host')==args.policy_monitor_host:
            Handler.inference_manager.jobs=saved.get('jobs',[])
    Handler.training_manager = TrainingManager(EVALUATION_ROOT.parent, args.policy_monitor_host)
    try:
        devices = load_workbench_devices(EVALUATION_ROOT.parent)
    except (OSError, ValueError) as error:
        parser.error(f"设备配置无效: {error}")
    Handler.training_devices={'a100':Handler.training_manager}
    for key,device in devices.items():
        if key != 'a100':
            Handler.training_devices[key]=TrainingManager(EVALUATION_ROOT.parent,device['host'],key,device)
    from groot_device_eval import DeviceEvaluationManager
    Handler.evaluation_devices={key:DeviceEvaluationManager(EVALUATION_ROOT.parent,key,device,Handler.training_devices[key])
        for key,device in devices.items() if device.get('evaluation')}
    Handler.inference_devices={'a100':Handler.inference_manager}
    Handler.inference_devices.update({key:InferenceManager(EVALUATION_ROOT.parent,device['host'],device)
        for key,device in devices.items() if device.get('evaluation')})
    def resolve_policy(host,port):
        for manager in Handler.inference_devices.values():
            if host==manager.bind_host: return manager.resolve(host,port)
        return None
    Handler.launch_manager.policy_resolver=resolve_policy
    for manager in Handler.evaluation_devices.values():
        manager.policy_resolver=resolve_policy
    try:
        server = ThreadingHTTPServer((args.host, args.port), Handler)
    except OSError as error:
        parser.error(f"cannot start viewer at {args.host}:{args.port}: {error}")
    from groot_service_status import ServiceMonitor
    Handler.service_monitor = ServiceMonitor(args.policy_monitor_host,
        additional_hosts={key:device['host'] for key,device in devices.items() if key!='a100'})
    Handler.service_monitor.start()
    url = f"http://{args.host}:{server.server_address[1]}/"
    print(f"GR00T 评估结果面板: {url}", flush=True)
    print(f"扫描目录: {root}", flush=True)
    print("按 Ctrl+C 关闭面板服务（不会停止评估）", flush=True)
    if not args.no_browser:
        webbrowser.open(url, new=2)
    try:
        server.serve_forever(poll_interval=0.5)
    except KeyboardInterrupt:
        print("\n结果面板已关闭。", flush=True)
    finally:
        records=Handler.launch_manager.snapshot()
        for row in records:
            path=Path('/proc')/str(row['pid'])/'stat'
            row['process_start']=path.read_text().split()[21] if path.exists() else None
        evaluation_record.parent.mkdir(parents=True,exist_ok=True)
        evaluation_record.write_text(json.dumps(records,ensure_ascii=False))
        inference_record.parent.mkdir(parents=True,exist_ok=True)
        inference_record.write_text(json.dumps(dict(host=args.policy_monitor_host,jobs=Handler.inference_manager.jobs),ensure_ascii=False))
        server.server_close()
        Handler.service_monitor.close()
        for manager in Handler.training_devices.values():
            manager.close()
        for manager in Handler.evaluation_devices.values():
            manager.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
