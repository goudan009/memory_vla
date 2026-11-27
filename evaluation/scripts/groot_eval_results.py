"""Persist native BEHAVIOR episode results before vector environments reset."""

import csv
import fcntl
from functools import partial
import json
import os
from pathlib import Path
import sys
import time

sys.path.insert(0, str(Path(__file__).resolve().parent))
import b1k_metrics

MANIFEST_FIELDS = [
    "episode_index", "evaluation_mode", "task", "env_idx", "instance_id", "seed", "success", "b1k_success", "q_score", "task_progress",
    "valid", "episode_length", "duration_s", "terminated", "truncated", "video_file",
]

ACTIVE_WRITER = None


def run_eval_async_worker(*args, **kwargs):
    """Keep Gym's protocol; avoid Kit plugin static teardown after worker close."""
    import sys
    from gymnasium.vector.async_vector_env import _async_worker

    _async_worker(*args, **kwargs)
    # Gym has already replied to close and run env.close(). With a live Kit
    # singleton, interpreter teardown unloads SyntheticData before its graph
    # consumers and segfaults. Match the eval bootstrap's process exit policy;
    # episode files are closed before reaching this boundary.
    og = sys.modules.get("omnigibson")
    if og is not None and getattr(og, "app", None) is not None:
        print(f"EVAL_WORKER_EXIT pid={os.getpid()} gym_worker_returned=True", flush=True)
        sys.stderr.flush()
        os._exit(0)


def write_json(path, value):
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        json.dump(value, handle, indent=2, allow_nan=False)
        handle.write("\n")
    temporary.replace(path)


def write_manifest(outdir, records):
    temporary = outdir / "manifest.csv.tmp"
    with temporary.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=MANIFEST_FIELDS, extrasaction="ignore")
        writer.writeheader()
        for record in records:
            writer.writerow({**record, "success": int(record["success"]), "valid": int(record["valid"])})
    temporary.replace(outdir / "manifest.csv")


class EpisodeWriter:
    def __init__(self, outdir, env_idx=0, initialize_files=True):
        self.outdir = Path(outdir)
        self.metadata = json.loads((self.outdir / "run_meta.json").read_text())
        self.env_idx = env_idx
        self.started_at = time.perf_counter()
        self.records = self.load_records()
        self.reset_times = []
        self.step_times = []
        self.video_encoding_times = []
        if initialize_files:
            self.write_snapshot()

    def load_records(self):
        path = self.outdir / "episodes.jsonl"
        if not path.exists():
            return []
        records = []
        for line in path.read_text(encoding="utf-8").splitlines():
            try:
                records.append(json.loads(line))
            except json.JSONDecodeError:
                continue
        return records

    def append(
        self,
        *,
        instance_id,
        success,
        task_progress,
        valid,
        sim_steps,
        duration_s=0.0,
        reset_duration_s=0.0,
        env_chunk_times_s=None,
        video_encoding_s=None,
        seed=None,
        terminated=None,
        truncated=None,
        native_termination=None,
        q_score=None,
        b1k_success=None,
        failure_reason=None,
        video_source=None,
    ):
        record = {
            "evaluation_mode": self.metadata.get("preset", "official-eval"),
            "task": self.metadata["task_name"],
            "env_idx": self.env_idx,
            "instance_id": instance_id,
            "seed": int(self.metadata.get("seed", 0)) if seed is None else seed,
            "terminated": terminated,
            "truncated": truncated,
            "native_termination": native_termination,
            "success": bool(success),
            "b1k_success": b1k_metrics.terminal_success(dict(native_termination=native_termination, b1k_success=b1k_success)),
            "q_score": q_score,
            "task_progress": task_progress,
            "valid": bool(valid),
            "episode_length": sim_steps,
            "sim_steps": sim_steps,
            "duration_s": duration_s,
            "reset_duration_s": reset_duration_s,
            "env_chunk_times_s": env_chunk_times_s or [],
            "video_encoding_s": video_encoding_s or [],
            "video_file": "",
            "finished_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        }
        if failure_reason is not None:
            record["failure_reason"] = failure_reason
        lock_path = self.outdir / ".episodes.lock"
        with lock_path.open("a") as lock_handle:
            fcntl.flock(lock_handle.fileno(), fcntl.LOCK_EX)
            self.records = self.load_records()
            layout = self.metadata.get('episode_layout')
            if layout:
                slot = layout[self.env_idx]
                number = slot['offset'] + sum(row['env_idx'] == self.env_idx for row in self.records)
                record['episode_in_slot'] = number + 1
                record['instance_cycle'] = number // len(slot['instances']) + 1
            record["episode_index"] = len(self.records) + 1
            if self.metadata.get("public_output_dir"):
                from groot_eval_output import update_results

                public_record = {
                    **record, "gpu_index": int(self.metadata["gpu_index"]),
                    "task_id": self.metadata["task_id"],
                }
                update_results(self.metadata["public_output_dir"], episode=public_record,
                               video_source=video_source)
                record["video_file"] = public_record["video_file"]
            elif video_source is not None:
                from groot_eval_output import name_episode_video

                record["video_file"] = name_episode_video(video_source, {
                    **record, "gpu_index": self.metadata.get("gpu_index", 0),
                })
            # The append-only record survives a later runner error or interrupted task.
            with (self.outdir / "episodes.jsonl").open("a", encoding="utf-8") as handle:
                handle.write(json.dumps(record, allow_nan=False) + "\n")
                handle.flush()
                os.fsync(handle.fileno())
            self.records.append(record)
            summary = self.write_snapshot()
            self.write_profile()
            fcntl.flock(lock_handle.fileno(), fcntl.LOCK_UN)
        print(
            f"EPISODE_COMPLETE mode={record['evaluation_mode']} episode={record['episode_index']} instance={instance_id} "
            f"valid={valid} success={success} q_score={q_score} task_progress={task_progress} "
            f"cumulative_success={summary['successful_episodes']}/{summary['completed_episodes']} "
            f"success_rate={summary['success_rate']} mean_q_score={summary['mean_q_score']} "
            f"q_score_episodes={summary['q_score_episodes']}/{summary['completed_episodes']}",
            flush=True,
        )

    def write_snapshot(self):
        valid_records = [record for record in self.records if record["valid"]]
        successes = [record["success"] for record in valid_records]
        progress = [record["task_progress"] for record in valid_records]
        known_progress = [value for value in progress if value is not None]
        q_scores = [record["q_score"] for record in valid_records if record.get("q_score") is not None]
        mode = self.metadata.get("preset", "official-eval")
        summary = {
            "evaluation_mode": mode,
            "official_episodes": len(valid_records) if mode == "official-eval" else 0,
            "task_id": self.metadata["task_id"],
            "task_name": self.metadata["task_name"],
            "requested_episodes": self.metadata["episodes"],
            "recorded_episodes": len(self.records),
            "completed_episodes": len(valid_records),
            "invalid_episodes": len(self.records) - len(valid_records),
            "successful_episodes": sum(successes),
            "successes": successes,
            "success_rate": sum(successes) / len(successes) if successes and mode == "official-eval" else None,
            "task_progress": progress,
            "mean_task_progress": sum(known_progress) / len(known_progress) if known_progress else None,
            "mean_q_score": sum(q_scores) / len(q_scores) if q_scores and mode == "official-eval" else None,
            "q_score_episodes": len(q_scores),
            "missing_q_score_episodes": len(valid_records) - len(q_scores),
            "b1k_metrics": b1k_metrics.summarize(self.records, mode, self.metadata["episodes"]),
            "status": "running",
            "runner_exit_code": None,
            "saved_video_count": 0,
            "log": "run.log",
            "manifest": "manifest.csv",
            "episodes": "episodes.jsonl",
        }
        failure = next((record["failure_reason"] for record in self.records if record.get("failure_reason")), None)
        if failure is not None:
            summary["failure_reason"] = failure
        write_manifest(self.outdir, self.records)
        write_json(self.outdir / "results.json", summary)
        return summary

    @staticmethod
    def _distribution(values):
        import numpy as np

        values = np.asarray(values, dtype=float)
        if values.size == 0:
            return {"count": 0, "mean_s": None, "p95_s": None}
        return {
            "count": int(values.size),
            "mean_s": float(values.mean()),
            "p95_s": float(np.percentile(values, 95)),
        }

    def write_profile(self, *, initialization_s=None, policy_times=None, vector_step_times=None, elapsed_s=None):
        valid = [record for record in self.records if record["valid"]]
        elapsed_s = elapsed_s or max(
            (time.perf_counter() - self.started_at) if hasattr(self, "started_at") else 0.0,
            1e-9,
        )
        reset_times = self.reset_times or [record["reset_duration_s"] for record in self.records]
        step_times = self.step_times or [
            value for record in self.records for value in record["env_chunk_times_s"]
        ]
        video_times = self.video_encoding_times or [
            value for record in self.records for value in record["video_encoding_s"]
        ]
        report = {
            "evaluation_mode": self.metadata.get("preset", "official-eval"),
            "environment_initialization": self._distribution(
                [] if initialization_s is None else [initialization_s]
            ),
            "reset": self._distribution(reset_times),
            "policy_get_action": self._distribution(policy_times or []),
            "env_step": self._distribution(vector_step_times or step_times),
            "environment_step_chunk": self._distribution(step_times),
            "video_encoding": self._distribution(video_times),
            "episode_total": self._distribution([record["duration_s"] for record in self.records]),
            "valid_episodes": len(valid),
            "elapsed_s": elapsed_s,
            "valid_episodes_per_hour": (
                len(valid) * 3600.0 / elapsed_s
                if self.metadata.get("preset", "official-eval") == "official-eval" else None
            ),
        }
        write_json(self.outdir / "profiling.json", report)
        return report

    def wrap_env_fns(self, env_fns):
        return [
            partial(record_episode_env, factory=factory, outdir=str(self.outdir), env_idx=env_idx)
            for env_idx, factory in enumerate(env_fns)
        ]


def finish_run(outdir, return_code, interrupted=False):
    """Update status without discarding any episodes saved by the child."""
    path = outdir / "results.json"
    writer = EpisodeWriter(outdir, initialize_files=False)
    writer.write_snapshot()
    summary = json.loads(path.read_text())
    complete = return_code == 0 and summary["completed_episodes"] == summary["requested_episodes"]
    summary.update(
        runner_exit_code=return_code,
        status="interrupted" if interrupted else ("complete" if complete else "failed"),
    )
    write_json(path, summary)
    return summary


def install_episode_recording(outdir):
    """Wrap the single-env factory, leaving the upstream rollout loop unchanged."""
    import gymnasium as gym
    global ACTIVE_WRITER
    writer = EpisodeWriter(outdir)
    ACTIVE_WRITER = writer
    vector_env = gym.vector.SyncVectorEnv

    def recorded_vector_env(env_fns, *args, **kwargs):
        return vector_env(
            writer.wrap_env_fns(env_fns),
            *args, **kwargs,
        )

    gym.vector.SyncVectorEnv = recorded_vector_env


def record_episode_env(factory, outdir, env_idx):
    import gymnasium as gym
    import numpy as np

    writer = EpisodeWriter(outdir, env_idx=env_idx, initialize_files=False)
    fixed_seed = writer.metadata.get("seed_mode", "sequence") == "fixed"
    base_seed = int(writer.metadata.get("seed", 0)) + (0 if fixed_seed else env_idx)
    layout = writer.metadata.get('episode_layout')
    slot = layout[env_idx] if layout else None
    episode_offset = slot['offset'] if slot else 0

    def episode_seed(number):
        number += episode_offset
        if writer.metadata.get("seed_mode") == "random":
            return int(np.random.SeedSequence([
                int(writer.metadata.get("seed", 0)), env_idx, number,
            ]).generate_state(1)[0])
        return base_seed + (0 if fixed_seed else number)

    quota = None
    if writer.metadata.get("limit_env_episodes"):
        n_envs = int(writer.metadata.get("n_envs", 1))
        # Match BEHAVIOR's contiguous partition: remainder belongs to last slot.
        quota = int(writer.metadata["episodes"]) // n_envs
        if env_idx == n_envs - 1:
            quota += int(writer.metadata["episodes"]) % n_envs
        if slot:
            quota = slot['quota']

    def seed_rng(seed):
        import random

        random.seed(seed)
        np.random.seed(seed % (2**32))
        try:
            import torch

            torch.manual_seed(seed)
        except ImportError:
            pass

    # Seed before construction as well as reset: some simulator state is sampled
    # while the environment and its task are being created.
    seed_rng(episode_seed(0))

    class RecordingEnv(gym.Wrapper):
        def __init__(self, env):
            super().__init__(env)
            from groot_live_preview import PreviewPublisher

            self.preview = PreviewPublisher(writer.outdir, writer.metadata, env_idx)

        def seal_episode_video(self):
            current = self.env
            while current is not None:
                state = vars(current)
                if "video_recorder" in state and state.get("file_path") is not None:
                    path = Path(state["file_path"])
                    try:
                        state["video_recorder"].stop()
                    except Exception as error:
                        print(f"VIDEO_CLOSE_ERROR source={path} error={error}", flush=True)
                        return None
                    # The result writer owns this closed file now. The next
                    # reset must not append another suffix or rename it again.
                    current.file_path = None
                    return path if path.is_file() else None
                current = state.get("env")
            return None

        def reset(self, **kwargs):
            if quota is not None and getattr(self, "valid_completed", 0) >= quota:
                if not hasattr(self, 'idle_observation'):
                    self.idle_observation = self.env.reset(**kwargs)[0]
                    self.idle_progress = 0.0
                return self.idle_observation, {}
            self.episode_seed = episode_seed(getattr(self, "episode_count", 0))
            seed_rng(self.episode_seed)
            if slot:
                current = self.env
                while current is not None:
                    state = vars(current)
                    if '_instance_indices_this_env' in state:
                        current._instance_idx_pointer = (episode_offset + getattr(self, 'episode_count', 0)) % len(slot['instances'])
                        break
                    current = state.get('env')
            previous_native = vars(self).get('native_env')
            if (os.environ.get('GROOT_TASK40_TRACE') == '1' and previous_native is not None
                    and writer.metadata['task_id'] == 40):
                from groot_task40_trace import append_reset_trace
                trace_root = Path(writer.metadata.get('public_output_dir') or writer.outdir) / 'stages'
                append_reset_trace(previous_native, trace_root, self.runtime_identity,
                                   writer.metadata.get('gpu_index', 0), None, boundary='before_reset')
            reset_started = time.perf_counter()
            result = self.env.reset(**kwargs)
            self.reset_duration_s = time.perf_counter() - reset_started
            self.success = False
            self.sim_steps = 0
            self.outer_steps = 0
            self.episode_chunk_times = []
            self.episode_started_at = time.perf_counter()
            self.episode_count = getattr(self, "episode_count", 0) + 1
            self.video_encoding_seen = 0
            self.video_encoding_samples = []
            current = self.env
            while current is not None:
                encode_times = getattr(current, "encode_times", None)
                if encode_times is not None:
                    self.video_encoding_seen = len(encode_times)
                    break
                current = vars(current).get("env")
            self.instance_id = None
            current = self.env
            while current is not None:
                state = vars(current)
                if "_instance_indices_this_env" in state:
                    index = state["_instance_indices_this_env"][state["_instance_idx_pointer"] - 1]
                    self.instance_id = int(state["_all_test_instances"][index])
                    break
                current = state.get("env")
            current = self.env
            self.native_env = None
            while current is not None:
                state = vars(current)
                if "_task" in state and "_current_step" in state:
                    self.native_env = current
                    break
                current = state.get("env")
            self.runtime_identity = {
                "instance_id": self.instance_id, "env_idx": env_idx,
                "seed": self.episode_seed, "episode_number": self.episode_count,
                "native_termination_config": (
                    self.native_env.task._termination_config if self.native_env is not None else None
                ),
            }
            if slot:
                self.runtime_identity.update(
                    episode_in_slot=episode_offset+self.episode_count,
                    instance_cycle=(episode_offset+self.episode_count-1)//len(slot['instances'])+1)
            write_json(writer.outdir / f"runtime_env{env_idx}.json", self.runtime_identity)
            if (os.environ.get('GROOT_TASK40_TRACE') == '1' and self.native_env is not None
                    and writer.metadata['task_id'] == 40):
                from groot_task40_trace import install_video_trace, append_reset_trace
                trace_root = Path(writer.metadata.get('public_output_dir') or writer.outdir) / 'stages'
                install_video_trace(self.env, trace_root, self.runtime_identity,
                                    writer.metadata.get('gpu_index', 0))
                append_reset_trace(self.native_env, trace_root, self.runtime_identity,
                                   writer.metadata.get('gpu_index', 0), result[0])
            self.preview.reset(self.runtime_identity)
            self.preview.publish(result[0], action_steps=0, outer_steps=0)
            return result

        def step(self, action):
            if quota is not None and getattr(self, "valid_completed", 0) >= quota:
                # A completed slot must not wrap its instance list while other
                # slots finish. Do not step/reset the simulator or emit results.
                return self.idle_observation, 0.0, False, False, {
                    # No episode ends here. Adding valid would create a key
                    # with a missing/None value for active debug episodes,
                    # which have not yet emitted native task validity.
                    "success": [False],
                    "task_progress": [self.idle_progress], "rewards": [],
                }
            step_started = time.perf_counter()
            fault_trace = os.environ.get('GROOT_ACTION_FAULT_TRACE') == '1' and self.native_env is not None
            if fault_trace:
                import omnigibson.object_states as fault_states
                from groot_task40_trace import read_state
                fault_before = read_state(self.native_env, {n: getattr(fault_states, n) for n in ('Open', 'ToggledOn', 'Inside', 'Cooked')})
            observation, reward, terminated, truncated, info = self.env.step(action)
            chunk_s = time.perf_counter() - step_started
            writer.step_times.append(chunk_s)
            self.episode_chunk_times.append(chunk_s)
            current = self.env
            while current is not None:
                encode_times = getattr(current, "encode_times", None)
                if encode_times is not None:
                    new_times = encode_times[self.video_encoding_seen :]
                    writer.video_encoding_times.extend(new_times)
                    self.video_encoding_samples.extend(new_times)
                    self.video_encoding_seen = len(encode_times)
                    break
                current = vars(current).get("env")
            self.success |= bool(np.any(info.get("success", False)))
            self.sim_steps += len(info.get("rewards", [reward]))
            self.outer_steps += 1
            if (os.environ.get('GROOT_TASK40_TRACE') == '1' and self.native_env is not None
                    and writer.metadata['task_id'] == 40):
                from groot_task40_trace import append_action
                action_root = Path(writer.metadata.get('public_output_dir') or writer.outdir) / 'stages'
                action_root.mkdir(parents=True, exist_ok=True)
                append_action(action_root / f"actions_task40_gpu{writer.metadata.get('gpu_index',0)}_env{env_idx}.jsonl",
                              self.runtime_identity, self.sim_steps-len(info.get('rewards',[reward])),self.sim_steps,action)
                if fault_trace:
                    after = read_state(self.native_env, {n: getattr(fault_states, n) for n in ('Open', 'ToggledOn', 'Inside', 'Cooked')})
                    record = dict(self.runtime_identity, action_start=self.sim_steps-len(info.get('rewards',[reward])),
                                  action_end=self.sim_steps, before=fault_before, after=after)
                    with (action_root / f"feedback_gpu{writer.metadata.get('gpu_index',0)}_env{env_idx}.jsonl").open('a') as stream:
                        stream.write(json.dumps(record)+'\n')
            self.preview.publish(observation, action_steps=self.sim_steps, outer_steps=self.outer_steps,
                                 terminal=bool(terminated or truncated))
            if (os.environ.get("GROOT_TASK40_TRACE") == "1" and self.native_env is not None
                    and writer.metadata["task_id"] == 40
                    and (self.outer_steps % int(os.environ.get("GROOT_TASK40_TRACE_STRIDE", "10")) == 0 or terminated or truncated)):
                from groot_task40_trace import append_trace
                trace_root = Path(writer.metadata.get("public_output_dir") or writer.outdir) / "stages"
                trace_root.mkdir(parents=True, exist_ok=True)
                append_trace(self.native_env, trace_root / f"task40_gpu{writer.metadata.get('gpu_index', 0)}_env{env_idx}.jsonl",
                             self.runtime_identity, self.sim_steps)
            if self.outer_steps % 100 == 0 or terminated or truncated:
                progress_values = np.asarray(info.get("task_progress", [])).reshape(-1)
                write_json(writer.outdir / f"runtime_env{env_idx}.json", {
                    **self.runtime_identity, "action_steps": self.sim_steps,
                    "outer_steps": self.outer_steps,
                    "elapsed_s": time.perf_counter() - self.episode_started_at,
                    "success": self.success,
                    "task_progress": float(progress_values[-1]) if progress_values.size else None,
                    "terminated": bool(terminated), "truncated": bool(truncated),
                })
            if terminated or truncated:
                valid = bool(np.all(info.get("valid", True)))
                failure_reason = None
                if not valid:
                    current = self.env
                    physx_crashed = False
                    while current is not None:
                        state = vars(current)
                        physx_crashed |= bool(state.get("_physx_crashed", False))
                        current = state.get("env")
                    failure_reason = {
                        "code": "unrecoverable_physx_error" if physx_crashed else "invalid_episode",
                        "message": (
                            "PhysX 环境已不可恢复；停止评估，禁止在损坏环境中重试。"
                            if physx_crashed else
                            "环境返回无效回合；停止评估，保留记录并等待检查。"
                        ),
                        "gpu_index": writer.metadata.get("gpu_index"),
                        "task_id": writer.metadata["task_id"],
                        "env_idx": env_idx, "instance_id": self.instance_id,
                        "seed": self.episode_seed, "sim_steps": self.sim_steps,
                    }
                values = np.asarray(info.get("task_progress", [])).reshape(-1)
                progress = float(values[-1]) if values.size else None
                # Match rollout_policy's terminal chunk extraction. Q is provided
                # by the native metric at termination, never inferred from progress.
                q_values = np.asarray(info.get("q_score", []), dtype=float).reshape(-1)
                q_score = float(np.max(q_values)) if q_values.size else None
                # Official evaluator checks terminal info["done"]["success"].
                # Preserve GR00T's ever-success flag separately above.
                terminal_flags = np.asarray(info.get("success", [])).reshape(-1)
                b1k_success = bool(terminal_flags[-1]) if terminal_flags.size else None
                writer.append(
                    instance_id=self.instance_id,
                    success=self.success,
                    task_progress=progress,
                    q_score=q_score,
                    b1k_success=b1k_success,
                    valid=valid,
                    failure_reason=failure_reason,
                    sim_steps=self.sim_steps,
                    duration_s=time.perf_counter() - self.episode_started_at,
                    reset_duration_s=self.reset_duration_s,
                    env_chunk_times_s=self.episode_chunk_times,
                    video_encoding_s=self.video_encoding_samples,
                    seed=self.episode_seed,
                    terminated=bool(terminated),
                    truncated=bool(truncated),
                    native_termination=(
                        self.native_env.task._info.get("done") if self.native_env is not None else None
                    ),
                    video_source=self.seal_episode_video(),
                )
                if quota is not None and bool(np.all(info.get("valid", True))):
                    self.valid_completed = getattr(self, "valid_completed", 0) + 1
                    self.idle_observation = observation
                    self.idle_progress = progress
            return observation, reward, terminated, truncated, info

        def close(self):
            try:
                self.env.close()
            finally:
                self.preview.close()

    environment = factory()
    current = environment
    while current is not None:
        if vars(current).get('task_name') == 'make_microwave_popcorn':
            from groot_scene_reset import install_scene_reset
            install_scene_reset(current)
            break
        current = vars(current).get('env')
    return RecordingEnv(environment)


def get_active_writer():
    return ACTIVE_WRITER
