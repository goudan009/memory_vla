"""Episode-scoped options for independent evaluation clients; observations stay intact."""

import uuid
import json
import os
from pathlib import Path


class EpisodePolicy:
    def __init__(self, policy, n_envs, session_id=None):
        self.policy = policy
        self.session_id = session_id or uuid.uuid4().hex
        self.episodes = [0] * n_envs
        self.decisions = [0] * n_envs
        self.memory_trace = None
        if os.environ.get('GROOT_TASK40_TRACE') == '1' and os.environ.get('GROOT_EVAL_OUTPUT_DIR'):
            runtime = Path(os.environ['GROOT_EVAL_OUTPUT_DIR'])
            metadata = json.loads((runtime/'run_meta.json').read_text())
            if metadata.get('task_id') == 40:
                self.memory_trace = (runtime,metadata)
        # The fixed GR00T PolicyClient stores timeout_ms but never applies it
        # to ZMQ. Bound the first reset and subsequent requests as intended.
        socket = getattr(policy, "socket", None)
        timeout_ms = getattr(policy, "timeout_ms", None)
        if socket is not None and timeout_ms is not None:
            import zmq

            socket.setsockopt(zmq.RCVTIMEO, timeout_ms)
            socket.setsockopt(zmq.SNDTIMEO, timeout_ms)
            socket.setsockopt(zmq.LINGER, 0)
            print(f"POLICY_RPC_TIMEOUT host={policy.host} port={policy.port} timeout_ms={timeout_ms}", flush=True)

    def _options(self, env_ids):
        return {
            "session_id": self.session_id,
            "env_ids": list(env_ids),
            "episode_ids": [f"{self.session_id}/env{i}/episode{self.episodes[i]}" for i in env_ids],
        }

    def reset(self):
        return self.policy.reset(options=self._options(range(len(self.episodes))))

    def reset_ended(self, ended):
        env_ids = [i for i, done in enumerate(ended) if done]
        if not env_ids:
            return
        for i in env_ids:
            self.episodes[i] += 1
            self.decisions[i] = 0
        return self.policy.reset(options=self._options(env_ids))

    def get_action(self, observations):
        options = self._options(range(len(self.episodes)))
        capture = self.memory_trace is not None and any(t % 10 == 0 for t in self.decisions)
        if capture:
            options['memory_diagnostics'] = True
        if self.memory_trace is not None and any(t % 100 == 0 for t in self.decisions):
            options['memory_action_controls'] = True
        result = self.policy.get_action(observations, options=options)
        diagnostics = result[1].get('memory_diagnostics') if capture else None
        if diagnostics:
            runtime,metadata = self.memory_trace
            root = Path(metadata.get('public_output_dir') or runtime)/'stages'
            root.mkdir(parents=True,exist_ok=True)
            for env_idx,memory in enumerate(diagnostics.get('observations',[])):
                identity_path = runtime/f'runtime_env{env_idx}.json'
                identity = json.loads(identity_path.read_text()) if identity_path.exists() else {}
                record = dict(identity,schema_version=1,env_idx=env_idx,
                    session_id=self.session_id,episode_id=options['episode_ids'][env_idx],
                    decision_index=self.decisions[env_idx],memory=memory,
                    history_mode=diagnostics.get('history_mode'),
                    action_controls=result[1].get('memory_action_controls'),
                    coverage='每10个策略决策采样；decision_index不是仿真动作步或视频精确时间')
                with (root/f"memory_gpu{metadata.get('gpu_index',0)}_env{env_idx}.jsonl").open('a') as stream:
                    stream.write(json.dumps(record,ensure_ascii=False)+'\n')
        self.decisions = [t+1 for t in self.decisions]
        return result

    def __getattr__(self, name):
        return getattr(self.policy, name)
