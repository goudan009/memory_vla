"""Explicit current state / previously executed command contract.

No future q, action[t], task_info, contact truth, camera pose or global position
is added to V2. Demonstration action[t-1] is a recorded controller command,
not an observed achieved displacement and not a model's unexecuted chunk.
"""
import json
from pathlib import Path
import numpy as np
import torch

from .kinematics import RobotGraph, rpy_matrix


MAPPING = Path(__file__).parent / "configs/r1pro_mapping.json"


class BodyMapper:
    def __init__(self, urdf, mapping=MAPPING):
        self.spec = json.loads(Path(mapping).read_text())
        self.graph = RobotGraph(urdf, self.spec.get("extra_fixed", []))
        self.mapping = str(mapping)
        groups = self.spec["groups"]
        self.entries = [(self.graph.index[name], g["state"], g["velocity"], k, og, action)
                        for g in groups for k,(name,og,action) in enumerate(zip(g["joints"],g["og_indices"],g["action_indices"]))]
        if len({e[0] for e in self.entries}) != len(self.entries): raise ValueError("Duplicate joint mapping")

    def observation(self, fields, previous_action=None, timestamp=0., previous_action_time=None):
        n = len(self.graph.names)
        q, vel, qm, vm, pa, am = [np.zeros(n, np.float32) for _ in range(6)]
        qm[0] = 1
        for i,j in enumerate(self.graph.joints,1):
            if j.kind == "fixed": qm[i]=1
        prev_valid = previous_action is not None
        if prev_valid:
            previous_action = np.asarray(previous_action, np.float32)
            if previous_action.shape != (23,) or not np.isfinite(previous_action).all():
                raise ValueError("Executed command must be finite [23]")
            if previous_action_time is None or previous_action_time >= timestamp:
                raise ValueError("Executed command must precede current observation")
        for i,state,velocity,k,_,act in self.entries:
            if state not in fields: raise ValueError(f"Missing current state: {state}")
            q[i],qm[i] = np.asarray(fields[state]).reshape(-1)[k],1
            if velocity in fields:
                vel[i],vm[i] = np.asarray(fields[velocity]).reshape(-1)[k],1
            if prev_valid: pa[i],am[i] = previous_action[act],1
        # Existing robot velocity is world-oriented; rotate using measured
        # proprioceptive orientation already present in V1. No world position.
        local_velocity = np.zeros(6,np.float32)
        if all(k in fields for k in ("robot_lin_vel","robot_ang_vel","robot_ori_sin","robot_ori_cos")):
            angles = np.arctan2(fields["robot_ori_sin"],fields["robot_ori_cos"])
            rotation = rpy_matrix(torch.tensor(angles,dtype=torch.float32)).numpy()
            local_velocity[:3] = rotation.T @ np.asarray(fields["robot_lin_vel"]).reshape(3)
            local_velocity[3:] = rotation.T @ np.asarray(fields["robot_ang_vel"]).reshape(3)
        result = dict(q=q,qvel=vel,q_mask=qm,velocity_mask=vm,previous_action=pa,action_mask=am,
                      base=np.r_[local_velocity, previous_action[:3] if prev_valid else np.zeros(3),float(prev_valid)].astype(np.float32),
                      timestamp=np.array(timestamp,np.float64))
        if not all(np.isfinite(v).all() for v in result.values()): raise ValueError("Nonfinite Body observation")
        return result

    def episode(self, dataframe, modality, frames, raw_dataframe=None):
        """Use raw 256 proprio if supplied, else packed q + masked velocity.

        Velocity fallback is backward difference at full 30Hz, never central
        differences. State[t] is the recorded result of action[t-1]; t=0 has no
        prior command. Optional raw source must match episode/times/actions/q.
        """
        state = np.stack(dataframe["observation.state"]).astype(np.float32)
        actions = np.stack(dataframe["action"]).astype(np.float32)
        times = np.asarray(dataframe["timestamp"], np.float64)
        if len(set(dataframe.episode_index)) != 1: raise ValueError("Mixed episodes")
        np.testing.assert_allclose(times, np.arange(len(times))/30, atol=2e-5)
        if state.shape[1] != 82 or actions.shape != (len(state),23): raise ValueError("Expected V1 packed dataset")
        fields = {k:state[:,s["start"]:s["end"]] for k,s in modality["state"].items()}
        raw = None
        if raw_dataframe is not None:
            raw = np.stack(raw_dataframe["observation.state"])
            if raw.shape != (len(state),256): raise ValueError("Expected raw B1K 256 proprio")
            np.testing.assert_array_equal(raw_dataframe.episode_index,dataframe.episode_index)
            np.testing.assert_allclose(raw_dataframe.timestamp,times,atol=1e-8)
            np.testing.assert_array_equal(np.stack(raw_dataframe.action),actions)
            for g in self.spec["groups"]:
                np.testing.assert_allclose(raw[:,g["og_indices"]],fields[g["state"]],atol=2e-6)
                fields[g["velocity"]] = raw[:,np.array(g["og_indices"])+84]
        results = []
        for frame in frames:
            if not 0 <= frame < len(state): raise IndexError(frame)
            current = {k:v[frame].copy() for k,v in fields.items()}
            if raw is None and frame>0:
                for g in self.spec["groups"]:
                    delta = fields[g["state"]][frame]-fields[g["state"]][frame-1]
                    current[g["velocity"]] = delta / (times[frame]-times[frame-1])
            results.append(self.observation(current,actions[frame-1] if frame else None,
                float(times[frame]),float(times[frame-1]) if frame else None))
        return {k:torch.from_numpy(np.stack([r[k] for r in results])) for k in results[0]}


def slice_body(body, start, end, device):
    return {k:v[start:end].to(device) for k,v in body.items()}
