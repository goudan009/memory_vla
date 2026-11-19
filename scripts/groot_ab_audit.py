"""Opt-in, read-only A/B evidence, collected outside policy/env timing scopes."""

import hashlib
import json
import os
from pathlib import Path
import time

import numpy as np

from groot_eval_results import write_json


def describe_arrays(values):
    result = {}
    for key, value in values.items():
        array = np.asarray(value)
        raw = json.dumps(array.tolist(), ensure_ascii=False).encode() if array.dtype.kind in "OUS" else array.tobytes()
        result[key] = {
            "shape": list(array.shape), "dtype": str(array.dtype),
            "sha256": hashlib.sha256(raw).hexdigest(),
            "finite": bool(np.isfinite(array).all()) if array.dtype.kind in "biufc" else True,
        }
    return result


def render_inventory():
    import omnigibson as og
    import omnigibson.lazy as lazy
    from omnigibson.sensors.vision_sensor import VisionSensor

    sensors = []
    for sensor in VisionSensor.SENSORS.values():
        sensors.append({
            "name": sensor.name, "camera_path": sensor.prim_path,
            "resolution": [sensor.image_width, sensor.image_height],
            "annotator_render_product": str(sensor._render_product.path),
            "annotators": {key: str(value.get_node().get_prim_path()) for key, value in sensor._annotators.items() if value is not None},
            "viewport": sensor._viewport.name,
            "viewport_render_product": str(sensor._viewport.viewport_api.render_product_path),
            "camera_parameters": {name: str(sensor.prim.GetAttribute(name).Get()) for name in (
                "focalLength", "horizontalAperture", "verticalAperture", "clippingRange", "projection", "focusDistance", "fStop",
            )},
        })
    products = []
    for prim in lazy.pxr.Usd.PrimRange(og.sim.stage.GetPrimAtPath("/Render")):
        if prim.GetTypeName() == "RenderProduct":
            path = str(prim.GetPath())
            products.append({
                "path": path,
                "cameras": [str(target) for target in prim.GetRelationship("camera").GetTargets()],
                "resolution": str(prim.GetAttribute("resolution").Get()),
                "ordered_vars": [str(target) for target in prim.GetRelationship("orderedVars").GetTargets()],
                "annotator_owners": [s["name"] for s in sensors if s["annotator_render_product"] == path],
                "viewport_owners": [s["name"] for s in sensors if s["viewport_render_product"] == path],
            })
    viewports = [{"name": window.name, "visible": window.visible,
                  "updates_enabled": window.viewport_api.updates_enabled,
                  "render_product": str(window.viewport_api.render_product_path),
                  "camera": str(window.viewport_api.camera_path)}
                 for window in lazy.omni.kit.viewport.window.get_viewport_window_instances()]
    return {"sensors": sensors, "render_products": products, "viewports": viewports,
            "skip_while_invisible": lazy.carb.settings.get_settings().get('/app/renderer/skipWhileInvisible')}


class ABAudit:
    def __init__(self, env, policy, observations):
        import omnigibson as og

        started = time.perf_counter()
        self.outdir = Path(os.environ["GROOT_EVAL_OUTPUT_DIR"])
        self.client_id = self.outdir.name + f":pid{os.getpid()}"
        self.rows = []
        self.audit_s = 0.0
        self.initialized = time.time()
        current = env.envs[0]
        while type(current).__name__ != "BEHAVIORGr00tEnv":
            current = vars(current)["env"]
        self.behavior = current
        self.instance_id = int(current._all_test_instances[current._instance_indices_this_env[current._instance_idx_pointer - 1]])
        self.inventory = render_inventory()
        self.config = {
            "instance_id": self.instance_id,
            "task": current.task_name,
            "seed": int(os.environ["GROOT_EVAL_SEED"]),
            "native_termination_config": current.env.task._termination_config,
            "physics_dt": og.sim.get_physics_dt(),
            "rendering_dt": og.sim.get_rendering_dt(),
            "sim_step_dt": og.sim.get_sim_step_dt(),
            "robot_sensors": list(current.robot.sensors),
            "action_space": str(current.action_space),
            "socket_type": int(policy.socket.getsockopt(__import__("zmq").TYPE)),
            "policy_host": policy.host, "policy_port": policy.port,
        }
        np.savez_compressed(self.outdir / "initial_policy_observations.npz", **observations)
        initial_state = og.sim.dump_state(serialized=True).cpu().numpy()
        np.save(self.outdir / "initial_sim_state.npy", initial_state)
        write_json(self.outdir / "semantic_initial.json", {
            "client_id": self.client_id, "config": self.config,
            "observations": describe_arrays(observations),
            "sim_state": describe_arrays({"state": initial_state}),
            "inventory": self.inventory,
        })
        self.handle = (self.outdir / "policy_requests.jsonl").open("w", buffering=1)
        self.initial_audit_s = time.perf_counter() - started
        self.initialized = time.time()

    def cycle(self, index, observations, actions, next_obs, infos, terminated, truncated):
        started = time.perf_counter()
        row = {
            "client_id": self.client_id, "request_index": index,
            "instance_id": self.instance_id,
            "observations": describe_arrays(observations),
            "actions": describe_arrays(actions),
            "next_observations": describe_arrays(next_obs),
            "physx_crashed": bool(self.behavior._physx_crashed),
            "terminated": np.asarray(terminated).tolist(),
            "truncated": np.asarray(truncated).tolist(),
            "task_progress": np.asarray(infos.get("task_progress", [])).tolist(),
            "valid": np.asarray(infos.get("valid", [])).tolist(),
        }
        self.handle.write(json.dumps(row, allow_nan=False) + "\n")
        self.rows.append(row)
        self.audit_s += time.perf_counter() - started

    def snapshot(self):
        return {
            "client_id": self.client_id, "config": self.config, "inventory": self.inventory,
            "completed_request_response_pairs": len(self.rows),
            "all_finite": all(item["finite"] for row in self.rows for group in ("observations", "actions", "next_observations") for item in row[group].values()),
            "physx_crash_cycles": sum(row["physx_crashed"] for row in self.rows),
            "audit_s_outside_cycle_timers": self.audit_s,
            "initial_audit_s_outside_cycle_timers": self.initial_audit_s,
            "wall_start_unix": self.initialized, "wall_end_unix": time.time(),
            "completed_episodes_per_hour": None,
            "episode_rate_note": "diagnostic window, unfinished episode is not an episode throughput measurement",
        }
