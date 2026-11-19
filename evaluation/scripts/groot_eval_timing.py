"""Stage and simulator-step tracing for GR00T evaluation diagnostics."""

from contextlib import contextmanager
from functools import wraps
import os
import time


_SIM_STEP_CALLS = 0
_DETAIL_ENABLED = os.environ.get("GROOT_EVAL_DETAIL_PROFILE") == "1"
_ACTIVE = False
_STACK = []
_SAMPLES = {}
_POSITION = 0
_COUNTERS = {}
_METADATA = {}
_CAMERA_NODES = None
_CAMERA_RENDER_SUBSCRIPTION = None
_CAMERA_RENDER_EXAMPLES = {}


def _audit_render_event(event):
    kind = str(event.type)
    profile_count("camera_audit.render_event." + kind)
    examples = _CAMERA_RENDER_EXAMPLES.setdefault(kind, [])
    if _ACTIVE and len(examples) < 16:
        examples.append({str(key): str(value) for key, value in event.payload.get_dict().items()})
        profile_metadata("render_event_examples." + kind, list(examples))


def _start_camera_counters():
    """Read existing graph counters only; do not attach annotators or evaluate nodes."""
    global _CAMERA_NODES, _CAMERA_RENDER_SUBSCRIPTION
    if _CAMERA_NODES is not None or os.environ.get("GROOT_EVAL_CAMERA_COUNTERS") != "1":
        return
    from omnigibson.sensors.vision_sensor import VisionSensor
    import omnigibson.lazy as lazy

    _CAMERA_NODES = []
    for sensor in VisionSensor.SENSORS.values():
        for modality, annotator in (sensor._annotators or {}).items():
            if annotator is None:
                continue
            node = annotator.get_node()
            _CAMERA_NODES.append((
                sensor.name, modality, node, node.get_compute_count(),
                sensor.image_width, sensor.image_height,
            ))
    _CAMERA_RENDER_SUBSCRIPTION = lazy.omni.usd.get_context().get_rendering_event_stream().create_subscription_to_pop(
        _audit_render_event, name="groot_camera_graph_audit",
    )


def profile_count(name, amount=1):
    if _ACTIVE:
        _COUNTERS[name] = _COUNTERS.get(name, 0) + amount


def profile_metadata(name, value):
    if _ACTIVE:
        _METADATA[name] = value


def start_detail_cycle(active):
    """Limit detail sampling to the rollout window; never sample initialization/reset."""
    global _ACTIVE, _POSITION
    _ACTIVE = bool(active and _DETAIL_ENABLED)
    _POSITION = 0
    if _ACTIVE:
        _start_camera_counters()


def set_chunk_position(position):
    global _POSITION
    _POSITION = position


@contextmanager
def profile_scope(name):
    """Host wall time; self time subtracts instrumented direct children only."""
    if not _ACTIVE:
        yield
        return
    frame = [time.perf_counter(), 0.0]
    position = _POSITION
    _STACK.append(frame)
    try:
        yield
    finally:
        elapsed = time.perf_counter() - frame[0]
        _STACK.pop()
        if _STACK:
            _STACK[-1][1] += elapsed
        _SAMPLES.setdefault((name, position), []).append((elapsed, elapsed - frame[1]))


def profiled(name):
    def decorate(function):
        @wraps(function)
        def wrapped(*args, **kwargs):
            with profile_scope(name):
                return function(*args, **kwargs)
        return wrapped
    return decorate


def detail_snapshot():
    """Inclusive parents overlap children. Only self totals form an additive partition."""
    import numpy as np

    def summarize(samples):
        array = np.asarray(samples, dtype=float)
        return {
            "count": len(samples),
            "inclusive_mean_s": float(array[:, 0].mean()),
            "inclusive_p95_s": float(np.percentile(array[:, 0], 95)),
            "inclusive_total_s": float(array[:, 0].sum()),
            "self_mean_s": float(array[:, 1].mean()),
            "self_p95_s": float(np.percentile(array[:, 1], 95)),
            "self_total_s": float(array[:, 1].sum()),
        }

    grouped = {}
    by_position = {}
    for (name, position), samples in _SAMPLES.items():
        grouped.setdefault(name, []).extend(samples)
        by_position.setdefault(str(position), {})[name] = summarize(samples)
    camera_nodes = []
    for camera, modality, node, before, width, height in _CAMERA_NODES or []:
        after = node.get_compute_count()
        camera_nodes.append({
            "camera": camera, "modality": modality, "node": str(node.get_prim_path()),
            "compute_count_before": before, "compute_count_after": after,
            "compute_count_delta": after - before if after >= before else None,
            "width": width, "height": height,
        })
    return {
        "enabled": _DETAIL_ENABLED,
        "clock": "perf_counter host wall time, no added GPU synchronization",
        "stages": {name: summarize(samples) for name, samples in grouped.items()},
        "by_chunk_position": by_position,
        "counters": dict(_COUNTERS),
        "metadata": dict(_METADATA),
        "camera_graph_counters": camera_nodes,
    }


def trace_event(stage, event, **fields):
    if os.environ.get("GROOT_EVAL_TRACE") != "1":
        return
    process_started = float(os.environ.get("GROOT_EVAL_PROCESS_STARTED_AT", "0") or 0)
    since_process = time.perf_counter() - process_started if process_started else None
    values = [
        f"wall_unix={time.time():.6f}",
        f"mono={time.perf_counter():.6f}",
        f"pid={os.getpid()}",
        f"stage={stage}",
        f"event={event}",
    ]
    if since_process is not None:
        values.append(f"since_process_s={since_process:.3f}")
    values.extend(f"{key}={value}" for key, value in fields.items())
    print("GROOT_TRACE " + " ".join(values), flush=True)


@contextmanager
def timed_stage(stage):
    started = time.perf_counter()
    trace_event(stage, "begin")
    try:
        yield
    finally:
        trace_event(stage, "end", elapsed_s=f"{time.perf_counter() - started:.6f}")


def count_sim_step():
    global _SIM_STEP_CALLS
    _SIM_STEP_CALLS += 1


def sim_step_count():
    return _SIM_STEP_CALLS
