"""On-demand snapshots of existing policy observations; never render or step."""

import base64
import json
from pathlib import Path
import re
import time


CAMERAS = (
    ("head", "头部", "video.observation.images.rgb.head_256_256"),
    ("left_wrist", "左腕", "video.observation.images.rgb.left_wrist_256_256"),
    ("right_wrist", "右腕", "video.observation.images.rgb.right_wrist_256_256"),
)
LEASE_SECONDS = 15
INTERVAL_SECONDS = 1.0


def preview_directory(run):
    return Path(run) / ".runtime" / "preview"


def write_snapshot(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False), encoding="utf-8")
    temporary.replace(path)


def subscribe(run, client, enabled, now=None):
    if not isinstance(client, str) or not re.fullmatch(r"[A-Za-z0-9_-]{8,100}", client):
        raise ValueError("Invalid preview client")
    lease = preview_directory(run) / "requests" / f"{client}.json"
    if enabled:
        write_snapshot(lease, {"expires_at": (time.time() if now is None else now) + LEASE_SECONDS})
    else:
        lease.unlink(missing_ok=True)


def requested(run, now=None):
    now = time.time() if now is None else now
    for lease in (preview_directory(run) / "requests").glob("*.json"):
        try:
            if json.loads(lease.read_text())["expires_at"] > now:
                return True
        except (OSError, ValueError, KeyError):
            continue
    return False


def read_preview(run):
    environments = []
    for path in sorted((preview_directory(run) / "frames").glob("*.json")):
        try:
            environments.append(json.loads(path.read_text(encoding="utf-8")))
        except (OSError, ValueError):
            continue
    return {"supported": bool(environments), "environments": environments, "server_time": time.time()}


class PreviewPublisher:
    def __init__(self, outdir, metadata, env_idx):
        self.run = Path(metadata.get("public_output_dir") or outdir)
        self.gpu = int(metadata.get("gpu_index", 0))
        # One current snapshot per GPU/environment; resumes replace old slots.
        self.path = preview_directory(self.run) / "frames" / f"gpu{self.gpu}_env{env_idx}.json"
        self.state = {"gpu_index": self.gpu, "env_idx": env_idx, "task_id": metadata["task_id"],
                      "status": "waiting", "captured_at": None, "cameras": []}
        self.last_check = float("-inf")
        self.last_frame = float("-inf")
        self.enabled = False
        self._save()

    def _save(self):
        try:
            write_snapshot(self.path, self.state)
        except OSError as error:
            print(f"PREVIEW_WRITE_ERROR path={self.path} error={error}", flush=True)

    def reset(self, identity):
        self.state.update({key: identity.get(key) for key in ("instance_id", "seed", "episode_number")})
        self.state.update(status="waiting", action_steps=0, outer_steps=0, captured_at=None, cameras=[])
        self.last_frame = float("-inf")
        self._save()

    def publish(self, observation, *, action_steps, outer_steps, terminal=False):
        now = time.monotonic()
        if now - self.last_check >= INTERVAL_SECONDS:
            self.enabled = requested(self.run)
            self.last_check = now
        if not self.enabled or now - self.last_frame < INTERVAL_SECONDS:
            return
        self.last_frame = now
        try:
            self._publish_frame(observation, action_steps, outer_steps, terminal)
        except Exception as error:
            # Display-only failures must not interrupt an evaluation episode.
            if self.state.get("status") != "error":
                print(f"PREVIEW_ENCODE_ERROR path={self.path} error={error}", flush=True)
            self.state.update(status="error", cameras=[], captured_at=None)
            self._save()

    def _publish_frame(self, observation, action_steps, outer_steps, terminal):
        # Imports, array copies and JPEG encoding happen only with a viewer.
        import cv2
        import numpy as np

        cameras = []
        for name, label, key in CAMERAS:
            if key not in observation:
                continue
            rgb = np.asarray(observation[key])
            if rgb.ndim == 4:
                rgb = rgb[-1]  # MultiStepWrapper history ends at delta index 0.
            if rgb.ndim != 3 or rgb.shape[-1] != 3 or rgb.dtype != np.uint8:
                continue
            ok, encoded = cv2.imencode(".jpg", cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR),
                                       [cv2.IMWRITE_JPEG_QUALITY, 80])
            if ok:
                cameras.append({"name": name, "label": label,
                                "src": "data:image/jpeg;base64," + base64.b64encode(encoded).decode("ascii")})
        self.state.update(cameras=cameras, captured_at=time.time(), action_steps=action_steps,
                          outer_steps=outer_steps, status="terminal" if terminal else "running")
        self._save()

    def close(self):
        self.state["status"] = "closed"
        self._save()
