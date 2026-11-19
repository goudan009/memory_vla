"""Hold diagnostic workers until both can start their measured window."""

import os
import json
from pathlib import Path
import time

from groot_eval_results import write_json


def wait_for_pair(audit=None):
    location = os.environ.get("GROOT_EVAL_BARRIER_DIR")
    if not location:
        return
    directory = Path(location)
    gpu = os.environ["OMNIGIBSON_GPU_ID"]
    write_json(directory / f"gpu{gpu}.ready.json", {
        "pid": os.getpid(), "gpu": gpu,
        "output": os.environ["GROOT_EVAL_OUTPUT_DIR"],
        "audit_config": audit.config if audit is not None else json.loads(
            (Path(os.environ["GROOT_EVAL_OUTPUT_DIR"]) / "runtime_env0.json").read_text()
        ),
    })
    # Do not call app.update while waiting: no extra physical or render steps.
    while not (directory / "release.json").exists():
        time.sleep(0.05)
    if audit is not None:
        audit.initialized = time.time()
