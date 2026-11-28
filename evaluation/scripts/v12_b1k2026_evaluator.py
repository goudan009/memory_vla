#!/usr/bin/env python3
"""Preserve the B1K-2026 transport dependency before Isaac Kit changes sys.path."""

import json
import runpy


def preload_transport():
    # Kit's pip archive contains websockets 12. Import the B1K-compatible
    # implementation before SimulationApp inserts that archive into sys.path.
    import websockets
    import websockets.asyncio.server
    import websockets.sync.client

    return {"websockets": websockets.__version__, "path": websockets.__file__}


def main():
    print("V1.2 evaluator transport: " + json.dumps(preload_transport()), flush=True)
    runpy.run_module("omnigibson.eval.eval", run_name="__main__", alter_sys=True)


if __name__ == "__main__":
    main()
