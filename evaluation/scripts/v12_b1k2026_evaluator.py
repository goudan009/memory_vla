#!/usr/bin/env python3
"""Preserve the B1K-2026 transport dependency before Isaac Kit changes sys.path."""

import json
import os
from pathlib import Path
import runpy
import sys


def preload_transport():
    # Kit's pip archive contains websockets 12. Import the B1K-compatible
    # implementation before SimulationApp inserts that archive into sys.path.
    import websockets
    import websockets.asyncio.server
    import websockets.sync.client

    return {"websockets": websockets.__version__, "path": websockets.__file__}


def preload_cffi():
    # B1K pins cffi==1.17.1. Reuse Isaac's matching Python/binary pair before
    # another library imports a backend from the shared dependency directory.
    isaac = Path(os.environ["ISAAC_PATH"])
    archive = isaac / "extscache" / "omni.kit.pip_archive-0.0.0+69cbf6ad.lx64.cp311" / "pip_prebundle"
    sys.path.insert(0, str(archive))
    try:
        import cffi
        import _cffi_backend

        cffi.FFI()
        if cffi.__version__ != "1.17.1" or _cffi_backend.__version__ != "1.17.1":
            raise RuntimeError("B1K-2026 requires a matched cffi 1.17.1 Python/binary pair")
        return {"cffi": cffi.__version__, "path": cffi.__file__, "backend": _cffi_backend.__file__}
    finally:
        sys.path.remove(str(archive))


def main():
    print("V1.2 evaluator CFFI: " + json.dumps(preload_cffi()), flush=True)
    print("V1.2 evaluator transport: " + json.dumps(preload_transport()), flush=True)
    runpy.run_module("omnigibson.eval.eval", run_name="__main__", alter_sys=True)


if __name__ == "__main__":
    main()
