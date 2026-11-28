#!/usr/bin/env python3
"""Isolate B1K-2026 Python imports across Isaac Kit startup."""

from functools import wraps
from importlib.machinery import PathFinder
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


def preserve_numba_imports(finder):
    original = finder.find_spec

    @classmethod
    def find_spec(cls, fullname, path=None, target=None):
        # Kit's namespace resolution must remain active for omni/isaacsim.
        # This compiled dependency pair instead uses the loaded parent path.
        if path is not None and fullname.split(".", 1)[0] in {"numba", "llvmlite"}:
            return PathFinder.find_spec(fullname, path, target)
        return original(fullname, path, target)

    finder.find_spec = find_spec


def configure_simulation_app(output):
    import isaacsim
    import numba
    import llvmlite
    from omni.ext._impl.fast_importer import FastFinder

    preserve_numba_imports(FastFinder)

    original = isaacsim.SimulationApp

    @wraps(original)
    def simulation_app(config=None, *args, **kwargs):
        app = original(config, *args, **kwargs)
        import carb.settings

        enabled = carb.settings.get_settings().get("/app/extensions/fastImporter/enabled")
        # Exercise the same lazy Numba runtime used by the controller after
        # real Kit initialization, not just an approximation of sys.path.
        import numpy as np
        import numba
        import numba.core.runtime.context
        import llvmlite

        @numba.njit
        def copy_array(value):
            return value.copy()

        np.testing.assert_array_equal(copy_array(np.arange(3.0)), np.arange(3.0))
        evidence = {
            "fast_importer_enabled": enabled,
            "numba": numba.__version__, "numba_path": numba.__file__,
            "runtime_path": numba.core.runtime.context.__file__,
            "llvmlite": llvmlite.__version__, "jit": "passed",
        }
        (output / "import_boundary.json").write_text(json.dumps(evidence, indent=2) + "\n")
        print("V1.2 Kit import boundary: " + json.dumps(evidence), flush=True)
        return app

    isaacsim.SimulationApp = simulation_app


def main():
    print("V1.2 evaluator CFFI: " + json.dumps(preload_cffi()), flush=True)
    print("V1.2 evaluator transport: " + json.dumps(preload_transport()), flush=True)
    output = Path(sys.argv[sys.argv.index("--output-dir") + 1])
    configure_simulation_app(output)
    runpy.run_module("omnigibson.eval.eval", run_name="__main__", alter_sys=True)


if __name__ == "__main__":
    main()
