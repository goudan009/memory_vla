"""Minimal coverage API shim for the pinned Numba/Isaac Sim runtime.

The B1K-2026 Python 3.11 environment carries Numba 0.66, whose optional
coverage integration expects type aliases removed from the Isaac Sim bundled
coverage 7.4 package. Coverage itself is not enabled for simulation runs.
"""

from types import SimpleNamespace


class _Tracer:
    pass


class _Types:
    Tracer = _Tracer
    TTraceData = dict
    TShouldTraceFn = object
    TFileDisposition = object
    TShouldStartContextFn = object
    TWarnFn = object
    TTraceFn = object


types = _Types()


class Coverage:
    @classmethod
    def current(cls):
        return None
