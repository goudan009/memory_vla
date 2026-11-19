"""Opt-in deeper simulator timing, using existing callbacks without extra GPU sync."""

from contextlib import nullcontext
from functools import wraps
import os
import threading

import groot_eval_timing as timing


ENABLED = os.environ.get("GROOT_EVAL_SIM_PROFILE") == "1"
_PHYSICS_SPANS = {}


def sim_scope(name):
    return timing.profile_scope(name) if ENABLED else nullcontext()


def sim_count(name, amount=1):
    if ENABLED:
        timing.profile_count(name, amount)


def sim_metadata(name, value):
    if ENABLED:
        timing.profile_metadata(name, value)


def sim_profiled(name):
    return timing.profiled(name) if ENABLED else lambda function: function


def physics_pre(function):
    if not ENABLED:
        return function

    @wraps(function)
    def wrapped(*args, **kwargs):
        with sim_scope("physics.pre_callback"):
            result = function(*args, **kwargs)
        if timing._ACTIVE:
            ident = threading.get_ident()
            if ident in _PHYSICS_SPANS:
                sim_count("physics.unpaired_pre")
                _PHYSICS_SPANS.pop(ident).__exit__(None, None, None)
            context = sim_scope("physics.native_interval")
            context.__enter__()
            _PHYSICS_SPANS[ident] = context
        return result
    return wrapped


def physics_post(function):
    if not ENABLED:
        return function

    @wraps(function)
    def wrapped(*args, **kwargs):
        context = _PHYSICS_SPANS.pop(threading.get_ident(), None)
        if context is not None:
            context.__exit__(None, None, None)
        elif timing._ACTIVE:
            sim_count("physics.unpaired_post")
        with sim_scope("physics.post_callback"):
            return function(*args, **kwargs)
    return wrapped
