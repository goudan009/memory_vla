#!/usr/bin/env python3
"""Pass an explicitly recorded task description to the pinned official wrapper."""

import argparse
from contextlib import contextmanager
import runpy
import sys


@contextmanager
def task_language(wrapper_module, text_prompt):
    original = wrapper_module.B1KPolicyWrapper

    def create_wrapper(*args, **kwargs):
        kwargs["text_prompt"] = text_prompt
        return original(*args, **kwargs)

    wrapper_module.B1KPolicyWrapper = create_wrapper
    try:
        yield
    finally:
        wrapper_module.B1KPolicyWrapper = original


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--official-script", required=True)
    parser.add_argument("--text-prompt", required=True)
    args, upstream_args = parser.parse_known_args()
    from gr00t.eval import eval_b1k_wrapper

    print(f"V1.2 explicit task description: {args.text_prompt}", flush=True)
    sys.argv = [args.official_script, *upstream_args]
    with task_language(eval_b1k_wrapper, args.text_prompt):
        runpy.run_path(args.official_script, run_name="__main__")


if __name__ == "__main__":
    main()
