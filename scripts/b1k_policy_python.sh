#!/usr/bin/env bash
set -euo pipefail

POLICY_ENV="/nvme3/envs/b1k2025-policy-wyh"
export PATH="$POLICY_ENV/bin:$PATH"
export PYTHONPATH="/nvme3/shared/b1k2025/policy${PYTHONPATH:+:$PYTHONPATH}"
export PYTHONNOUSERSITE=1
export PYTHONUNBUFFERED=1
export XLA_PYTHON_CLIENT_MEM_FRACTION="${XLA_PYTHON_CLIENT_MEM_FRACTION:-0.85}"
export XLA_PYTHON_CLIENT_ALLOCATOR="${XLA_PYTHON_CLIENT_ALLOCATOR:-platform}"

unset PYTHONHOME
exec "$POLICY_ENV/bin/python" -s "$@"
