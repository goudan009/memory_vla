#!/usr/bin/env bash
# Compatibility entry; the public evaluation tools live under evaluation/scripts.
set -euo pipefail
ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
exec bash "$ROOT/evaluation/scripts/run_groot_eval.sh" "$@"
