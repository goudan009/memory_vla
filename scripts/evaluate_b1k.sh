#!/bin/sh
set -eu
project_root=$(CDPATH= cd -- "$(dirname -- "$0")/.." && pwd)
cd "$project_root"
exec sh scripts/b1k_python.sh evaluation/serf_rgb/eval.py "$@"
