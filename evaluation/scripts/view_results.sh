#!/usr/bin/env bash
set -euo pipefail
TOOLS=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
exec python3 "$TOOLS/view_results.py" "$@"
