#!/usr/bin/env bash
# Local, bounded checks. Does not contact or restart training/policy services.
set -euo pipefail
bench_root="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$bench_root"
export PYTHONPATH="$bench_root/src${PYTHONPATH:+:$PYTHONPATH}"
export NO_ALBUMENTATIONS_UPDATE=1 PYTHONNOUSERSITE=1
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
bench_python="$bench_root/src/v1_groot_memoryvla/.venv/bin/python"
bench_mode="${1:-fast}"
if [[ $# -gt 0 ]]; then shift; fi
unit_check() {
    "$bench_python" -m pytest -q src/v1_groot_memoryvla/tests tests/test_groot_training_dashboard.py
}
retrieval_check() {
    "$bench_python" -m v1_groot_memoryvla.audit.retrieval_benchmark \
      --weights outputs/diagnostics/v1_cognitive_16000_weights.pt \
      --replay outputs/diagnostics/v1_real_replay_features.pt \
      --output outputs/benchmarks/retrieval_fixed.json "$@"
}
action_check() {
    "$bench_python" -m v1_groot_memoryvla.audit.action_smoke \
      --checkpoint checkpoints/groot_n1d6_base \
      --replay outputs/diagnostics/v1_real_replay_features.pt \
      --output outputs/benchmarks/action_smoke_memory.json "$@"
}
libero_check() {
    env/libero-smoke/bin/python scripts/libero_smoke.py "$@"
}
case "$bench_mode" in
  fast) unit_check; retrieval_check "$@" ;;
  unit) unit_check ;;
  retrieval) retrieval_check "$@" ;;
  action) action_check "$@" ;;
  libero) libero_check "$@" ;;
  all) unit_check; retrieval_check; action_check; libero_check ;;
  *) echo 'Usage: bash scripts/run_v1_quick_checks.sh [fast|unit|retrieval|action|libero|all] [benchmark arguments]' >&2; exit 2 ;;
esac
