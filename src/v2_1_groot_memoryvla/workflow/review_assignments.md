# Review fix assignments

Working tree: `/nvme3/projects/memory_vla-v21-review`, branch `review/v2.1`.
Do not commit, push, merge, train or start GPU inference. Preserve V1 source and research choices.

## data_review

Own only `model/memory.py` and a new `tests/test_distance.py` under `src/v2_1_groot_memoryvla`.
Fix Banks.distance constant-angle dilution. Add failing regression first: lo=zeros82,
hi=zeros82 except hi[0]=1; same visuals, x differs by1; alpha=.2 gives .2, currently .05.
Only training-range-valid fields enter the physical average. Yaw validity uses original
range before substituting pi; roll/pitch validity uses corresponding sin/cos ranges.
All-constant physical state must yield zero (no NaN). Preserve shortest-angle differences.
Run CPU focused tests using `/nvme3/envs/b1k2025-policy/bin/python`, PYTHONPATH=src,
OMP_NUM_THREADS=2. Report results and modified files.

## gr00t_review

Own new `loading.py`, `policy.py`, and new `tests/test_loading.py` under V2.1 only.
Fix service precision mismatch: V1 Policy constructor converts entire base to BF16,
but V2.1 training preserves action expert FP32, backbone BF16. Introduce shared strict
V2.1 loader (path,stage,device) using train.load_base behavior, no hardcoded cuda.
Policy must initialize BasePolicy/processor/modality/language fields directly without
calling the V1 constructor that rounds weights. Reuse inherited runtime/RPC methods.
Make loader import lightweight enough for CPU fake-loader regression where practical.
Tests must prove FP32 head weights survive actual V2.1 Policy construction and reject
loading mismatches. No real HF weights/GPU execution.
Root agent owns train.py and diagnose.py: tell root loader interface for integration.

## root

Own train.py, diagnose.py, config-identity regression tests and review documentation.
Record requested state_weight separately from calibrated actual lambda_s; bind preflight
and resume identity to initial request, persist and verify actual calibrated value.
Check complete-environment CLI, run full CPU tests and review final diffs.

## Outcome

data_review completed the distance fix and regressions. GR00T and design agents
provided the initial read-only findings. Follow-up GR00T messages could not be
reliably read across the agent transport, so root completed the shared loader,
policy precision regression, training identity and final verification locally.
The detailed review records which checks used real data and which used substitutes.
