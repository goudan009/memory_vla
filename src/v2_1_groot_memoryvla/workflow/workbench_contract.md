# V2.1 workbench interface (in progress)

Agent workbench_v21 owns train/dashboard.py, train/dashboard_remote.py, train/v21_workbench.py and tests/test_v21_workbench.py. No GPU jobs will be submitted by this agent.

Root trainer contract: V2.1 uses standard args --stage --dataset --checkpoint --labels --output --steps --seed --microbatch --lr --prior --fusion-layers --weak-weight --state-weight, optional --split --calibration-reference --preflight-only --accept-preflight --resume and --execute. Torchrun four ranks. Workbench trainer output is TRAINING/<run>/run/; dashboard status lives in TRAINING/<run>/<arm>/ where arm baseline for stage1/continued and memory_v1 for memory; parent config/web_launch.json records version=v2.1. No formal config changes.

Workbench stop will write trainer-output/stop.json {"stop_after_step": integer, "reason": ...}; trainer must check this at optimizer boundaries and save full state then return. Preferred trainer status.json includes status,step,total_steps,training_complete. Worker supervisor also derives actual step from metrics.jsonl and terminal status, but cannot substitute checkpoint validity. Resume locates newest checkpoints/*/COMPLETE.json plus training_state.pt; same original source snapshot and exact CLI, replacing --resume only. Each rank RNG/runtime persistence remains trainer responsibility.

Root: please ensure trainer jsonwrite uses atomic pending+rename (status polled concurrently). Current direct write_text can expose partial JSON to controller. Workbench argv now includes --checkpoint-prefix set to exact run_name to avoid run002 being saved as run001. Protocol stop.json compatibility verified in evolving train.py.

Validation in progress: new tests 7 passed; broad test run first had 8 failures, 7 caused by absent PYTHONPATH=src / interpreter missing pidfd APIs; one introduced fake-root RPC fixture issue was reproduced and fixed by locating bundled backend beside dashboard.py if fake root omits it. Using systematic-debugging skill for the failure diagnosis.

2026-11-27 latest: `PYTHONPATH=src python3 -m pytest -q tests/test_v21_workbench.py tests/test_groot_training_dashboard.py`: **113 passed** (12.87s). Adapter ready for root integration; no GPU launch performed. New files must be staged (`train/v21_workbench.py`, `tests/test_v21_workbench.py`).

Permanent preflight purge also removes only its exact PROJECT/deployments/<same-run> source snapshot, checks no other recorded source consumer, and retains deletion audit outside the deleted directory. Normal delete retains existing recoverable .trash convention. Always archive evidence externally first.

## Handoff ready

All owned edits are complete, no commit and no GPU launch. Final targeted suite after purge extension: 9 passed (2.58s); immediately preceding full legacy+new suite: 113 passed (12.87s). py_compile and git diff --check clean. Parent can now review/integrate these paths. Root's trainer now has atomic JSON and matching stop.json handling.

Request example: `{ "version":"v2.1", "task":40, "stage":"stage1", "run_id":"v2.1__test01__exp14__run001__preflight001", "preflight_only":true, "gpus":[0,1,2,3], "dataset":"<remote absolute dataset>", "checkpoint":"<remote generic GR00T>", "labels":"<remote absolute labels>" }`. Explicit defaults optional: steps6000,seed42,microbatch2,learning_rate1e-5,global_windows_per_step2,weak_weight.5,state_weight.01. Formal request removes preflight suffix, sets preflight_only false, adds external accept_preflight JSON. Memory stage uses another canonical runNNN and valid Stage1 step6000. For jobs/detail/stop/resume/delete use run and arm recorded by start.

For permanent cleanup: `{ "run":"<preflight run>", "arm":"baseline", "delete_files":true, "expected_output":"<start.output>", "purge_preflight":true, "evidence_path":"<remote permanent nonempty JSON evidence>" }`. This removes only confirmed terminal preflight output and its exact unused source snapshot, not formal weights or labels.

Important HTTP visibility: existing running main-root TrainingManager reads main/train/dashboard_remote.py on each RPC, and its manifest currently omits V2.1. A new worktree-root TrainingManager can manage these jobs using the same queue/storage, but HTTP's old main-root module cannot correctly display V2.1 until reviewed backend/source integration is present in its root (merge/deploy or safe reload). Do not claim old HTTP already supports this based solely on direct worktree-manager success. Existing auto-sync from main may overwrite live between worktree-manager operations; snapshots remain immutable. Root must coordinate one-way sync during launch.

Temporary preflight cannot be promoted as a Stage1 step6000 source. Permanent evidence should live outside disposable output before delete. delete_files true moves output to .trash by existing workbench convention; explicit purge_preflight true may permanently remove only a terminal preflight output after evidence path is supplied and references checked.
