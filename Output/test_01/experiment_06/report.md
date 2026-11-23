# Task40 Memory Validity Experiment 06

Single-checkpoint paired A/B diagnostic. Every pair restores the same simulator checkpoint, rebuilds the original nine-decision memory, and resets policy RNG to the listed seed. The only intervention is clearing both memory banks in B. A success is a measured `Open=False -> True`; toggle while the door stays closed is reported separately.

| Seed | Normal | Empty | Normal first Open | Empty first Open | Normal decisions | Empty decisions | Inputs/RNG match |
|---:|---|---|---:|---:|---:|---:|---|
| 73002 | early-toggle | early-toggle | - | - | 38 | 15 | True/True |
| 73003 | open | open | 321 | 236 | 32 | 21 | True/True |
| 73004 | open | timeout | 219 | - | 19 | 60 | True/True |
| 73005 | early-toggle | early-toggle | - | - | 16 | 13 | True/True |
| 73006 | open | open | 202 | 209 | 17 | 18 | True/True |
| 73007 | open | open | 242 | 428 | 22 | 45 | True/True |
| 73008 | timeout | early-toggle | - | - | 60 | 14 | True/True |
| 73009 | early-toggle | timeout | - | - | 20 | 60 | True/True |
| 73010 | early-toggle | early-toggle | - | - | 14 | 14 | True/True |
| 73012 | open | early-toggle | 236 | - | 21 | 17 | True/True |

## Paired Summary
- Completed pairs: 10/10. Policy inputs match exactly in 10/10 pairs. Policy RNG snapshots match in 10/10 pairs.
- Door opening: normal-only 2, empty-only 0, both 3, neither 5. Discordant-pair exact two-sided p=0.5000.
- Premature toggle with door closed: normal-only 1, empty-only 2; observed totals are 4/10 versus 5/10.
- Toggle discordance exact two-sided p=1.0000.
- Per-seed input, RNG, action, normalized action-expert features, memory clocks, first physical contacts, and hinge excursions are in `metrics/paired_results.json` and `metrics/paired_results.csv`.
- A nonzero action or expert-feature delta confirms intervention sensitivity only. It does not by itself establish task improvement.
- Refreshed first observations after state restore are saved and diffed per camera in the JSON; this checks renderer cache variation independently from the identical anchor images actually sent to the policy.
- Restored camera `left_wrist` mean pixel MAE across paired loads is 39.93; checkpoint-to-restored MAE is 46.49.
- Restored camera `right_wrist` mean pixel MAE across paired loads is 66.66; checkpoint-to-restored MAE is 89.10.
- Restored camera `head` mean pixel MAE across paired loads is 31.03; checkpoint-to-restored MAE is 48.12.
- Camera world poses after restore still vary despite exact robot/object joints and six render refreshes: max paired position delta 2.423 m, sign-invariant quaternion component delta 0.604. The first model inputs were frozen to be identical, but downstream visual feedback is not yet verified as synchronized.
- Across pairs, first action MAE averages 0.004460.
- `action-expert input` delta averages MAE 0.225853, with 85.3% of elements changed.
- `normalized state` delta averages MAE 0.000000, with 0.0% of elements changed.
- `DiT condition` delta averages MAE 0.054146, with 85.3% of elements changed.

## Failed-Opening History

- Captured genuine failed-attempt checkpoints: 5. Each checkpoint must show repeated contact with the microwave door, measurable hinge excursion, and `Open=false`, `ToggledOn=false`; recorded observations are used to rebuild the policy memory, with deterministic replay checked twice.
- Completed paired continuations: 5. Per-checkpoint starting truth, event outcomes and contact/hinge evidence are in `metrics/failed_history_pairs.json`.

## Validity Notes

- Checkpoint restoration requires robot/object position, quaternion, joints, and controller state to compare within 1e-4; per-run evidence is `metrics/*/restore.json`.
- The active simulator emitted asset USD hash mismatch warnings. Physical restoration was checked numerically, but the warnings and camera-pose drift reduce confidence that all collision/render assets match the historical checkpoint version.
- No model weights were modified. No training or formal three-condition ablation was run.


| Failed-history checkpoint | Normal continuation | Empty continuation | Open steps A/B | Toggle steps A/B |
|---|---|---|---:|---:|
| failure_73002_d39 | open | early-toggle | 494/- | -/384 |
| failure_73003_d39 | open | open | 323/321 | -/- |
| failure_73004_d26 | open | open | 219/218 | -/- |
| failure_73007_d29 | open | open | 242/240 | -/- |
| failure_73012_d28 | open | open | 232/232 | -/- |

- Failed-history pairs opening successfully: normal memory 5/5, empty memory 4/5.
- Premature toggles after the checkpoint: normal memory 0/5, empty memory 1/5.
- The `failure_73002_d39` pair is the clearest divergence. The normal-history branch opened the door; the paired empty-memory branch toggled the microwave while the door remained closed.
- The other four checkpoints opened under both memory conditions in one or two decisions. Those outcomes show that a door-contact checkpoint can be near the natural opening threshold, so they are not independent evidence of memory benefit.

## Conclusion

- Normal memory changes the action expert condition and produces paired behavior differences, so the model consumes the stored history.
- From the common door-front state, normal memory has a higher observed opening rate than empty memory, but only two discordant pairs and p=0.50 give weak evidence. Premature toggles are 4/10 versus 5/10, a one-trial net difference with p=1.00.
- After genuine door-contact attempts, normal-history memory succeeds in the strongest failed-attempt pair, while four near-opening checkpoints succeed with either condition. This is promising but not a stable general benefit estimate.
- Keep MemoryVLA as a research path, but prioritize explicit task-stage and failed-interaction state supervision before claiming robust task improvement. The available evidence supports follow-up, not a formal three-group ablation.
- Do not treat the trajectory comparison as a final memory efficacy claim until post-restore camera transforms and RGB feedback are synchronized and revalidated.

## Reproduction

- Policy weights: `Output/task40_memory_overnight/replay/original_step4000`.
- Environment checkpoint: `Output/test_01/experiment_03/checkpoints`; paired seed list: `Output/test_01/experiment_05/metrics/empty_memory_sweep_summary.json`.
- The anchor phase ran 10 valid seeds, 8 actions per decision and up to 60 decisions. A rebuilt both banks from nine prefix observations; B cleared both banks while retaining the same decision clock.
- Five closed-door checkpoints replayed recorded policy actions from normal-memory runs. Each checkpoint ended after verified microwave-door contact and nonzero hinge movement while `Open=false` and `ToggledOn=false`.
- Run the policy service with `CUDA_VISIBLE_DEVICES=1 HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 PYTHONPATH="$PWD/src" /nvme3/projects/GR00T/official-baseline-env/bin/python -u Output/test_01/experiment_06/scripts/policy_server.py --output Output/test_01/experiment_06 --anchor Output/test_01/experiment_03/checkpoints --weights Output/task40_memory_overnight/replay/original_step4000 --port 5586`.
- Run the single-environment simulator with `env -u OMNIGIBSON_DATA_PATH OMNIGIBSON_GPU_ID=0 OMNIGIBSON_HEADLESS=true bash scripts/b1k_python.sh Output/test_01/experiment_06/scripts/paired_sim.py --phase anchor --output Output/test_01/experiment_06 --anchor Output/test_01/experiment_03/checkpoints --seeds Output/test_01/experiment_05/metrics/empty_memory_sweep_summary.json --behavior /nvme3/projects/GR00T/BEHAVIOR-1K-task-progress --groot /nvme3/projects/GR00T/Isaac-GR00T-n1d6-9b37aa1 --port 5586 --cycles 60`.
- The failed-history phase uses `--phase capture` with the five `failure_*:branch:decision` specifications in `scripts/paired_sim.py`.