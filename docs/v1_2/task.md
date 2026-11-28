# V1.2 Official Baseline Acceptance

## Goal

Finish environment commissioning, native B1K-2026 workbench integration, and one official GR00T N1.7 radio public-test closed-loop acceptance. Keep all new environments and extracted assets under `/nvme3`.

## Locked scope

- Method/protocol: `V_1_2`, GR00T N1.7 official radio checkpoint, B1K-2026 WebSocket, temporal ensemble, RTC off.
- Source: Isaac-GR00T `ace36d935b376fbf25cd56371e23877b95407c40`; BEHAVIOR-1K `a8247a8cc1633fe1ca0cc66aa07243d46c64f155` (`v3.9.3-post2`, not the unavailable exact `v3.9.3` tag).
- Evaluation target: one public-test `turning_on_radio` instance, one environment, one rollout, with native JSON/video outputs. The isolated workbench is on `8767`; `8766` is occupied by another project's service and is not to be taken over.
- Run note: `V1.2 / N1.7 / B1K2026: official radio, RTC off`.
- No formal training, no V2.2 development and no B1K-2025 adaptation. On 2026-11-28 the user authorized stopping confirmed legacy B1K-2025/V2.1 runtime after B1K-2026 passes native closed-loop acceptance; retain historical data, checkpoints and source.
- No `testNN` / `expNN` / `runNNN` is allocated because this task commissions infrastructure and evaluates a fixed official checkpoint; do not invent training identity.

## Acceptance

1. Independent N1.7 environment imports pinned GR00T source and checkpoint processor.
2. B1K-2026 evaluator imports with required public-test assets and Isaac Sim available.
3. Existing workbench preview and start route bind the exact source revision, checkpoint digest, protocol, task, instance and concise note; no legacy 2025 command is reachable from V1.2.
4. One real workbench job completes and exposes native result JSON, video, success and Q-score evidence.
5. Historical 2025 Task40 audit is reported as schema-readable but not compatible with the 2026 checkpoint/training route; no conversion is performed.
6. Final report records actual job ID/output, environment and asset verification, `training_submissions=0`, and legacy services stopped or retained under the transition gate. Preserve historical V2.1 files and results.

## Execution records

- Latest plan: [plan-v025.json](workflow/plan-v025.json)
- Prior Task40 audit: [local-preflight-2026-11-27.md](local-preflight-2026-11-27.md)
- Progress: [progress.md](progress.md)
