# V1.2 Progress

## 2026-11-28 - Official baseline commissioning

- Status: `PARTIAL / BLOCKED`; no V1.2 evaluation job submitted.
- Current plan: [workflow/plan-v003.json](workflow/plan-v003.json). It supersedes plan-v002 because the B1K-2026 runtime and safe live workbench activation are unavailable.
- Completed and reverified: N1.7 source revision `ace36d935b376fbf25cd56371e23877b95407c40`; official radio checkpoint identity SHA256 `cd97a22ff74bd91838350e18fe94b48e1902d407f1256f5d48eb20befe277dc3`; B1K-2026 source revision `a8247a8cc1633fe1ca0cc66aa07243d46c64f155` (`v3.9.3-post2`); Task0/40 demos and asset archives downloaded under `/nvme3` and recorded in [downloads/progress.md](downloads/progress.md). The exact `v3.9.3` tag is absent and is not claimed.
- The isolated N1.7 environment imports its pinned source and validates the official checkpoint processor/digest. Local V1.2 route contract tests passed previously (`7 passed` with pytest plugin autoload disabled); these are not live workbench or simulator acceptance.
- Runtime gate is not met. Isaac Sim 5.1 exists at `/opt/isaac-sim-5.1` and imports with its embedded Python, but OmniGibson is absent there. The available `b1k2025-sim` imports point to a 2025 checkout and fail OmniGibson import due to the NumPy 2.2.6 / OpenCV ABI mismatch. No preinstalled B1K-2026 `behavior` environment was found. Per the B1K repository instructions, no environment was created and no NVIDIA EULA was accepted.
- Asset archives are verified downloads under `/nvme3/datasets/b1k2026-assets/archives`; they have not been extracted into or recognized by a B1K-2026 runtime. `/nvme3` has approximately 502 GiB free at this check.
- Workbench `GET /api/launch/options?device=local` still serves legacy B1K-2025 options; the running service has not loaded the V1.2 route. It was not restarted. No safe in-process activation path was established.
- Historical 2025 Task40 remains schema-readable but incompatible with the N1.7/B1K-2026 route; no conversion or B1K-2025 adaptation was performed.
- No preview was accepted, no evaluation/inference job was submitted, no native rollout JSON/video or success/Q-score exists, and `training_submissions=0`. No formal training was started. V2.1 files, jobs, and service were left untouched.
- Resume only after an authorized preinstalled or user-approved B1K-2026 runtime is available, its pinned evaluator and extracted asset layout pass import/instance checks, and the workbench can activate the V1.2 route without interrupting V2.1. Then run the same one-instance preview, verify resources and note, submit through the workbench, and archive the actual job and native outputs.
