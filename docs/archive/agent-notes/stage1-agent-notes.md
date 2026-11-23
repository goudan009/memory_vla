# Stage 1：旧协作记录

本文件归档早期 `Agent_Readme/` 的局部说明；现行规则见 [仓库 AGENTS.md](../../../AGENTS.md)。

## Current source of truth

- The active base is GR00T N1.6. A candidate NVIDIA BEHAVIOR1k checkpoint is located on A100 at `/nvme3/huggingface/train_model/GR00T/GR00T-N1.6-BEHAVIOR1k`, Hub revision `300db814db8ab5dd010026d5631f280048d06b91`; exact file hashes and runtime status are in `Task_01_pipeline.md`.
- The canonical 4090 checkout is `/nvme3/projects/memory_vla`; `/media/y/workdisk/project/VLA-MEMORY-4090` is the current Codex workspace path. After editing these notes, verify their hashes at the canonical checkout; this update was confirmed present there. Never overwrite newer remote changes.
- The reviewed target layout is an independent `src/groot_memory/` package, with pinned N1.6 source under `third_party/Isaac-GR00T/`; this is a recommendation, not an implemented scaffold. Do not create the full tree before G0 and the real N1.6 contract are verified.
- The user describes the checkpoint as fine-tuned over all 50 B1K tasks, but the model card does not include a 50-task manifest. The checkpoint config has `tune_llm=false` and `tune_visual=false`; do not describe it as all-parameter fine-tuning.
- The exact NVIDIA Isaac-GR00T N1.6 `n1d6` source snapshot is present at `third_party/Isaac-GR00T/`, commit `9b37aa1ce69c73c6d165233fa88128283bba4508`, codeload archive SHA256 `0b355e6c0782802ee5510bc050e1f2228efbcfc522970250d58f4ea714b3f02f`. It has no `.git` metadata and its external submodules are not present; determine whether inference needs them and create an isolated runtime before using. Do not use the existing A100 `/nvme3/projects/GR00T/Isaac-GR00T` `main` checkout (commit `7b304f95b95790490b79bd263ba57015c3c12dfd`) or its `.venv` as a substitute; that checkout is the N1.7 line.
- π0.5 is historical only and is kept at the end of the task/process documents. Existing `README.md`, `env/README.md`, policy scripts, checkpoint symlink and paper draft may still describe the earlier π0.5 setup; do not silently treat those as evidence that GR00T is configured or working.
- The shared note `/media/y/workdisk/Agent维护/VLA-MEMORY-4090_算力资源与项目背景.md` contains useful host/resource details but predates the GR00T switch. Use its hardware/address facts as a guide and use the Stage 1 files here for the active model. Verify paths and GPU state before acting.

## Scope and preservation

- Keep this Stage 1 focused on checkpoint identity, baseline reproduction, N1.6/R1 Pro interfaces, action/kinematics mapping, and a small isolated memory-interface feasibility test.
- Do not modify the evaluator, simulator/task definitions, GROOT checkpoint, or π0.5 code path as a shortcut. Do not claim unrun steps as completed.
- The repository contains user-owned uncommitted changes, including deletion of the old root `pipeline.md`. Do not restore, reset, clean, overwrite unrelated files, or commit/push.
- Record private machine addresses only in the designated internal project notes. Never record credentials, private keys, tokens, or vault data.

## 4090 Codex/Claude repair routing

If Codex or Claude on the 4090 host has an application/runtime problem (startup, login, extension, service/process, or configuration), **do not self-repair it**. Preserve the error/log evidence and ask Hermes to use its existing repair Agent and repair memory. A local Agent may assist with inspection, but must not independently kill processes, reinstall/upgrade extensions, or edit their configuration. Ask the user before any action that may interrupt an active session. This routing applies to Codex/Claude installation/runtime faults, not ordinary project-code bugs.
