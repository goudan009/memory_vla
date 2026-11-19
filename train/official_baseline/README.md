# Official GR00T N1.6 baseline

This run uses unchanged NVIDIA training code and unchanged NVIDIA-converted
BEHAVIOR datasets. No Memory modules, custom converters, language rewriting or
custom training loops are used.

- Source: NVIDIA/Isaac-GR00T `9b37aa1ce69c73c6d165233fa88128283bba4508`.
- Data: NVIDIA/PhysicalAI-Robotics-GR00T-X-Embodiment-Sim
  `ea7ac0b68f87da62f1e726771bba0fe74300802f`, Tasks 0 / 34 / 40.
- Initialization: `checkpoints/groot_n1d6_behavior1k`.
- Official multi-dataset configuration and `gr00t.experiment.experiment.run`.
  The local launcher only reads JSON and calls this unchanged official entrypoint;
  it avoids a tyro assertion when parsing the upstream union-based training CLI.
- Official optimizer, schedule, augmentation, modality, padding and sampling.
- Short-run hardware settings: one 4090 GPU0, batch4, accumulation8
  (32 examples per optimizer update), 1,000 optimizer updates, save every250.
- The official model's default trainability is retained, including its top
  four LLM layers. `tune_llm=false` alone does not freeze these layers.
- Environment: `/nvme3/projects/GR00T/official-baseline-env`; existing dependency
  versions reused, optional DeepSpeed omitted for this single-GPU run.

The reduced batch and training budget are for a quick baseline experiment;
this is not a reproduction of the published 150,000-step / batch1024 result.
The three tasks use equal mix ratios through the official dataset factory.

```bash
# Run training (project root)
GROOT_TRAIN_GPU=0 bash train/official_baseline/run.sh

# After training: terminal1, official policy server on GPU1, port5558
bash train/official_baseline/evaluate.sh server

# Terminal2, simulation GPU0. Replace 0 with 34 or40 for the other tasks.
bash train/official_baseline/evaluate.sh eval 0
```

Training started on 2026-11-13 at approximately16:13, workstation4090 GPU0,
PID2923493. The first optimizer updates passed, with approximately38 GiB total
GPU memory occupied. Log: `outputs/official_baseline_task0_34_40_1000/train.log`.
All 2,421 official dataset files (600 episodes,11.35 GiB) were downloaded and
checksum-verified against the pinned dataset tree before launch. The official
loader may generate runtime statistics in dataset metadata as part of its
normal initialization. No raw trajectories or videos were converted locally.

Expected final checkpoint after training:
`outputs/official_baseline_task0_34_40_1000/checkpoint-1000`.

Evaluation uses the existing official-eval runner: 10 full episodes,
seed0, action chunk8, native timeout and success rules. Results are stored
under `evaluation/V_1/official/task*_eval*`. No evaluation is started
automatically by this training setup.
