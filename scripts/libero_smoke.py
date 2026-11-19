"""Fixed-initial-state LIBERO rendering/reset/action-contract smoke, not policy evaluation."""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import time


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--suite', choices=['libero_spatial', 'libero_object', 'libero_goal', 'libero_10', 'libero_90'],
                   default='libero_spatial')
    p.add_argument('--tasks', type=int, nargs='+', default=[0])
    p.add_argument('--steps', type=int, default=20)
    p.add_argument('--seed', type=int, default=42)
    p.add_argument('--save-frames', action='store_true', help='Save initial/final camera observations for the web viewer')
    p.add_argument('--output', type=Path, default=Path('outputs/benchmarks/libero_smoke.json'))
    args = p.parse_args()
    if args.steps < 1:
        p.error('--steps must be positive')
    root = Path(__file__).resolve().parents[1]
    repo = root/'third_party/LIBERO'
    sys.path.insert(0, str(repo))
    # Reuse Torch from the training environment after isolated sim dependencies.
    sys.path.append(str(root/'src/v1_groot_memoryvla/.venv/lib/python3.10/site-packages'))
    os.environ.setdefault('MUJOCO_GL', 'egl')
    os.environ.setdefault('PYOPENGL_PLATFORM', 'egl')
    config = root/'outputs/benchmarks/libero_config'
    config.mkdir(parents=True, exist_ok=True)
    os.environ['LIBERO_CONFIG_PATH'] = str(config)
    import yaml
    assets = repo/'libero/libero'
    (config/'config.yaml').write_text(yaml.safe_dump(dict(benchmark_root=str(assets),
        bddl_files=str(assets/'bddl_files'), init_states=str(assets/'init_files'),
        assets=str(assets/'assets'), datasets=str(root/'outputs/benchmarks'))))
    import numpy as np
    import torch
    from libero.libero import benchmark
    from libero.libero.envs import OffScreenRenderEnv
    suite = benchmark.get_benchmark_dict()[args.suite]()
    rows = []
    started = time.monotonic()
    for task_id in args.tasks:
        task = suite.get_task(task_id)
        env = OffScreenRenderEnv(bddl_file_name=str(assets/'bddl_files'/task.problem_folder/task.bddl_file),
                                 camera_heights=128, camera_widths=128)
        try:
            env.seed(args.seed)
            env.reset()
            # Official pinned repository files contain NumPy arrays; Torch >=2.6
            # defaults to weights_only, which rejects these trusted init states.
            states = torch.load(assets/'init_files'/task.problem_folder/task.init_states_file,
                                map_location='cpu', weights_only=False)
            obs = env.set_init_state(states[0])
            shapes = {k: list(v.shape) for k, v in obs.items() if 'image' in k}
            assert shapes and all(shape == [128, 128, 3] for shape in shapes.values())
            frames = {}
            def capture(label):
                if not args.save_frames:
                    return
                from PIL import Image
                directory = args.output.parent/f'task-{task_id}-frames'
                directory.mkdir(parents=True, exist_ok=True)
                frames[label] = {}
                for camera in ('agentview', 'robot0_eye_in_hand'):
                    file = directory/f'{label}-{camera}.png'
                    Image.fromarray(obs[camera+'_image'][::-1]).save(file)
                    frames[label][camera] = str(file.relative_to(args.output.parent))
            capture('start')
            for _ in range(args.steps):
                # OSC_POSE: 3 translation + 3 rotation + 1 gripper. No B1K mapping.
                obs, reward, done, info = env.step(np.zeros(7, dtype=np.float32))
                assert all(np.isfinite(v).all() for v in obs.values())
                if done:
                    break
            capture('end')
            rows.append(dict(task=task_id, language=task.language, image_shapes=shapes, frames=frames,
                             steps=_+1, reward=float(reward), success=bool(env.check_success())))
        finally:
            env.close()
    report = dict(scope='Environment smoke with zero actions; NOT a GR00T success rate',
        suite=args.suite, seed=args.seed, initial_state_index=0, tasks=rows,
        libero_commit=subprocess.check_output(['git', '-C', str(repo), 'rev-parse', 'HEAD'], text=True).strip(),
        elapsed_seconds=time.monotonic()-started, passed=True)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2)+'\n')
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()
