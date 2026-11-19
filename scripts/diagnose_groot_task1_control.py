"""Actual B1K arm controller exercise, then unchanged recorded demo actions.

This is an isolated control diagnostic, not a scored episode. Replaying a clip
does not restore its object states and cannot establish grasp success.
"""
import json
import os
from pathlib import Path
import sys
import time

ROOT = Path('/nvme3/projects/memory_vla')
OUT = ROOT / 'outputs/diagnostics/task1_groot_b1k/control'
SOURCE = '/nvme3/projects/GR00T/Isaac-GR00T-n1d6-9b37aa1'
BEHAVIOR = '/nvme3/projects/GR00T/BEHAVIOR-1K-task-progress'


def main():
    import isaacsim  # required before OG
    sys.path[:0] = [SOURCE, BEHAVIOR + '/OmniGibson', BEHAVIOR + '/bddl', str(ROOT / 'scripts')]
    import av
    import numpy as np
    import pandas as pd
    import torch
    import gymnasium as gym
    import omnigibson as og
    from gr00t.eval.sim.BEHAVIOR import behavior_env as be
    OUT.mkdir(parents=True, exist_ok=True)
    np.random.seed(0); torch.manual_seed(0)
    be.register_behavior_envs()
    env = gym.make('sim_behavior_r1_pro/collecting_childrens_toys', env_idx=0, total_n_envs=1)
    inner = env.unwrapped
    # Gym unwrapped is the BEHAVIOR wrapper's OG environment in some versions.
    current = env
    while not isinstance(current, be.BEHAVIORGr00tEnv):
        current = current.env
    robot = current.robot
    obs, info = env.reset(seed=0)
    selection = json.loads((OUT.parent / 'selection.json').read_text())['clip']
    groups = {'base': (0, 3), 'torso': (3, 7), 'left_arm': (7, 14),
              'left_gripper': (14, 15), 'right_arm': (15, 22), 'right_gripper': (22, 23)}
    rows = []; cmds = []; actuals = []; frames = []
    started = time.monotonic()
    def arr(v):
        return v.detach().cpu().numpy().copy() if hasattr(v, 'detach') else np.asarray(v).copy()
    joint_indices = {side: arr(robot.arm_control_idx[side]).tolist() for side in ('left', 'right')}
    identity = {'task': 21, 'instance': 0, 'seed': 0, 'controller_order': robot.controller_order,
                'arm_joint_indices': joint_indices, 'joint_names': list(robot.joints),
                'physics_dt': og.sim.get_physics_dt(), 'rendering_dt': og.sim.get_rendering_dt(),
                'action_dt': og.sim.get_sim_step_dt(), 'native_horizon': current.env.task._termination_config,
                'control_mode': 'unchanged production BEHAVIOR controllers, physics, cameras and task rules',
                'replay_scope': 'real raw action clip; only robot joint pose aligned, object states NOT restored; no grasp success claim'}
    def run(command, phase):
        nonlocal obs
        action = {'action.' + k: np.asarray(command[lo:hi], np.float32).copy() for k, (lo, hi) in groups.items()}
        before = {side: arr(robot.get_joint_positions())[joint_indices[side]] for side in ('left', 'right')}
        obs, _, term, trunc, info = env.step(action)
        after = {side: arr(robot.get_joint_positions())[joint_indices[side]] for side in ('left', 'right')}
        goal_error = {}
        for side, g in [('left', 'left_arm'), ('right', 'right_arm')]:
            lo, hi = groups[g]
            goal = arr(robot.controllers['arm_' + side]._goal['target'])
            goal_error[side] = float(np.max(np.abs(goal - command[lo:hi])))
        target = np.r_[np.zeros(3), arr(obs['state.trunk_qpos']), arr(obs['state.arm_left_qpos']),
                       np.zeros(1), arr(obs['state.arm_right_qpos']), np.zeros(1)]
        cmds.append(command.copy()); actuals.append(target)
        row = {'step': len(rows), 'phase': phase, 'goal_command_max_error': goal_error,
               'left_arm_motion': float(np.max(np.abs(after['left'] - before['left']))),
               'right_arm_motion': float(np.max(np.abs(after['right'] - before['right']))),
               'left_gripper_qpos': arr(obs['state.gripper_left_qpos']).tolist(),
               'right_gripper_qpos': arr(obs['state.gripper_right_qpos']).tolist(),
               'terminated': bool(term), 'truncated': bool(trunc), 'valid': info.get('valid')}
        rows.append(row)
        if len(rows) % 3 == 0:
            frames.append(np.concatenate([arr(obs[f'video.observation.images.rgb.{c}_256_256']) for c in ('head', 'left_wrist', 'right_wrist')], axis=1))
        if term or trunc: raise RuntimeError(f'Diagnostic unexpectedly terminated: {row}')
    hold = np.r_[np.zeros(3), arr(obs['state.trunk_qpos']), arr(obs['state.arm_left_qpos']),
                 np.ones(1), arr(obs['state.arm_right_qpos']), np.ones(1)].astype(np.float32)
    for side, index in [('left', 7), ('right', 15)]:
        # Small positive first-joint motion, inside actual robot limits.
        j = joint_indices[side][0]
        delta = min(.12, float(arr(robot.joint_upper_limits)[j]) - hold[index] - .02)
        if delta < .05: raise ValueError('Selected joint lacks positive travel margin')
        for i in range(50):
            command = hold.copy(); command[index] += delta * min((i + 1) / 25., 1.)
            run(command, 'manual_' + side)
        for i in range(25):
            command = hold.copy(); command[index] += delta * (1 - (i + 1) / 25.)
            run(command, 'return_' + side)
    for value, phase in [(-1., 'manual_close'), (1., 'manual_open')]:
        for _ in range(25):
            command = hold.copy(); command[[14, 22]] = value
            run(command, phase)
    a = np.stack(pd.read_parquet(ROOT / f'assets/datasets/b1k2025/data/task-0021/episode_{selection["episode"]:08d}.parquet', columns=['action'])['action']).astype(np.float32)
    # Gradual bridge into the recorded pose; no instantaneous joint teleport.
    start = selection['start']; bridge = a[start].copy()
    for i in range(60):
        command = hold + (bridge - hold) * ((i + 1) / 60.)
        command[:3] = 0
        run(command, 'demo_pose_bridge')
    for t in range(start, selection['end']):
        run(a[t], 'demonstration')  # all 23 values unchanged
        if (t - start) % 80 == 0: print('DEMO_STEP', t, flush=True)
    with av.open(str(OUT / 'manual_and_demo.mp4'), 'w') as container:
        stream = container.add_stream('libx264', rate=10)
        stream.width=768; stream.height=256; stream.pix_fmt='yuv420p'
        for frame in frames:
            for packet in stream.encode(av.VideoFrame.from_ndarray(frame, format='rgb24')): container.mux(packet)
        for packet in stream.encode(): container.mux(packet)
    np.savez_compressed(OUT / 'control_arrays.npz', commands=np.asarray(cmds), actual=np.asarray(actuals))
    (OUT / 'steps.jsonl').write_text(''.join(json.dumps(r) + '\n' for r in rows))
    phases = {}
    for phase in sorted({r['phase'] for r in rows}):
        ids = [i for i, r in enumerate(rows) if r['phase'] == phase]
        c, q = np.asarray(cmds)[ids], np.asarray(actuals)[ids]
        phases[phase] = {'steps': len(ids), 'left_arm_mae_rad': float(np.abs(c[:, 7:14] - q[:, 7:14]).mean()),
                         'right_arm_mae_rad': float(np.abs(c[:, 15:22] - q[:, 15:22]).mean()),
                         'left_arm_joint_excursion': np.ptp(q[:,7:14], axis=0).tolist(),
                         'right_arm_joint_excursion': np.ptp(q[:,15:22], axis=0).tolist(),
                         'goal_command_max_error': max(max(rows[i]['goal_command_max_error'].values()) for i in ids)}
    identity.update(phases=phases, steps=len(rows), elapsed_s=time.monotonic()-started,
                    gripper_qpos_ranges={side: [float(np.min([r[f'{side}_gripper_qpos'] for r in rows])),
                                               float(np.max([r[f'{side}_gripper_qpos'] for r in rows]))] for side in ('left','right')})
    (OUT / 'report.json').write_text(json.dumps(identity, indent=2, default=str) + '\n')
    print('CONTROL_COMPLETE', json.dumps(phases), flush=True)
    env.close(); sys.stdout.flush(); os._exit(0)


if __name__ == '__main__':
    main()
