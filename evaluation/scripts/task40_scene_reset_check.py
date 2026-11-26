"""Real Task40 furniture reset regression, launched by BenchmarkManager."""
import argparse
import copy
import json
import os
from pathlib import Path
import random
import sys
import traceback


def main(args):
    from task40_restore_worker import lease_sim_cache
    lease = lease_sim_cache(args.output.parent/'isaac_appdata',
                            int(os.environ['OMNIGIBSON_GPU_ID']), 2)
    source = Path('/nvme3/projects/GR00T')
    sys.path[:0] = [str(source/'Isaac-GR00T-n1d6-9b37aa1'),
                   str(source/'BEHAVIOR-1K-task-progress/OmniGibson'),
                   str(source/'BEHAVIOR-1K-task-progress/bddl')]
    import isaacsim
    import numpy as np
    import torch
    torch.set_num_threads(2)
    import omnigibson as og
    from gr00t.eval.sim.BEHAVIOR import behavior_env as behavior
    from PIL import Image

    random.seed(0)
    np.random.seed(0)
    torch.manual_seed(0)
    os.environ['GROOT_EVAL_INSTANCE_IDS'] = '0,1,4'
    if args.arm == 'fixed':
        from groot_scene_reset import install_scene_reset
        install_scene_reset()
    env = behavior.BEHAVIORGr00tEnv('make_microwave_popcorn')
    scene = env.env.scene
    furniture = [obj for obj in scene.objects
                 if any(word in obj.category for word in ('chair', 'table'))]
    if not furniture:
        raise RuntimeError('No furniture found in Task40 scene')

    def poses():
        return {obj.name: {'category': obj.category,
                          'position': obj.get_position_orientation()[0].cpu().tolist(),
                          'orientation': obj.get_position_orientation()[1].cpu().tolist()}
                for obj in furniture}

    rows = []
    inject_active = False

    def capture(label):
        row = dict(boundary=label, furniture=poses())
        rows.append(row)
        with (args.output.parent/'boundaries.jsonl').open('a') as stream:
            stream.write(json.dumps(row)+'\n')
        print('SCENE_RESET_BOUNDARY '+label, flush=True)
        return row['furniture']

    # Preserve the actual constructor snapshot, before any per-instance rewrite.
    torch.save(copy.deepcopy(scene._initial_file), args.output.parent/'constructor_scene.pt')
    capture('constructor')
    original_update = scene.update_initial_file

    def tracked_update(*a, **kw):
        capture('before_update_initial_file')
        if args.inject_reset_drift and inject_active and not a and not kw:
            chair = next(obj for obj in furniture if obj.category == 'straight_chair')
            pos, _ = chair.get_position_orientation()
            pos = pos.clone()
            pos[0] += 0.5
            chair.set_position_orientation(pos, torch.tensor([0.7071068, 0., 0., 0.7071068]))
            chair.keep_still()
            capture('injected_settle_drift')
        return original_update(*a, **kw)

    scene.update_initial_file = tracked_update
    original_restore = scene.restore

    def tracked_restore(*a, **kw):
        capture('before_restore')
        result = original_restore(*a, **kw)
        capture('after_restore')
        return result

    scene.restore = tracked_restore

    def reset_image(index):
        obs, _ = env.reset(seed=index)
        for camera in ('head', 'left_wrist', 'right_wrist'):
            Image.fromarray(obs[f'video.observation.images.rgb.{camera}_256_256']).save(
                args.output.parent/f'reset-{index}-{camera}.png')
        return capture(f'reset_{index}_zero_action')

    reference = reset_image(0)
    targets = [obj for obj in furniture
               if obj.category in ('straight_chair', 'breakfast_table') and not obj.fixed_base]
    if not targets:
        raise RuntimeError('No kitchen chairs/table to perturb')
    trials = []
    for index in range(1, 4):
        if args.collision_steps:
            from gr00t.eval.sim.BEHAVIOR.behavior_env import preprocess_obs
            obs, _ = env.env.get_obs()
            obs = preprocess_obs(env, obs)
            action = {'action.base': np.array([1., 0., 0.], dtype=np.float32),
                      'action.torso': obs['state.trunk_qpos'].copy(),
                      'action.left_arm': obs['state.arm_left_qpos'].copy(),
                      'action.right_arm': obs['state.arm_right_qpos'].copy(),
                      'action.left_gripper': np.array([-1.], dtype=np.float32),
                      'action.right_gripper': np.array([-1.], dtype=np.float32)}
            env.robot.set_position_orientation(torch.tensor([4.29, 0.25, 0.005]),
                                               torch.tensor([0., 0., 0.7071068, 0.7071068]))
            for _ in range(args.collision_steps):
                env.step(action)
                if env._physx_crashed:
                    raise RuntimeError('Physics failed during the contact reproduction')
        else:
            for obj in targets:
                pos, _ = obj.get_position_orientation()
                pos = pos.clone()
                pos[0] += 2.0
                pos[2] += 0.5
                obj.set_position_orientation(pos, torch.tensor([0.7071068, 0., 0., 0.7071068]))
                obj.keep_still()
            for _ in range(240):
                og.sim.step_physics()
        before = capture(f'perturbed_{index}')
        inject_active = True
        after = reset_image(index)
        errors = {}
        for name in reference:
            a, b = reference[name], after[name]
            q, r = np.asarray(a['orientation']), np.asarray(b['orientation'])
            errors[name] = dict(position_m=float(np.linalg.norm(np.asarray(a['position'])-b['position'])),
                               angle_rad=float(2*np.arccos(np.clip(abs(q@r), 0, 1))))
        trial = dict(index=index, perturbed_objects=[obj.name for obj in targets], before=before, after=after, errors=errors,
                     restored=all(e['position_m'] < 0.03 and e['angle_rad'] < 0.05 for e in errors.values()))
        trials.append(trial)
        print(json.dumps(dict(step=index, restored=trial['restored'], errors=errors)), flush=True)
    restored = all(trial['restored'] for trial in trials)
    result = dict(passed=(restored if args.arm == 'fixed' else True), arm=args.arm,
                  collision_steps=args.collision_steps,
                  inject_reset_drift=args.inject_reset_drift,
                  resets_restored=restored, furniture=reference, trials=trials,
                  boundaries=str(args.output.parent/'boundaries.jsonl'))
    args.output.write_text(json.dumps(result, indent=2)+'\n')
    return result['passed']


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--arm', choices=('original', 'fixed'), required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--collision-steps', type=int, choices=(0, 120), default=0)
    parser.add_argument('--inject-reset-drift', action='store_true')
    args = parser.parse_args()
    try:
        passed = main(args)
    except BaseException as error:
        traceback.print_exc()
        args.output.write_text(json.dumps(dict(passed=False, error=str(error)), indent=2)+'\n')
        passed = False
    sys.stdout.flush()
    sys.stderr.flush()
    os._exit(0 if passed else 1)
