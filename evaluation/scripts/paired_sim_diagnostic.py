import argparse
import copy
import json
import os
import random
import sys
import traceback
from pathlib import Path

sys.path.insert(0, str(Path('evaluation/scripts').resolve()))
from groot_task40_trace import read_state, save_policy_input, serializable
from task40_replay_probe import write_json


def main(args):
    sys.path[:0] = [str(args.behavior / 'OmniGibson'), str(args.behavior / 'bddl'), str(args.groot)]
    import isaacsim
    import av
    import carb.settings
    import numpy as np
    import omnigibson as og
    import torch
    import zmq
    from gr00t.eval.sim.BEHAVIOR.behavior_env import BEHAVIORGr00tEnv, preprocess_action, preprocess_obs
    from gr00t.policy.server_client import PolicyClient
    from omnigibson import object_states
    from omnigibson.utils.usd_utils import GripperRigidContactAPI, PoseAPI
    from PIL import Image, ImageDraw

    def read_json(path):
        return json.loads(path.read_text())

    def load_arrays(path):
        with np.load(path, allow_pickle=False) as archive:
            return dict(archive)

    start = read_json(args.anchor / 'start.json')
    session = start['clock']
    random.seed(start['seed'])
    np.random.seed(start['seed'])
    torch.manual_seed(start['seed'])
    os.environ['GROOT_EVAL_INSTANCE_IDS'] = ','.join(map(str, args.instances)) if args.phase == 'collect' else str(start['instance'])
    env = BEHAVIORGr00tEnv('make_microwave_popcorn')
    if args.phase != 'collect':
        env.reset(seed=start['seed'])
    aa_key = '/rtx/post/aa/op'
    settings = carb.settings.get_settings()
    aa_initial = settings.get(aa_key)
    if args.renderer_aa is not None:
        settings.set_int(aa_key, args.renderer_aa)
    aa_after_request = settings.get(aa_key)
    base = env.env
    while 'env' in vars(base):
        base = vars(base)['env']
    robot = env.robot
    microwave = base.task.object_scope['microwave.n.02_1'].wrapped_obj
    bag = base.task.object_scope['popcorn__bag.n.01_1'].wrapped_obj
    states = {name: getattr(object_states, name) for name in ('Open', 'Inside', 'ToggledOn', 'Cooked')}
    toggle = microwave.states[states['ToggledOn']]
    door_joint, = microwave.states[states['Open']].relevant_joints_info[1]
    door = next(link for link in microwave.links.values() if link.prim_path == door_joint.body1)
    sensors = {name: sensor for name, sensor in robot.sensors.items()
               if 'zed_link' in name or 'realsense_link' in name}
    assert len(sensors) == 3
    camera_keys = [f'video.observation.images.rgb.{name}_256_256' for name in ('left_wrist', 'right_wrist', 'head')]
    client = None
    if args.phase != 'renderer_probe':
        client = PolicyClient(host=args.policy_host, port=args.port)
        client.socket.setsockopt(zmq.RCVTIMEO, 300000)

    def batch(obs):
        return {key: [value] if isinstance(value, str) else np.asarray(value)[None, None] for key, value in obs.items()}

    def snapshot(decision):
        return {'sim': og.sim.dump_state(serialized=True).detach().cpu().clone(), 'step': base._current_step,
                'progress': copy.deepcopy(env._task_progress_dict), 'info': copy.deepcopy(env.info),
                'task_success': base.task._success, 'metrics': [copy.deepcopy(vars(metric)) for metric in env.metrics],
                'python_rng': random.getstate(), 'numpy_rng': np.random.get_state(),
                'torch_rng': torch.get_rng_state(), 'cuda_rng': torch.cuda.get_rng_state_all(),
                'decision': decision, 'session': session}

    def physical():
        return serializable({'poses': {item.name: item.get_position_orientation() for item in (robot, microwave, bag)},
                'joints': {item.name: item.get_joint_positions() for item in (robot, microwave)},
                'controllers': {key: controller.dump_state() for key, controller in robot.controllers.items()}})

    def features(obs):
        return {key: np.asarray(obs[key]) for key in obs if not key.startswith('annotation.')}

    def controller_state(value):
        if isinstance(value, dict):
            return {key: controller_state(item) for key, item in value.items()}
        if isinstance(value, list):
            return torch.as_tensor(value)
        return value

    def restore(saved, reference, folder):
        assert saved['session'] == session
        og.sim.load_state(saved['sim'], serialized=True)
        base._current_step = saved['step']
        base.task._success = saved['task_success']
        env._task_progress_dict = copy.deepcopy(saved['progress'])
        env.info = copy.deepcopy(saved['info'])
        for metric, values in zip(env.metrics, saved['metrics'], strict=True):
            vars(metric).clear()
            vars(metric).update(copy.deepcopy(values))
        random.setstate(saved['python_rng'])
        np.random.set_state(saved['numpy_rng'])
        torch.set_rng_state(saved['torch_rng'])
        torch.cuda.set_rng_state_all(saved['cuda_rng'])
        robot.set_joint_positions(torch.as_tensor(reference['joints'][robot.name]))
        microwave.set_joint_positions(torch.as_tensor(reference['joints'][microwave.name]))
        for name, controller in robot.controllers.items():
            controller.load_state(controller_state(reference['controllers'][name]))
        PoseAPI.invalidate()
        sim_time_before_settle = float(og.sim.current_time)
        og.sim.step_physics()
        for name, controller in robot.controllers.items():
            controller.load_state(controller_state(reference['controllers'][name]))
        PoseAPI.invalidate()
        restored = physical()
        position = {name: float(np.abs(np.asarray(pose[0]) - restored['poses'][name][0]).max())
                    for name, pose in reference['poses'].items()}
        rotation = {name: float(min(np.abs(np.asarray(pose[1]) - restored['poses'][name][1]).max(),
                                    np.abs(np.asarray(pose[1]) + restored['poses'][name][1]).max()))
                    for name, pose in reference['poses'].items()}
        joints = {name: float(np.abs(np.asarray(value) - restored['joints'][name]).max())
                  for name, value in reference['joints'].items()}
        controllers = restored['controllers'] == reference['controllers']
        sim_time = float(og.sim.current_time)
        render_deltas = []
        previous_images = None
        render_settings = []
        for render_index in range(6):
            og.sim.render()
            observation = preprocess_obs(env, base.get_obs()[0])
            if args.renderer_audit:
                images = {key: np.asarray(observation[key]).astype(np.int16) for key in camera_keys}
                if previous_images is not None:
                    render_deltas.append({key: float(np.abs(images[key] - previous_images[key]).mean())
                                          for key in camera_keys})
                previous_images = images
            if args.phase == 'renderer_probe':
                render_settings.append(settings.get(aa_key))
                np.savez_compressed(folder / f'render_{render_index + 1:02d}.npz', **features(batch(observation)))
        cached = load_arrays(args.anchor / 'anchor_live_observation.npz')
        camera_rgb_mae = {key.split('.')[-1].removesuffix('_256_256'): float(np.mean(np.abs(
            np.asarray(observation[key], dtype=np.float32) -
            np.asarray(cached[key], dtype=np.float32)))) for key in camera_keys}
        assert sim_time == float(og.sim.current_time)
        write_json(folder / 'restore.json', {'position': position, 'rotation': rotation, 'joints': joints,
                   'controllers_equal': controllers,
                   'renderer_aa': settings.get(aa_key),
                   'renderer_aa_readbacks': render_settings,
                   'consecutive_render_rgb_mae': render_deltas,
                   'settle_physics_steps': 1,
                   'settle_sim_time_delta_s': sim_time - sim_time_before_settle,
                   'camera_rgb_mae_vs_anchor': camera_rgb_mae,
                   'camera_poses': {key: sensor.get_position_orientation() for key, sensor in sensors.items()},
                   'truth': read_state(base, states)})
        assert max((*position.values(), *rotation.values(), *joints.values())) < 1e-2 and controllers
        if not args.renderer_audit:
            assert max(camera_rgb_mae.values()) < 25.0, f'Camera RGB mismatch after restore: {camera_rgb_mae}'
        np.savez_compressed(folder / 'first_raw_restored.npz', **features(batch(observation)))
        return observation

    def truth(decision):
        row = read_state(base, states)
        low, high = door.aabb
        interaction = {'door_body': door.prim_path, 'door_aabb': [low, high], 'door_distance_m': {}, 'contacts': {},
                       'button_overlap_steps': toggle.robot_can_toggle_steps, 'button_distance_m': {}}
        if toggle.visual_marker is not None:
            marker = toggle.visual_marker.get_position_orientation()[0]
            interaction['button_world'] = marker
        for arm, fingers in robot.finger_links.items():
            points = [finger.get_position_orientation()[0] for finger in fingers]
            interaction['door_distance_m'][arm] = min(float(torch.linalg.vector_norm(
                torch.maximum(low - point, point - high).clamp_min(0))) for point in points)
            if 'button_world' in interaction:
                interaction['button_distance_m'][arm] = min(float(torch.linalg.vector_norm(point - marker)) for point in points)
            interaction['contacts'][arm] = [{'body': body, 'finger': finger, 'point': point, 'impulse': force}
                for body, finger, force, point, _, _ in GripperRigidContactAPI.get_contact_data(
                    robot.scene.idx, column_prim_paths={finger.prim_path for finger in fingers})
                if body.startswith(microwave.prim_path + '/') and float(force) != 0.0]
        hinge_position, hinge_velocity, _ = door_joint.get_state()
        hinge = float(hinge_position.reshape(-1)[0])
        row.update(step=base._current_step, decision=decision, sim_time_s=float(og.sim.current_time),
                   interaction=interaction, hinge=hinge, hinge_velocity=float(hinge_velocity.reshape(-1)[0]),
                   Open=row['microwaves'][microwave.name]['Open'],
                   ToggledOn=row['microwaves'][microwave.name]['ToggledOn'],
                   Inside=row['popcorn'][bag.name]['inside'][microwave.name])
        return serializable(row)

    def save_failure(name, branch, decision, observation, evidence, fresh=False):
        folder = args.output / 'checkpoints' / name
        folder.mkdir(exist_ok=True)
        torch.save(snapshot(decision), folder / 'sim.pt')
        write_json(folder / 'physical.json', physical())
        observed = batch(observation)
        np.savez_compressed(folder / 'observation.npz', **features(observed))
        check = client.call_endpoint('save_failure', {'name': name, 'branch': branch, 'decision': decision, 'fresh': fresh})
        save_snapshots(check, folder)
        write_json(folder / 'source.json', {'branch': branch, 'decision': decision, 'step': base._current_step,
                   'memory_replay': check, 'evidence': evidence,
                   'language': {key: value for key, value in observed.items() if key.startswith('annotation.')}})
        print('TASK40_FAILURE_CANDIDATE', name, base._current_step, check['equal'], flush=True)

    def save_snapshots(response, folder):
        snapshots = response.pop('snapshots', {})
        if args.remote_policy and set(snapshots) != {'memory.pt', 'policy_rng_online.pt'}:
            raise RuntimeError('Remote policy did not return the online memory and RNG snapshots')
        for filename, payload in snapshots.items():
            (folder / filename).write_bytes(payload)

    anchor = torch.load(args.anchor / 'sim_checkpoint.pt', map_location='cpu', weights_only=False)
    anchor_input = load_arrays(args.anchor / 'anchor_live_observation.npz')
    anchor_input.update(read_json(args.anchor / 'prefix' / f"{anchor['decision'] - 1:04d}" / 'decision.json')['language'])
    initial_physical = read_json(args.anchor / 'anchor.json')

    if args.phase == 'renderer_probe':
        output = args.output / 'metrics'
        output.mkdir()
        blocks = []
        for index in range(anchor['decision'], anchor['decision'] + args.cycles):
            entry = args.renderer_probe_actions / f'{index:04d}'
            proposed = load_arrays(entry / 'proposed.npz')
            executed = load_arrays(entry / 'executed.npz')['robot_r1']
            assert len(executed) == 8
            block = [{key: value[0, j] for key, value in proposed.items()} for j in range(8)]
            assert np.array_equal(executed, np.stack([
                preprocess_action(action)['robot_r1'].cpu().numpy() for action in block]))
            blocks.append(block)
        write_json(output / 'config.json', {'source_actions': str(args.renderer_probe_actions),
                   'aa_initial': aa_initial, 'aa_requested': args.renderer_aa, 'aa_after_request': aa_after_request,
                   'copies': args.restore_repeats, 'blocks': args.cycles})
        try:
            for i in range(1, args.restore_repeats + 1):
                folder = output / f'repeat_{i:02d}'
                folder.mkdir()
                restore(anchor, initial_physical, folder)
                with (folder / 'truth.jsonl').open('x') as stream:
                    for offset, block in enumerate(blocks):
                        decision = anchor['decision'] + offset
                        for action in block:
                            observation, _, terminated, truncated, _ = env.step(action)
                            stream.write(json.dumps(truth(decision)) + '\n')
                            if terminated or truncated or env._physx_crashed:
                                raise RuntimeError('Fixed-action restore probe encountered a simulator stop')
                        np.savez_compressed(folder / f'observation_{decision:04d}.npz', **features(batch(observation)))
                print(f'TASK40_RENDERER_PROBE_DONE repeat={i} steps={args.cycles * 8}', flush=True)
        finally:
            og.shutdown()
        return

    def capture_one(failure_name, branch, decision):
        folder = args.output / 'metrics' / f'capture_{failure_name}'
        folder.mkdir(exist_ok=True)
        obs = restore(anchor, initial_physical, folder)
        evidence_rows = []
        for replay_decision in range(anchor['decision'], decision):
            action_file = args.output / 'metrics' / branch / f'{replay_decision:04d}' / 'proposed.npz'
            with np.load(action_file, allow_pickle=False) as archive:
                proposed = {key: archive[key] for key in archive.files}
            assert all(value.shape[1] >= 8 for value in proposed.values())
            for step in range(8):
                action = {key: value[0, step] for key, value in proposed.items()}
                obs, _, terminated, truncated, info = env.step(action)
                row = truth(replay_decision)
                evidence_rows.append(row)
                if row['Open'] or row['ToggledOn'] or terminated or truncated or env._physx_crashed:
                    raise RuntimeError(f'Cannot capture closed-door failed-attempt checkpoint: {row}')
        final = evidence_rows[-1]
        contacts = [row['step'] for row in evidence_rows if any(
            contact['body'] == row['interaction']['door_body']
            for arm_contacts in row['interaction']['contacts'].values() for contact in arm_contacts)]
        hinge_delta = max(abs(row['hinge'] - evidence_rows[0]['hinge']) for row in evidence_rows)
        if len(contacts) < 3 or hinge_delta < args.min_hinge_excursion:
            raise RuntimeError(f'No real opening attempt at requested checkpoint: contacts={len(contacts)}, hinge={hinge_delta}')
        save_failure(failure_name, branch, decision, obs,
                     {'door_contact_steps': contacts, 'max_hinge_excursion_rad': hinge_delta,
                      'final_open': final['Open'], 'final_toggled_on': final['ToggledOn'],
                      'replayed_policy_decisions': decision - anchor['decision'], 'source_actions_replayed': True})

    def run(label, mode, seed, origin=None, failure_name=None, fresh_instance=None):
        nonlocal session, microwave, bag, toggle, door_joint, door
        name = f'{label}_{mode}'
        folder = args.output / 'metrics' / name
        if (folder / 'trial.json').exists():
            return
        folder.mkdir(exist_ok=True)
        write_json(args.output / 'metrics' / 'active_stage.json', {'trial': name, 'stage': 'prepare_policy'})
        fresh = fresh_instance is not None
        if fresh:
            actual_instance = env._all_test_instances[env._instance_indices_this_env[env._instance_idx_pointer]]
            assert actual_instance == fresh_instance
            random.seed(seed)
            np.random.seed(seed)
            torch.manual_seed(seed)
            observation, _ = env.reset(seed=seed)
            microwave = base.task.object_scope['microwave.n.02_1'].wrapped_obj
            bag = base.task.object_scope['popcorn__bag.n.01_1'].wrapped_obj
            toggle = microwave.states[states['ToggledOn']]
            door_joint, = microwave.states[states['Open']].relevant_joints_info[1]
            door = next(link for link in microwave.links.values() if link.prim_path == door_joint.body1)
            session = {'session_id': args.output.name, 'env_ids': [0], 'episode_ids': [name]}
            saved = {'decision': 0, 'step': base._current_step}
        elif origin:
            source = args.output / 'checkpoints' / origin
            saved = torch.load(source / 'sim.pt', map_location='cpu', weights_only=False)
            reference = read_json(source / 'physical.json')
            first_input = load_arrays(source / 'observation.npz') | read_json(source / 'source.json')['language']
        else:
            saved, reference, first_input = anchor, initial_physical, anchor_input
        prepared = client.call_endpoint('prepare_pair', {'mode': mode, 'seed': seed, 'origin': origin, 'fresh': fresh})
        if not fresh:
            write_json(args.output / 'metrics' / 'active_stage.json', {'trial': name, 'stage': 'restore_sim'})
            observation = restore(saved, reference, folder)
        write_json(args.output / 'metrics' / 'active_stage.json', {'trial': name, 'stage': 'closed_loop'})
        start_decision = saved['decision']
        before = truth(start_decision - 1)
        assert not before['Open'] and not before['ToggledOn']
        write_json(folder / 'initial.json', {'policy': prepared, 'truth': before,
                   'camera_poses': {key: sensor.get_position_orientation() for key, sensor in sensors.items()}})
        print('TASK40_PAIRED_START', name, seed, start_decision, flush=True)
        reason = 'decision_limit'
        steps = 0
        contact_history = []
        hinges = []
        candidate = None
        first_error = None
        first_error_ordinal = None
        first_open_step = None
        first_open_ordinal = None
        post_event_chunks_completed = 0
        hard_stop = False
        video_path = args.output / 'videos' / f'{name}.mp4'
        with (folder / 'truth.jsonl').open('x', buffering=1) as stream, av.open(str(video_path), 'w') as container:
            video = container.add_stream('libx264', rate=30)
            video.width, video.height, video.pix_fmt = 768, 256, 'yuv420p'
            for ordinal in range(args.cycles):
                decision = start_decision + ordinal
                entry = folder / f'{decision:04d}'
                pre_action = before if ordinal == 0 else row
                inputs = batch(observation)
                language = save_policy_input(entry, inputs)
                proposed, diagnostics = client.call_endpoint('infer_pair', {'observation': inputs,
                    'session': session, 'branch': name, 'decision': decision, 'first': ordinal == 0})
                arrays = diagnostics.pop('model_features', None)
                if args.remote_policy and arrays is None:
                    raise RuntimeError('Remote policy must run with diagnostic artifact transfer enabled')
                if arrays is not None:
                    np.savez_compressed(entry / 'model_features.npz', **arrays)
                    if ordinal == 0:
                        np.savez_compressed(folder / 'first_expert.npz', **arrays)
                assert diagnostics['memory_after']['decision'] == decision + 1
                assert all(value.shape[1] >= 8 and np.isfinite(value).all() for value in proposed.values())
                np.savez_compressed(entry / 'proposed.npz', **proposed)
                actions = []
                for index in range(8):
                    action = {key: value[0, index] for key, value in proposed.items()}
                    actions.append(preprocess_action(action)['robot_r1'].cpu().numpy())
                    observation, _, terminated, truncated, info = env.step(action)
                    row = truth(decision)
                    stream.write(json.dumps(row) + '\n')
                    steps += 1
                    interacting = row['interaction']
                    if row['Open'] and first_open_step is None:
                        first_open_step = row['step']
                        first_open_ordinal = ordinal
                    if row['ToggledOn'] and first_open_step is None and first_error is None:
                        first_error = {'step': row['step'], 'decision': decision,
                                       'Open': row['Open'], 'ToggledOn': row['ToggledOn']}
                        first_error_ordinal = ordinal
                    if any(contact['body'] == interacting['door_body'] for contacts in interacting['contacts'].values()
                           for contact in contacts):
                        contact_history.append(row['step'])
                    hinges.append(abs(row['hinge'] - before['hinge']))
                    image = Image.fromarray(np.concatenate([observation[key] for key in camera_keys], axis=1))
                    ImageDraw.Draw(image).text((5, 5), f"{name}  d={ordinal + 1:02d}  step={steps:03d}  Open={int(row['Open'])}  On={int(row['ToggledOn'])}",
                                              fill='white', stroke_width=2, stroke_fill='black')
                    for packet in video.encode(av.VideoFrame.from_ndarray(np.asarray(image), format='rgb24')):
                        container.mux(packet)
                    if terminated or truncated or env._physx_crashed:
                        reason = 'physx_crashed' if env._physx_crashed else 'environment_terminated' if terminated else 'environment_truncated'
                        hard_stop = True
                        break
                np.savez_compressed(entry / 'executed.npz', robot_r1=np.stack(actions))
                write_json(entry / 'decision.json', {'decision': decision, 'language': language, 'session': session,
                           'start_step': pre_action['step'], 'end_step': row['step'],
                           'state_before': pre_action, 'state_after': row, 'executed_steps': len(actions),
                           'instance': fresh_instance, 'seed': seed, **diagnostics})
                if first_error_ordinal is not None and ordinal > first_error_ordinal and len(actions) == 8:
                    post_event_chunks_completed = ordinal - first_error_ordinal
                if hard_stop:
                    break
                if first_error is None and first_open_step is not None:
                    reason = 'opened'
                    if not fresh and ordinal - first_open_ordinal >= args.observe_open_chunks:
                        break
                if first_error is not None:
                    if post_event_chunks_completed >= args.post_event_action_chunks:
                        reason = 'opened_after_premature_toggle' if first_open_step is not None else 'premature_toggle_observed'
                    else:
                        reason = 'post_event_observation'
                    if post_event_chunks_completed >= args.post_event_action_chunks:
                        break
                if (failure_name and candidate is None and not row['Open'] and not row['ToggledOn']
                        and len(contact_history) >= 3 and max(hinges) >= 0.01
                        and row['step'] - contact_history[-1] >= 2):
                    candidate = failure_name
                    save_failure(candidate, name, decision + 1, observation,
                                 {'door_contact_steps': contact_history.copy(), 'max_hinge_excursion_rad': max(hinges),
                                  'final_open': row['Open'], 'final_toggled_on': row['ToggledOn']}, fresh=fresh)
                if (ordinal + 1) % 10 == 0:
                    print('TASK40_PAIRED_PROGRESS', name, ordinal + 1, steps, len(contact_history), max(hinges), flush=True)
            for packet in video.encode():
                container.mux(packet)
        if not hard_stop and first_error is not None and post_event_chunks_completed < args.post_event_action_chunks:
            reason = 'decision_limit_after_event'
        assert steps == base._current_step - saved['step']
        online_prefix = None
        if fresh and not env._physx_crashed:
            online_prefix = client.call_endpoint('save_online_prefix', {'branch': name, 'decision': decision + 1})
            prefix_folder = args.output / 'checkpoints' / f'{name}_online_prefix'
            prefix_folder.mkdir(exist_ok=True)
            save_snapshots(online_prefix, prefix_folder)
        write_json(folder / 'trial.json', {'branch': name, 'seed': seed, 'mode': mode, 'origin': origin,
                   'instance': fresh_instance, 'decisions': ordinal + 1, 'steps': steps, 'stop_reason': reason, 'door_contact_steps': contact_history,
                   'max_hinge_excursion_rad': max(hinges), 'failure_candidate': candidate, 'start_step': saved['step'],
                   'end_step': base._current_step, 'first_error': first_error, 'first_open_step': first_open_step,
                   'metric_termination_step': (first_error['step'] if first_error else first_open_step),
                   'observation_end_step': base._current_step,
                   'online_prefix': online_prefix,
                   'post_event_action_chunks_requested': args.post_event_action_chunks,
                   'post_event_action_chunks_completed': post_event_chunks_completed})
        print('TASK40_PAIRED_DONE', name, reason, ordinal + 1, steps, len(contact_history), max(hinges), flush=True)

    if args.phase == 'capture':
        for specification in args.capture_specs:
            failure_name, branch, decision = specification.split(':')
            capture_one(failure_name, branch, int(decision))
        for index, specification in enumerate(args.capture_specs):
            failure_name = specification.split(':')[0]
            for mode in ('A_normal', 'B_empty'):
                run(failure_name, mode, 74001 + index, origin=failure_name)
        og.shutdown()
        return

    seeds = [row['policy_seed'] for row in read_json(args.seeds)['trials']]
    assert 1 <= len(seeds) <= 10 and len(seeds) == len(set(seeds))
    try:
        if args.phase == 'anchor':
            for seed in seeds:
                run(f'anchor_{seed}', 'A_normal', seed, failure_name=f'failure_{seed}')
                run(f'anchor_{seed}', 'B_empty', seed)
        elif args.phase == 'restore':
            for seed in seeds:
                for repeat in range(1, args.restore_repeats + 1):
                    label = f'restore_{seed}'
                    if args.restore_repeats > 1:
                        label += f'_repeat{repeat:02d}'
                    run(label, args.memory_mode, seed,
                        failure_name=f'{label}_failure' if args.capture_failed_attempt else None)
            write_json(args.output / 'metrics' / 'active_stage.json',
                       {'trial': f'{label}_{args.memory_mode}', 'stage': 'complete',
                        'decisions': args.cycles})
        elif args.phase == 'collect':
            for index, seed in enumerate(seeds):
                instance = args.instances[index % len(args.instances)]
                name = f'episode_i{instance}_s{seed}'
                run(name, 'A_normal', seed, failure_name=f'{name}_failure', fresh_instance=instance)
                if env._physx_crashed:
                    raise RuntimeError('PhysX crashed; invalid episode preserved')
        else:
            for index, origin in enumerate(args.failures):
                source = read_json(args.output / 'checkpoints' / origin / 'source.json')
                assert source['memory_replay']['equal'] and not source['evidence']['final_open']
                for mode in ('A_normal', 'B_empty'):
                    run(origin, mode, 74001 + index, origin=origin)
    except BaseException:
        (args.output / 'logs' / 'failure.txt').write_text(traceback.format_exc())
        raise
    finally:
        og.shutdown()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    for argument in ('output', 'anchor', 'seeds', 'behavior', 'groot'):
        parser.add_argument(f'--{argument}', type=Path, required=True)
    parser.add_argument('--port', type=int, default=5586)
    parser.add_argument('--policy-host', default='127.0.0.1')
    parser.add_argument('--remote-policy', action='store_true')
    parser.add_argument('--cycles', type=int, default=60)
    parser.add_argument('--post-event-action-chunks', type=int, default=2)
    parser.add_argument('--phase', choices=('anchor', 'failure', 'capture', 'restore', 'collect', 'renderer_probe'), default='anchor')
    parser.add_argument('--instances', type=int, nargs='+', default=[4])
    parser.add_argument('--restore-repeats', type=int, choices=range(1, 4), default=1)
    parser.add_argument('--renderer-audit', action='store_true')
    parser.add_argument('--renderer-aa', type=int, choices=(0, 3))
    parser.add_argument('--renderer-probe-actions', type=Path)
    parser.add_argument('--capture-failed-attempt', action='store_true')
    parser.add_argument('--observe-open-chunks', type=int, default=0)
    parser.add_argument('--memory-mode', choices=('A_normal', 'B_empty'), default='A_normal')
    parser.add_argument('--failures', nargs='*', default=[])
    parser.add_argument('--capture-specs', nargs='*', default=[
        'failure_73002_d39:anchor_73002_A_normal:39',
        'failure_73004_d26:anchor_73004_A_normal:26',
        'failure_73003_d39:anchor_73003_A_normal:39',
        'failure_73007_d29:anchor_73007_A_normal:29',
        'failure_73012_d28:anchor_73012_A_normal:28'])
    parser.add_argument('--min-hinge-excursion', type=float, default=0.005)
    args = parser.parse_args()
    if args.post_event_action_chunks < 0:
        parser.error('--post-event-action-chunks must be non-negative')
    if args.renderer_aa is not None and not args.renderer_audit:
        parser.error('--renderer-aa requires --renderer-audit')
    if args.phase == 'renderer_probe' and (not args.renderer_audit or not args.renderer_probe_actions):
        parser.error('renderer_probe requires --renderer-audit and --renderer-probe-actions')
    main(args)
