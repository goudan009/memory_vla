"""Task40 simulator phase of the existing restore and collection worker.

Promoted from Experiment 07's paired simulator with optional online trace
recording and post-event collection for Experiment 09.
"""
import argparse
import copy
import json
import os
import random
import sys
import time
import traceback
from hashlib import sha256
from pathlib import Path

sys.path.insert(0, str(Path('evaluation/scripts').resolve()))
from groot_task40_trace import read_state, save_policy_input, serializable
from task40_replay_probe import write_json


CAMERA_KEYS = tuple(f'video.observation.images.rgb.{name}_256_256'
                    for name in ('left_wrist', 'right_wrist', 'head'))
RENDER_AA_KEYS = ('/rtx/post/aa/op', '/rtx-defaults/post/aa/op')
GI_HISTORY_KEY = '/rtx/indirectDiffuse/denoiser/temporal/maxHistory'
INDIRECT_DIFFUSE_KEYS = ('/rtx/indirectDiffuse/enabled', '/rtx-defaults/indirectDiffuse/enabled')


def configure_restore_aa(app, settings, requested):
    if requested == 0:
        # Replicator enables unrestricted modes before selecting non-DLSS AA.
        settings.set_bool('/rtx-transient/post/aa/limitedOps', False)
    # Both the active renderer and its stage defaults must agree.
    app.config['anti_aliasing'] = requested
    for key in RENDER_AA_KEYS:
        settings.set_int(key, requested)
    return read_restore_aa(settings, requested)


def read_restore_aa(settings, requested):
    values = [settings.get(key) for key in RENDER_AA_KEYS]
    if requested is not None and values != [requested, requested]:
        raise RuntimeError(f'Renderer AA request {requested} did not persist: {values}')
    return values


def read_json(path):
    return json.loads(path.read_text())


def load_arrays(path):
    import numpy as np

    with np.load(path, allow_pickle=False) as archive:
        return dict(archive)


def remote_snapshot_name(output, name, collection_protocol):
    # The remote endpoint uses a flat directory shared by successive jobs on a port.
    return f'{output.name}_{name}' if collection_protocol else name


def sweep_clearance(points, pivot, axis, angle, obstacles):
    """Conservatively bound each sector of a revolute door's collision hull."""
    import numpy as np

    points, pivot, axis = (np.asarray(value, dtype=float) for value in (points, pivot, axis))
    if (points.ndim != 2 or points.shape[1] != 3 or not len(points)
            or not all(np.isfinite(value).all() for value in (points, pivot, axis, angle))
            or not np.isclose(np.linalg.norm(axis), 1) or abs(angle) > 2 * np.pi):
        return None
    relative = points - pivot
    parallel = np.outer(relative @ axis, axis)
    radial = relative - parallel
    tangent = np.cross(axis, radial)
    radius = np.linalg.norm(radial, axis=1).max()
    sectors = max(1, int(np.ceil(abs(angle) / np.deg2rad(5))))
    width = angle / sectors
    # Every point is at most this far from its position at the sector midpoint.
    padding = 2 * radius * np.sin(abs(width) / 4)
    lower, upper = [], []
    for sector in range(sectors):
        theta = (sector + 0.5) * width
        hull = pivot + parallel + np.cos(theta) * radial + np.sin(theta) * tangent
        lower.append(hull.min(axis=0) - padding)
        upper.append(hull.max(axis=0) + padding)
    lower, upper = np.asarray(lower), np.asarray(upper)
    clearance = {}
    for name, bounds in obstacles.items():
        low, high = np.asarray(bounds, dtype=float)
        if not np.isfinite([low, high]).all() or (high < low).any():
            return None
        separation = np.maximum(np.maximum(lower - high, low - upper), 0)
        clearance[name] = float(np.linalg.norm(separation, axis=1).min())
    return clearance


def set_controlled_reclosure(open_state, door_joint):
    # Open.set_value samples a side using Python RNG, even for fully=True.
    rng = random.getstate()
    try:
        closed = open_state.set_value(False, fully=True)
        if closed:
            door_joint.set_vel(0.0)
        return bool(closed)
    finally:
        random.setstate(rng)


def verify_reclosed_truth(before, after):
    return (after['step'] == before['step'] + 1 and after['Open'] is False
            and all(after[key] == before[key] for key in ('Inside', 'ToggledOn')))


def align_sim_cuda_rng(states, device_count):
    if len(states) == device_count:
        return states
    if states and device_count and all(states[0].equal(value) for value in states[1:]):
        return [states[0]] * device_count
    raise ValueError(f"Cannot map {len(states)} non-equivalent simulator CUDA RNG states to {device_count} devices")


def load_targeted_restore(checkpoint, parent_trial):
    import numpy as np
    import torch
    from task40_memory_instrumentation import pack_rng, unpack_memory

    source = read_json(checkpoint / 'source.json')
    parent = read_json(parent_trial / 'trial.json')
    if (source['branch'] != parent_trial.name or parent['branch'] != source['branch']
            or parent['start_step'] != 0 or parent.get('origin') or parent['mode'] != 'A_normal'
            or parent['stop_reason'] == 'physx_crashed' or type(parent['instance']) is not int):
        raise ValueError('Expected the original full A_normal parent trial of the failure checkpoint')
    saved = torch.load(checkpoint / 'sim.pt', map_location='cpu', weights_only=False)
    decision, step, session = saved['decision'], saved['step'], saved['session']
    if (decision, step) != (source['decision'], source['step']) or not (0 < step < parent['end_step']):
        raise ValueError('Archived checkpoint is outside the parent trajectory or disagrees with source.json')
    for index in range(decision):
        record = read_json(parent_trial / f'{index:04d}' / 'decision.json')
        if (record['decision'], record['session'], record['start_step'], record['end_step']) != (
                index, session, index * 8, (index + 1) * 8):
            raise ValueError(f'Parent decision prefix is discontinuous at decision {index}')
    entry = parent_trial / f'{decision:04d}'
    record = read_json(entry / 'decision.json')
    if (record['decision'], record['start_step'], record['session'], record['language']) != (
            decision, step, session, source['language']) or step != decision * 8:
        raise ValueError('Parent pre-action decision does not match the archived checkpoint')
    observation = load_arrays(checkpoint / 'observation.npz')
    parent_observation = load_arrays(entry / 'observation.npz')
    if observation.keys() != parent_observation.keys() or any(
            value.dtype != parent_observation[name].dtype or not np.array_equal(value, parent_observation[name])
            for name, value in observation.items()):
        raise ValueError('Archived observation differs from the parent decision input')
    truth = read_json(parent_trial / 'initial.json')['truth']
    if truth['step'] != 0 or truth['Open']:
        raise ValueError('Expected a closed-door parent starting at physical step zero')
    with (parent_trial / 'truth.jsonl').open() as stream:
        for expected, line in enumerate(stream, start=1):
            truth = json.loads(line)
            if truth['step'] != expected or truth['Open']:
                raise ValueError('Parent truth is discontinuous or contains an earlier opening')
            if expected == step:
                break
    if (truth['step'] != step or truth['decision'] != decision - 1 or truth['ToggledOn']
            or any(truth[field] != record['state_before'][field]
                   for field in ('step', 'Open', 'ToggledOn', 'Inside', 'hinge'))):
        raise ValueError('Parent truth does not match the closed failure checkpoint boundary')
    memory = (checkpoint / 'memory.pt').read_bytes()
    key = (str(session['session_id']), str(session['env_ids'][0]), str(session['episode_ids'][0]))
    if unpack_memory(memory)['slots'] != {key[:2]: (key, decision)}:
        raise ValueError('Archived memory is for a different session or decision')
    rng = pack_rng(torch.load(checkpoint / 'policy_rng_online.pt', map_location='cpu', weights_only=False))
    lineage = {'source': 'directed_restore', 'origin_id': f"instance={parent['instance']}/seed={parent['seed']}",
               'checkpoint': str(checkpoint), 'prefix': str(parent_trial), 'parent_branch': parent['branch'],
               'instance': parent['instance'], 'source_seed': parent['seed'],
               'start_step': step, 'start_decision': decision, 'session': session,
               'archived_observation_matches_parent': True,
               'memory_sha256': sha256(memory).hexdigest(), 'rng_sha256': sha256(rng).hexdigest()}
    return saved, observation, truth, memory, rng, lineage


def restore_targeted_policy(client, session, memory, rng, decision):
    reply = client.call_endpoint('task40_restore_memory', {
        'session': session, 'memory': memory, 'rng': rng, 'decision': decision})
    clock = reply.get('clock', reply)
    key = [str(session['session_id']), str(session['env_ids'][0]), str(session['episode_ids'][0])]
    if clock.get('decision') != decision or ('key' in clock and list(clock['key'] or ()) != key):
        raise RuntimeError(f'Task40 policy restore clock disagrees with archived decision {decision}: {clock}')
    return {**reply, 'clock': clock}


def replay_policy_history(client, parent, session, decision):
    client.reset()
    inputs = sha256()
    for index in range(decision):
        entry = parent / f'{index:04d}'
        record = read_json(entry / 'decision.json')
        vision = entry / 'observation.npz'
        inputs.update(sha256(vision.read_bytes()).digest())
        inputs.update(json.dumps(record['language'], sort_keys=True).encode())
        client.call_endpoint('get_action', {'observation': load_arrays(vision) | record['language'],
                                             'options': session})
    snapshot = client.call_endpoint('task40_memory_snapshot', {'session': session})
    return snapshot['memory'], inputs.hexdigest()


def verify_targeted_trace(trace, memory, rng):
    import torch
    from task40_memory_instrumentation import BANKS, _clock, unpack_memory

    expected = unpack_memory(memory)
    observed = unpack_memory(trace['memory_base'])
    banks_equal = _clock(expected) == _clock(observed) and all(
        left.dtype == right.dtype and torch.equal(left, right)
        for bank in BANKS
        for key, slots in expected[bank]['bank'].items()
        for (_, left), (_, right) in zip(slots, observed[bank]['bank'][key]))
    py1, np1, cpu1, cuda1 = unpack_memory(rng)
    py2, np2, cpu2, cuda2 = unpack_memory(trace['rng_before'])
    rng_equal = (py1 == py2 and np1[0] == np2[0] and np1[2:] == np2[2:]
                 and torch.equal(np1[1], np2[1]) and torch.equal(cpu1, cpu2)
                 and len(cuda1) == len(cuda2) and all(torch.equal(a, b) for a, b in zip(cuda1, cuda2)))
    return {'both_memory_banks_equal': banks_equal, 'rng_equal': rng_equal}


def observation_correspondence(restored, archived):
    import numpy as np

    if restored.keys() != archived.keys():
        raise ValueError('Restored and archived observation fields differ')
    comparison = {}
    for name, current in restored.items():
        expected = archived[name]
        if current.shape != expected.shape or current.dtype != expected.dtype:
            raise ValueError(f'Restored and archived observation shape or dtype differs for {name}')
        delta = np.abs(np.subtract(current, expected, dtype=np.float64))
        comparison[name] = {'exact': np.array_equal(current, expected), 'mae': float(delta.mean()),
                            'max_abs': float(delta.max())}
    return comparison


def configure_restore_rendering(sim, *, targeted):
    if not targeted:
        return None
    previous = sim.get_block_on_render()
    # Physics snapshots do not flush the in-flight Hydra render products.
    sim.set_block_on_render(True)
    if not sim.get_block_on_render():
        raise RuntimeError('Directed restore could not enable the Isaac Sim render barrier')
    return previous


def compare_targeted_repeats(left, right):
    """Compare both runs at shared absolute decision and physical steps."""
    import numpy as np

    folders = (left, right)
    trials = [read_json(folder / 'trial.json') for folder in folders]
    initial = [read_json(folder / 'initial.json') for folder in folders]
    restore = [read_json(folder / 'restore.json') for folder in folders]
    entries = []
    for folder, entry, trial in zip(folders, initial, trials, strict=True):
        check = folder / 'policy_entry_check.json'
        if check.is_file():
            entries.append(read_json(check))
        elif entry['policy']['policy_weights'] == 'GR00T-N1.6-BEHAVIOR1k':
            entries.append({'rng_matches_archive': entry['policy']['rng_sha256'] == trial['lineage']['rng_sha256']})
        else:
            raise FileNotFoundError(check)
    traces = [{path.parent.name: path.parent for path in folder.glob('*/decision.json')} for folder in folders]
    action_differences = []
    for decision in sorted(traces[0].keys() & traces[1].keys(), key=int):
        a, b = (trace[decision] for trace in traces)
        records = [read_json(path / 'decision.json') for path in (a, b)]
        if records[0]['start_step'] != records[1]['start_step']:
            raise ValueError(f'Repeated decision {decision} has mismatched physical steps')
        executed = [load_arrays(path / 'executed.npz')['robot_r1'] for path in (a, b)]
        with np.load(a / 'observation.npz', allow_pickle=False) as ra, \
                np.load(b / 'observation.npz', allow_pickle=False) as rb:
            rgb = observation_correspondence({key: ra[key] for key in CAMERA_KEYS},
                                             {key: rb[key] for key in CAMERA_KEYS})
        count = min(map(len, executed))
        action_differences.append({
            'decision': int(decision), 'shared_executed_steps': count,
            'policy_input_rgb': rgb,
            'proposed': observation_correspondence(load_arrays(a / 'proposed.npz'),
                                                   load_arrays(b / 'proposed.npz')),
            'executed': observation_correspondence({'robot_r1': executed[0][:count]},
                                                   {'robot_r1': executed[1][:count]})['robot_r1'],
        })

    physics = []
    for folder, entry in zip(folders, initial, strict=True):
        with (folder / 'truth.jsonl').open() as stream:
            rows = [entry['truth'], *(json.loads(line) for line in stream)]
        physics.append({row['step']: row for row in rows})
    shared = sorted(physics[0].keys() & physics[1].keys())
    states = [(physics[0][step], physics[1][step]) for step in shared]
    robot = [(a['robots']['robot_r1'], b['robots']['robot_r1']) for a, b in states]
    flags = ('Open', 'ToggledOn', 'Inside')
    return {
        'runs': [folder.name for folder in folders],
        'policy_entry_checks': entries,
        'first_memory_clock_equal': (initial[0]['policy']['clock'] == initial[1]['policy']['clock']
                                     if 'clock' in initial[0]['policy'] else None),
        'camera_poses_exact': initial[0]['camera_poses'] == initial[1]['camera_poses'],
        'controller_restore_checks': [item['controllers_equal'] for item in restore],
        'renderer_aa': [item['renderer_aa'] for item in restore],
        'renderer_block_on_render': [item.get('renderer_block_on_render') for item in restore],
        'first_observation': observation_correspondence(
            load_arrays(left / 'first_raw_restored.npz'), load_arrays(right / 'first_raw_restored.npz')),
        'actions': action_differences,
        'shared_physical_steps_including_anchor': len(shared),
        'max_hinge_abs_rad': max(abs(a['hinge'] - b['hinge']) for a, b in states),
        'max_robot_position_distance_m': max(float(np.linalg.norm(np.subtract(a['position'], b['position'])))
                                             for a, b in robot),
        'max_robot_joint_abs': max(float(np.abs(np.subtract(a['joint_positions'], b['joint_positions'])).max())
                                   for a, b in robot),
        'max_eef_position_distance_m': {
            arm: max(float(np.linalg.norm(np.subtract(a['end_effectors'][arm][0], b['end_effectors'][arm][0])))
                     for a, b in robot) for arm in ('left', 'right')},
        'first_truth_mismatch_step': {
            flag: next((a['step'] for a, b in states if a[flag] != b[flag]), None) for flag in flags},
        'outcomes': [{key: trial[key] for key in ('start_step', 'end_step', 'decisions', 'stop_reason',
                                                 'first_open_step', 'first_error', 'door_contact_steps',
                                                 'post_event_action_chunks_completed')} for trial in trials],
    }


def main(args):
    sys.path[:0] = [os.environ.get('OMNIGIBSON_ROOT', str(args.behavior / 'OmniGibson')),
                   os.environ.get('BDDL_ROOT', str(args.behavior / 'bddl')), str(args.groot)]
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
    from omnigibson.utils import transform_utils as transforms
    from omnigibson.utils.constants import JointType
    from omnigibson.utils.usd_utils import GripperRigidContactAPI, PoseAPI
    from PIL import Image, ImageDraw

    checkpoint = getattr(args, 'restore_checkpoint', None)
    targeted = checkpoint is not None
    if targeted:
        anchor, archived_observation, parent_truth, policy_memory, policy_rng, lineage = load_targeted_restore(
            checkpoint, args.parent_trial)
        initial_physical = read_json(checkpoint / 'physical.json')
        archived_language = read_json(checkpoint / 'source.json')['language']
        start = {'clock': anchor['session'], 'seed': lineage['source_seed'], 'instance': lineage['instance']}
    else:
        start = read_json(args.anchor / 'start.json')
    session = start['clock']
    random.seed(start['seed'])
    np.random.seed(start['seed'])
    torch.manual_seed(start['seed'])
    os.environ['GROOT_EVAL_INSTANCE_IDS'] = ','.join(map(str, args.instances)) if args.phase == 'collect' else str(start['instance'])
    env = BEHAVIORGr00tEnv('make_microwave_popcorn')
    if args.phase != 'collect':
        if targeted:
            actual_instance = env._all_test_instances[env._instance_indices_this_env[env._instance_idx_pointer]]
            if actual_instance != start['instance']:
                raise RuntimeError(f"Simulator selected instance {actual_instance}, expected {start['instance']}")
        env.reset(seed=start['seed'])
    block_on_render_before = configure_restore_rendering(og.sim, targeted=targeted)
    settings = carb.settings.get_settings()
    aa_initial = read_restore_aa(settings, None)
    aa_after_request = (configure_restore_aa(og.app, settings, args.renderer_aa)
                        if args.renderer_aa is not None else aa_initial)
    gi_history_initial = settings.get(GI_HISTORY_KEY) if args.phase == 'renderer_probe' else None
    indirect_diffuse_initial = [settings.get(key) for key in INDIRECT_DIFFUSE_KEYS] if args.phase == 'renderer_probe' else None
    if args.renderer_gi_history is not None:
        if settings.get('/rtx/indirectDiffuse/denoiser/method') != 0 or not settings.get(INDIRECT_DIFFUSE_KEYS[0]):
            raise RuntimeError('GI history limit requires active indirect diffuse lighting and NVRTD denoiser')
        settings.set_int(GI_HISTORY_KEY, args.renderer_gi_history)
    if args.renderer_disable_indirect_diffuse:
        if indirect_diffuse_initial[0] is not True:
            raise RuntimeError(f'Expected indirect diffuse lighting at startup: {indirect_diffuse_initial}')
        for key in INDIRECT_DIFFUSE_KEYS:
            settings.set_bool(key, False)
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
    camera_keys = CAMERA_KEYS
    client = None
    if args.phase != 'renderer_probe':
        client = PolicyClient(host=args.policy_host, port=args.port)
        client.socket.setsockopt(zmq.RCVTIMEO, 300000)
        if args.collection_protocol and not args.trace_memory and not args.official_policy:
            # Older workbench servers omit capabilities from the recorded identity.
            # Ask the reserved endpoint before making the first stateful call.
            try:
                client.call_endpoint('task40_memory_snapshot', {
                    'session': {'session_id': args.output.name, 'env_ids': [0],
                                'episode_ids': ['capability_probe']}})
            except RuntimeError as error:
                if 'Unknown endpoint: task40_memory_snapshot' not in str(error):
                    raise
            else:
                args.trace_memory = True

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

    def restore(saved, reference, folder, *, reset_accumulation=False):
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
        cuda_states = align_sim_cuda_rng(saved['cuda_rng'], torch.cuda.device_count())
        torch.cuda.set_rng_state_all(cuda_states)
        write_json(folder / 'sim_cuda_rng.json', {'saved_devices': len(saved['cuda_rng']),
                   'restored_devices': len(cuda_states),
                   'mapping': 'unchanged' if len(saved['cuda_rng']) == len(cuda_states) else 'identical_states_only'})
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
        blocking_readbacks = []
        render_controls = []
        if reset_accumulation:
            import omni.usd

            omni.usd.get_context().reset_renderer_accumulation()
        for render_index in range(6):
            if args.renderer_aa is not None:
                read_restore_aa(settings, args.renderer_aa)
            og.sim.render()
            if args.renderer_aa is not None:
                render_settings.append(read_restore_aa(settings, args.renderer_aa))
            if targeted:
                blocking_readbacks.append(bool(og.sim.get_block_on_render()))
            observation = preprocess_obs(env, base.get_obs()[0])
            if args.renderer_audit:
                images = {key: np.asarray(observation[key]).astype(np.int16) for key in camera_keys}
                if previous_images is not None:
                    render_deltas.append({key: float(np.abs(images[key] - previous_images[key]).mean())
                                          for key in camera_keys})
                previous_images = images
            if args.phase == 'renderer_probe' and args.renderer_aa is None:
                render_settings.append(read_restore_aa(settings, args.renderer_aa))
            if args.phase == 'renderer_probe':
                render_controls.append({key: settings.get(key) for key in (
                    '/rtx/rendermode', *INDIRECT_DIFFUSE_KEYS,
                    '/rtx/indirectDiffuse/denoiser/method', GI_HISTORY_KEY, '/rtx/post/histogram/enabled',
                    '/rtx/post/histogram/tau')})
                if args.renderer_gi_history is not None and (
                        render_controls[-1][GI_HISTORY_KEY] != args.renderer_gi_history):
                    raise RuntimeError(f'Renderer GI history limit did not persist: {render_controls[-1]}')
                if args.renderer_disable_indirect_diffuse and any(
                        render_controls[-1][key] is not False for key in INDIRECT_DIFFUSE_KEYS):
                    raise RuntimeError(f'Indirect diffuse disable did not persist: {render_controls[-1]}')
                np.savez_compressed(folder / f'render_{render_index + 1:02d}.npz', **features(batch(observation)))
        cached = archived_observation if targeted else load_arrays(args.anchor / 'anchor_live_observation.npz')
        camera_rgb_mae = {key.split('.')[-1].removesuffix('_256_256'): float(np.mean(np.abs(
            np.asarray(observation[key], dtype=np.float32) -
            np.asarray(cached[key], dtype=np.float32)))) for key in camera_keys}
        assert sim_time == float(og.sim.current_time)
        audit = {'position': position, 'rotation': rotation, 'joints': joints,
                   'controllers_equal': controllers,
                   'renderer_aa': settings.get(RENDER_AA_KEYS[0]),
                   'renderer_aa_readbacks': [values[0] for values in render_settings],
                   'renderer_aa_defaults_readbacks': [values[1] for values in render_settings],
                   'consecutive_render_rgb_mae': render_deltas,
                   'settle_physics_steps': 1,
                   'settle_sim_time_delta_s': sim_time - sim_time_before_settle,
                   'camera_rgb_mae_vs_anchor': camera_rgb_mae,
                   'camera_poses': {key: sensor.get_position_orientation() for key, sensor in sensors.items()},
                   'truth': read_state(base, states)}
        restored_input = features(batch(observation))
        if targeted:
            audit['renderer_block_on_render'] = {
                'before': block_on_render_before, 'after_each_render': blocking_readbacks}
            audit['observation_correspondence'] = observation_correspondence(restored_input, cached)
        if args.phase == 'renderer_probe':
            audit.update(reset_accumulation_called=reset_accumulation, render_controls=render_controls)
        write_json(folder / 'restore.json', audit)
        np.savez_compressed(folder / 'first_raw_restored.npz', **restored_input)
        if targeted and not all(blocking_readbacks):
            raise RuntimeError('Isaac Sim render barrier changed during directed restore')
        assert max((*position.values(), *rotation.values(), *joints.values())) < 1e-2 and controllers
        if not args.renderer_audit:
            assert max(camera_rgb_mae.values()) < 25.0, f'Camera RGB mismatch after restore: {camera_rgb_mae}'
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

    def reclosure_clearance(row):
        margin_m = 0.02
        open_state = microwave.states[states['Open']]
        both_sides, joints, directions = open_state.relevant_joints_info
        if (both_sides or len(joints) != 1 or directions[0] not in (-1, 1)
                or door_joint.joint_type != JointType.JOINT_REVOLUTE
                or row['Inside'] or row['ToggledOn']
                or any(contact['body'] == door.prim_path for arm in row['interaction']['contacts'].values()
                       for contact in arm)):
            return None
        parent = next((link for link in microwave.links.values() if link.prim_path == door_joint.body0), None)
        if parent is None or door_joint.axis not in ('X', 'Y', 'Z'):
            return None
        axis = torch.eye(3)['XYZ'.index(door_joint.axis)]
        joint_pos, joint_quat = transforms.pose_transform(
            *parent.get_position_orientation(), door_joint.local_position_0, door_joint.local_orientation_0)
        child_pos, child_quat = transforms.pose_transform(
            *door.get_position_orientation(), door_joint.local_position_1, door_joint.local_orientation_1)
        axis_world = transforms.quat_apply(joint_quat, axis)
        if (float(torch.linalg.vector_norm(child_pos - joint_pos)) > 0.005
                or not torch.allclose(axis_world, transforms.quat_apply(child_quat, axis), atol=0.005, rtol=0)):
            return None
        hull = door.collision_boundary_points_world
        if hull is None:
            return None
        objects = {link.name: link for arm, fingers in robot.finger_links.items()
                   for link in (*fingers, robot.eef_links[arm])}
        objects[bag.name] = bag
        obstacles = {name: [bound.detach().cpu().numpy() for bound in obj.aabb] for name, obj in objects.items()}
        closed_end = float(door_joint.lower_limit if directions[0] == 1 else door_joint.upper_limit)
        clearance = sweep_clearance(hull.detach().cpu().numpy(), joint_pos.detach().cpu().numpy(),
                                    axis_world.detach().cpu().numpy(), closed_end - row['hinge'], obstacles)
        if clearance is None or min(clearance.values()) < margin_m:
            return None
        return {'min_required_m': margin_m, 'lower_bound_m': clearance, 'closed_hinge_rad': closed_end,
                'door_link': door.prim_path, 'hinge_joint': door_joint.prim_path}

    def save_failure(name, branch, decision, observation, evidence, fresh=False):
        folder = args.output / 'checkpoints' / name
        folder.mkdir(exist_ok=True)
        torch.save(snapshot(decision), folder / 'sim.pt')
        write_json(folder / 'physical.json', physical())
        observed = batch(observation)
        np.savez_compressed(folder / 'observation.npz', **features(observed))
        remote_name = remote_snapshot_name(args.output, name, args.collection_protocol)
        check = client.call_endpoint('save_failure', {'name': remote_name, 'branch': branch, 'decision': decision, 'fresh': fresh})
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

    if not targeted:
        anchor = torch.load(args.anchor / 'sim_checkpoint.pt', map_location='cpu', weights_only=False)
        initial_physical = read_json(args.anchor / 'anchor.json')

    if args.phase == 'renderer_probe':
        output = args.output / 'metrics'
        output.mkdir(exist_ok=True)
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
                   'aa_initial': aa_initial[0], 'aa_requested': args.renderer_aa,
                   'aa_after_request': aa_after_request[0],
                   'aa_defaults_initial': aa_initial[1], 'aa_defaults_after_request': aa_after_request[1],
                   'copies': args.restore_repeats, 'blocks': args.cycles,
                   'compare_accumulation_reset': args.compare_accumulation_reset,
                   'gi_history_initial': gi_history_initial, 'gi_history_requested': args.renderer_gi_history,
                   'indirect_diffuse_initial': indirect_diffuse_initial,
                   'indirect_diffuse_disabled': args.renderer_disable_indirect_diffuse})
        groups = {False: []}
        if args.compare_accumulation_reset:
            groups[True] = []
        try:
            for i, reset_history in enumerate(
                    list(groups) * args.restore_repeats, start=1):
                label = 'reset' if reset_history else 'control'
                folder = output / f'{label}_repeat_{len(groups[reset_history]) + 1:02d}'
                folder.mkdir()
                groups[reset_history].append(folder)
                write_json(output / 'active_stage.json', {'stage': 'restore_sim', 'repeat': i, 'trial': label})
                restore(anchor, initial_physical, folder, reset_accumulation=reset_history)
                with (folder / 'truth.jsonl').open('x') as stream:
                    for offset, block in enumerate(blocks):
                        decision = anchor['decision'] + offset
                        for action in block:
                            observation, _, terminated, truncated, _ = env.step(action)
                            stream.write(json.dumps(truth(decision)) + '\n')
                            if terminated or truncated or env._physx_crashed:
                                raise RuntimeError('Fixed-action restore probe encountered a simulator stop')
                        np.savez_compressed(folder / f'observation_{decision:04d}.npz', **features(batch(observation)))
                        write_json(output / 'active_stage.json', {'stage': 'fixed_action_replay',
                                   'trial': label, 'repeat': i, 'decisions': offset + 1, 'steps': (offset + 1) * 8})
                print(f'TASK40_RENDERER_PROBE_DONE repeat={i} reset={reset_history} steps={args.cycles * 8}', flush=True)
            for reset_history, repeats in groups.items():
                if len(repeats) != 2:
                    continue
                left, right = repeats
                tracks = []
                for folder in repeats:
                    with (folder / 'truth.jsonl').open() as stream:
                        tracks.append([json.loads(line) for line in stream])
                rows = list(zip(*tracks, strict=True))
                if any(a['step'] != b['step'] for a, b in rows):
                    raise ValueError('Fixed-action repeats do not share the same physical time')
                image_diff = {}
                for name in sorted((*left.glob('render_*.npz'), *left.glob('observation_*.npz'))):
                    image_diff[name.name] = observation_correspondence(
                        load_arrays(name), load_arrays(right / name.name))
                robot = [(a['robots']['robot_r1'], b['robots']['robot_r1']) for a, b in rows]
                audit_name = 'accumulation_reset_audit.json' if reset_history else 'fixed_action_audit.json'
                first_state = [read_json(folder / 'restore.json') for folder in repeats]
                write_json(output / audit_name, {
                    'shared_steps': len(rows), 'archived_actions_exactly_verified': True,
                    'run_order': [folder.name for folder in repeats],
                    'first_state': first_state,
                    'camera_poses_exact': first_state[0]['camera_poses'] == first_state[1]['camera_poses'],
                    'observations': image_diff,
                    'physics_exact_excluding_absolute_sim_time': all(
                        all(value == b[key] for key, value in a.items() if key != 'sim_time_s')
                        for a, b in rows),
                    'absolute_sim_time_offset_s': rows[0][1]['sim_time_s'] - rows[0][0]['sim_time_s'],
                    'max_hinge_abs_rad': max(abs(a['hinge'] - b['hinge']) for a, b in rows),
                    'max_robot_joint_abs': max(float(np.max(np.abs(np.subtract(
                        a['joint_positions'], b['joint_positions'])))) for a, b in robot),
                    'max_eef_position_distance_m': {
                        arm: max(float(np.linalg.norm(np.subtract(a['end_effectors'][arm][0],
                                  b['end_effectors'][arm][0]))) for a, b in robot)
                        for arm in ('left', 'right')},
                    'truth_mismatch_steps': {
                        key: [a['step'] for a, b in rows if a[key] != b[key]]
                        for key in ('Open', 'ToggledOn', 'Inside', 'interaction')},
                    'door_contact_steps': [[row['step'] for row in track if any(
                        contact['body'] == row['interaction']['door_body']
                        for arm in row['interaction']['contacts'].values() for contact in arm)]
                        for track in tracks],
                })
            write_json(output / 'active_stage.json', {'stage': 'complete', 'decisions': args.cycles,
                       'steps': args.cycles * 8, 'repeats': sum(map(len, groups.values()))})
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

    targeted_lineage = {}

    def run(label, mode, seed, origin=None, failure_name=None, fresh_instance=None):
        nonlocal session, microwave, bag, toggle, door_joint, door, policy_memory
        started = time.monotonic()
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
        else:
            saved, reference = anchor, initial_physical
        if targeted:
            write_json(folder / 'lineage.json', lineage)
            targeted_lineage[str(folder.resolve())] = lineage
            write_json(args.output / 'metrics' / 'lineage_metadata.json', targeted_lineage)
        elif args.official_policy:
            client.reset()
            prepared = {'policy_weights': 'GR00T-N1.6-BEHAVIOR1k', 'seed': seed}
        else:
            prepared = client.call_endpoint('prepare_pair',
                                            {'mode': mode, 'seed': seed, 'origin': origin, 'fresh': fresh})
        if not fresh:
            write_json(args.output / 'metrics' / 'active_stage.json', {'trial': name, 'stage': 'restore_sim'})
            observation = restore(saved, reference, folder)
        start_decision = saved['decision']
        before = truth(start_decision - 1)
        if targeted:
            language = {key: value for key, value in batch(observation).items() if key.startswith('annotation.')}
            truth_fields = ('step', 'decision', 'Open', 'ToggledOn', 'Inside')
            correspondence = {key: before[key] == parent_truth[key] for key in truth_fields}
            correspondence['hinge_abs_rad'] = abs(before['hinge'] - parent_truth['hinge'])
            write_json(folder / 'initial_correspondence.json', {
                'language_equal': language == archived_language, 'source_truth': parent_truth,
                'restored_truth': before, 'truth_correspondence': correspondence})
            if (language != archived_language or not all(correspondence[key] for key in truth_fields)
                    or correspondence['hinge_abs_rad'] > 1e-5):
                raise RuntimeError('Restored initial truth or language disagrees with the verified parent boundary')
            write_json(args.output / 'metrics' / 'active_stage.json', {'trial': name, 'stage': 'prepare_policy'})
            if args.official_policy:
                client.reset()
                prepared = {'policy_weights': 'GR00T-N1.6-BEHAVIOR1k', 'seed': seed,
                            **client.call_endpoint('task40_restore_rng', {'rng': policy_rng})}
            else:
                if args.rebuild_policy_history:
                    policy_memory, prefix_hash = replay_policy_history(client, args.parent_trial, session, start_decision)
                    (folder / 'policy_rebuilt.pt').write_bytes(policy_memory)
                    lineage.update(replay_decisions=start_decision, replay_input_sha256=prefix_hash,
                                   rebuilt_memory_sha256=sha256(policy_memory).hexdigest())
                    write_json(folder / 'lineage.json', lineage)
                prepared = restore_targeted_policy(client, session, policy_memory, policy_rng, start_decision)
                if args.rebuild_policy_history:
                    prepared['anchor_source'] = 'checkpoint_native_observation_replay'
            prepared['seed'] = seed
        write_json(args.output / 'metrics' / 'active_stage.json', {'trial': name, 'stage': 'closed_loop'})
        assert not before['Open'] and not before['ToggledOn']
        write_json(folder / 'initial.json', {'policy': prepared, 'truth': before,
                   'camera_poses': {key: sensor.get_position_orientation() for key, sensor in sensors.items()}})
        print('TASK40_PAIRED_START', name, seed, start_decision, flush=True)
        reason = 'decision_limit'
        steps = 0
        contact_history = []
        closed_contact_peak = 0.0
        hinges = []
        candidate = None
        approach_candidate = None
        first_error = None
        first_error_ordinal = None
        first_open_step = None
        first_open_ordinal = None
        first_task_success_step = None
        post_event_chunks_completed = 0
        hard_stop = False
        pilot = getattr(args, 'controlled_reclosure', False)
        marker_path = folder / 'controlled_reclosure.json'
        marker = {'schema_version': 1, 'source': 'controlled', 'branch': name, 'instance': fresh_instance,
                  'seed': seed, 'origin_id': f'instance={fresh_instance}/seed={seed}',
                  'status': 'not_attempted', 'valid': False} if pilot else None
        open_input = None
        video_path = args.output / 'videos' / f'{name}.mp4'
        with (folder / 'truth.jsonl').open('x', buffering=1) as truth_stream, av.open(str(video_path), 'w') as container:
            video = container.add_stream('libx264', rate=30)
            video.width, video.height, video.pix_fmt = 768, 256, 'yuv420p'
            max_cycles = args.cycles
            if args.collection_protocol and not args.full_task_eval:
                max_cycles += args.post_event_action_chunks
            for ordinal in range(max_cycles):
                if ordinal >= args.cycles and first_error_ordinal is None:
                    break
                decision = start_decision + ordinal
                entry = folder / f'{decision:04d}'
                pre_action = before if ordinal == 0 else row
                inputs = batch(observation)
                language = save_policy_input(entry, inputs)
                if args.trace_memory:
                    trace = client.call_endpoint('task40_trace_infer', {
                        'observation': inputs, 'session': session, 'decision': decision,
                        'include_base': ordinal == 0, 'detail': 'compact'})
                    if targeted and ordinal == 0:
                        check = verify_targeted_trace(trace, policy_memory, policy_rng)
                        write_json(folder / 'policy_entry_check.json', check)
                        if not all(check.values()):
                            raise RuntimeError('First traced forward used different memory banks or policy RNG')
                    proposed = trace.pop('actions')
                    artifacts = {}
                    for key in ('memory_base', 'memory_delta', 'rng_before'):
                        if key in trace:
                            content = trace.pop(key)
                            path = entry / f'{key}.pt'
                            with path.open('xb') as stream:
                                stream.write(content)
                            artifacts[key] = {'file': path.name, 'sha256': sha256(content).hexdigest()}
                    assert trace['decision'] == decision
                    diagnostics = {'model_features': trace['model_features'], 'retrieval': trace['retrieval'],
                                   'memory_after': {'decision': decision + 1}, 'instrumentation': trace['trace'],
                                   'memory_artifacts': artifacts}
                elif args.official_policy:
                    proposed, _ = client.call_endpoint('get_action', {'observation': inputs, 'options': session})
                    diagnostics = {'memory_after': {'decision': decision + 1}, 'memory_applicable': False}
                else:
                    proposed, diagnostics = client.call_endpoint('infer_pair', {'observation': inputs,
                        'session': session, 'branch': name, 'decision': decision, 'first': ordinal == 0})
                arrays = diagnostics.pop('model_features', None)
                if args.remote_policy and arrays is None and not args.official_policy:
                    raise RuntimeError('Remote policy must run with diagnostic artifact transfer enabled')
                if arrays is not None:
                    np.savez_compressed(entry / 'model_features.npz', **arrays)
                    if ordinal == 0:
                        np.savez_compressed(folder / 'first_expert.npz', **arrays)
                assert diagnostics['memory_after']['decision'] == decision + 1
                assert all(value.shape[1] >= 8 and np.isfinite(value).all() for value in proposed.values())
                np.savez_compressed(entry / 'proposed.npz', **proposed)
                if pilot and open_input is None and first_open_step is not None and pre_action['Open']:
                    open_input = {'decision': decision, 'step': pre_action['step'],
                                  'observation': f'{entry.name}/observation.npz'}
                closed_input = pilot and marker['status'] == 'awaiting_closed_decision'
                applied_this_block = False
                actions = []
                for index in range(8):
                    action = {key: value[0, index] for key, value in proposed.items()}
                    actions.append(preprocess_action(action)['robot_r1'].cpu().numpy())
                    if (pilot and marker['status'] == 'not_attempted' and index == 7
                            and ordinal + 1 < args.cycles and first_error is None
                            and open_input is not None and row['Open']):
                        clearance = reclosure_clearance(row)
                        if clearance is not None:
                            marker.update(status='attempted', first_open_step=first_open_step,
                                          open_policy_input=open_input, action_decision=decision,
                                          clearance=clearance, before=row,
                                          expected_after_step=row['step'] + 1)
                            write_json(marker_path, marker)
                            marker['setter_succeeded'] = set_controlled_reclosure(microwave.states[states['Open']], door_joint)
                            applied_this_block = True
                    observation, _, terminated, truncated, info = env.step(action)
                    row = truth(decision)
                    row['task_success'] = bool(info['success']) if 'success' in info else None
                    if args.full_task_eval and row['task_success'] is None:
                        raise RuntimeError('Full task evaluation requires the environment success field')
                    if args.full_task_eval and 'task_progress' in info:
                        row['task_progress'] = float(info['task_progress'])
                    if row['task_success'] and first_task_success_step is None:
                        first_task_success_step = row['step']
                    if applied_this_block:
                        row['controlled_reclosure'] = True
                        matched = verify_reclosed_truth(marker['before'], row)
                        usable = marker['setter_succeeded'] and matched and not (
                            terminated or truncated or env._physx_crashed)
                        marker.update(after=row, status='awaiting_closed_decision' if usable else 'invalid')
                        if not usable:
                            marker['reason'] = 'setter_failed' if not marker['setter_succeeded'] else (
                                'post_intervention_truth_mismatch' if not matched else 'environment_ended')
                        write_json(marker_path, marker)
                    truth_stream.write(json.dumps(row) + '\n')
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
                        if first_open_step is None:
                            closed_contact_peak = max(closed_contact_peak, abs(row['hinge'] - before['hinge']))
                    hinges.append(abs(row['hinge'] - before['hinge']))
                    image = Image.fromarray(np.concatenate([observation[key] for key in camera_keys], axis=1))
                    video_label = f"{name}  d={ordinal + 1:02d}  step={steps:03d}  Open={int(row['Open'])}  On={int(row['ToggledOn'])}"
                    if pilot and marker['status'] != 'not_attempted':
                        video_label += '  CONTROLLED'
                    ImageDraw.Draw(image).text((5, 5), video_label,
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
                           'instance': lineage['instance'] if targeted else fresh_instance, 'seed': seed, **diagnostics})
                if closed_input:
                    valid = (decision == marker['action_decision'] + 1
                             and verify_reclosed_truth(marker['before'], pre_action)
                             and len(actions) == 8 and not env._physx_crashed)
                    marker.update(status='valid' if valid else 'invalid', valid=valid,
                                  closed_policy_input={'decision': decision, 'step': pre_action['step'],
                                                       'observation': f'{entry.name}/observation.npz',
                                                       'record': f'{entry.name}/decision.json'})
                    if not valid:
                        marker['reason'] = 'closed_decision_incomplete_or_mismatched'
                    write_json(marker_path, marker)
                    if not hard_stop:
                        reason = 'controlled_reclosure_complete' if valid else 'controlled_reclosure_invalid'
                    break
                if applied_this_block and marker['status'] == 'invalid':
                    if not hard_stop:
                        reason = 'controlled_reclosure_invalid'
                    break
                if first_error_ordinal is not None and ordinal > first_error_ordinal and len(actions) == 8:
                    post_event_chunks_completed = ordinal - first_error_ordinal
                if hard_stop:
                    break
                if not args.full_task_eval and first_error is None and first_open_step is not None:
                    reason = 'opened'
                    if (not fresh and (not targeted or args.observe_open_chunks or args.stop_at_first_open)
                            and ordinal - first_open_ordinal >= args.observe_open_chunks):
                        break
                if not args.full_task_eval and first_error is not None:
                    if post_event_chunks_completed >= args.post_event_action_chunks:
                        reason = 'opened_after_premature_toggle' if first_open_step is not None else 'premature_toggle_observed'
                    else:
                        reason = 'post_event_observation'
                    if post_event_chunks_completed >= args.post_event_action_chunks:
                        break
                if (args.capture_approach and fresh and approach_candidate is None
                        and not contact_history and first_open_step is None and first_error is None
                        and ordinal + 1 < args.cycles and abs(row['hinge'] - before['hinge']) <= .005):
                    from groot_web_launch import task40_approach_distance
                    distance = task40_approach_distance(row)
                    if distance <= .12:
                        approach_candidate = name.removesuffix('_A_normal') + '_approach'
                        save_failure(approach_candidate, name, decision + 1, observation, {
                            'capture_stage': 'approach', 'door_distance_m': distance, 'door_contact_steps': [],
                            'final_open': False, 'final_toggled_on': False}, fresh=True)
                if fresh and args.collection_protocol:
                    capture_attempt = (first_open_step is None and first_error is None
                                       and ordinal + 1 < args.cycles and closed_contact_peak >= 0.005)
                    capture_peak = closed_contact_peak
                else:
                    capture_peak = max(hinges)
                    capture_attempt = (capture_peak >= 0.01 and bool(contact_history)
                                       and row['step'] - contact_history[-1] >= 2)
                if (failure_name and candidate is None and not row['Open'] and not row['ToggledOn']
                        and len(contact_history) >= 3 and capture_attempt):
                    candidate = failure_name
                    save_failure(candidate, name, decision + 1, observation,
                                 {'door_contact_steps': contact_history.copy(), 'max_hinge_excursion_rad': capture_peak,
                                  'final_open': row['Open'], 'final_toggled_on': row['ToggledOn']}, fresh=fresh)
                if ordinal == 0 or (ordinal + 1) % 10 == 0:
                    write_json(args.output / 'metrics' / 'active_stage.json',
                               {'trial': name, 'stage': 'closed_loop', 'decisions': ordinal + 1, 'steps': steps})
                    print('TASK40_PAIRED_PROGRESS', name, ordinal + 1, steps, len(contact_history), max(hinges), flush=True)
            executed_decisions = decision - start_decision + 1
            for packet in video.encode():
                container.mux(packet)
        if (not args.full_task_eval and not hard_stop and first_error is not None
                and post_event_chunks_completed < args.post_event_action_chunks):
            reason = 'decision_limit_after_event'
        assert steps == base._current_step - saved['step']
        if pilot:
            if marker['status'] == 'not_attempted':
                marker.update(first_open_step=first_open_step, open_policy_input=open_input,
                              reason='no_open_policy_input' if open_input is None else 'no_eligible_boundary')
            write_json(marker_path, marker)
        online_prefix = None
        if fresh and not env._physx_crashed and not args.official_policy:
            remote_name = remote_snapshot_name(args.output, name, args.collection_protocol)
            online_prefix = client.call_endpoint('save_online_prefix', {'branch': remote_name, 'decision': decision + 1})
            prefix_folder = args.output / 'checkpoints' / f'{name}_online_prefix'
            prefix_folder.mkdir(exist_ok=True)
            save_snapshots(online_prefix, prefix_folder)
        write_json(folder / 'trial.json', {'branch': name, 'seed': seed, 'mode': mode, 'origin': origin,
                   **({'lineage': lineage, 'source': 'directed_restore'} if targeted else {}),
                   **({'source': 'controlled', 'controlled_reclosure': {
                       'marker': marker_path.relative_to(args.output).as_posix(),
                       'status': marker['status'], 'valid': marker['valid']}} if pilot else {}),
                   'instance': lineage['instance'] if targeted else fresh_instance,
                   'decisions': executed_decisions, 'steps': steps, 'stop_reason': reason, 'door_contact_steps': contact_history,
                   'max_hinge_excursion_rad': max(hinges), 'failure_candidate': candidate, 'start_step': saved['step'],
                   **({'approach_candidate': approach_candidate} if args.capture_approach else {}),
                   'end_step': base._current_step, 'first_error': first_error, 'first_open_step': first_open_step,
                   'first_task_success_step': first_task_success_step,
                   **({'protocol': 'full_task_eval', 'cycle_limit': args.cycles} if args.full_task_eval else {}),
                   'metric_termination_step': (first_error['step'] if first_error else first_open_step),
                   'observation_end_step': base._current_step,
                   'online_prefix': online_prefix,
                   'post_event_action_chunks_requested': args.post_event_action_chunks,
                   'post_event_action_chunks_completed': post_event_chunks_completed,
                   'elapsed_seconds': time.monotonic() - started})
        write_json(args.output / 'metrics' / 'active_stage.json',
                   {'trial': name, 'stage': 'complete', 'decisions': executed_decisions, 'steps': steps})
        print('TASK40_PAIRED_DONE', name, reason, executed_decisions, steps, len(contact_history), max(hinges), flush=True)

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

    trials = read_json(args.seeds)['trials']
    seeds = [row['policy_seed'] for row in trials]
    assert 1 <= len(seeds) <= 10 and len(seeds) == len(set(seeds))
    try:
        if args.phase == 'anchor':
            for seed in seeds:
                run(f'anchor_{seed}', 'A_normal', seed, failure_name=f'failure_{seed}')
                run(f'anchor_{seed}', 'B_empty', seed)
        elif args.phase == 'restore':
            restore_trials = [(seed, seed) for seed in seeds]
            if targeted:
                trial, = trials
                if trial['policy_seed'] != lineage['source_seed']:
                    raise ValueError('Targeted restore seed record does not match the parent trajectory')
                restore_trials = [(trial['policy_seed'], trial.get('label_seed', trial['policy_seed']))]
            repeated = []
            for seed, label_seed in restore_trials:
                for repeat in range(1, args.restore_repeats + 1):
                    label = f'restore_{label_seed}'
                    if args.restore_repeats > 1:
                        label += f'_repeat{repeat:02d}'
                    run(label, args.memory_mode, seed,
                        failure_name=f'{label}_failure' if args.capture_failed_attempt else None)
                    if targeted and args.restore_repeats == 2:
                        if env._physx_crashed:
                            raise RuntimeError('PhysX crashed during targeted repeat')
                        repeated.append(args.output / 'metrics' / f'{label}_{args.memory_mode}')
            if repeated:
                write_json(args.output / 'metrics' / 'repeat_audit.json', compare_targeted_repeats(*repeated))
            if targeted:
                return
            write_json(args.output / 'metrics' / 'active_stage.json',
                       {'trial': f'{label}_{args.memory_mode}', 'stage': 'complete',
                        'decisions': args.cycles})
        elif args.phase == 'collect':
            for index, seed in enumerate(seeds):
                instance = args.instances[index % len(args.instances)]
                name = f'episode_i{instance}_s{seed}'
                run(name, 'A_normal', seed,
                    failure_name=None if args.official_policy else f'{name}_failure', fresh_instance=instance)
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


def parse_args(argv=None):
    parser = argparse.ArgumentParser()
    for argument in ('output', 'anchor', 'seeds', 'behavior', 'groot'):
        parser.add_argument(f'--{argument}', type=Path, required=True)
    parser.add_argument('--port', type=int, default=5586)
    parser.add_argument('--policy-host', default='127.0.0.1')
    parser.add_argument('--remote-policy', action='store_true')
    parser.add_argument('--trace-memory', action='store_true')
    parser.add_argument('--collection-protocol', action='store_true')
    parser.add_argument('--controlled-reclosure', action='store_true')
    parser.add_argument('--full-task-eval', action='store_true')
    parser.add_argument('--capture-approach', action='store_true')
    parser.add_argument('--cycles', type=int, default=60)
    parser.add_argument('--post-event-action-chunks', type=int, default=2)
    parser.add_argument('--phase', choices=('anchor', 'failure', 'capture', 'restore', 'collect', 'renderer_probe'), default='anchor')
    parser.add_argument('--instances', type=int, nargs='+', default=[4])
    parser.add_argument('--restore-repeats', type=int, choices=range(1, 4), default=1)
    parser.add_argument('--renderer-audit', action='store_true')
    parser.add_argument('--renderer-aa', type=int, choices=(0, 3))
    parser.add_argument('--renderer-probe-actions', type=Path)
    parser.add_argument('--compare-accumulation-reset', action='store_true')
    parser.add_argument('--renderer-gi-history', type=int, choices=(1,))
    parser.add_argument('--renderer-disable-indirect-diffuse', action='store_true')
    parser.add_argument('--restore-checkpoint', type=Path)
    parser.add_argument('--parent-trial', type=Path)
    parser.add_argument('--capture-failed-attempt', action='store_true')
    parser.add_argument('--observe-open-chunks', type=int, default=0)
    parser.add_argument('--stop-at-first-open', action='store_true')
    parser.add_argument('--memory-mode', choices=('A_normal', 'B_empty'), default='A_normal')
    parser.add_argument('--failures', nargs='*', default=[])
    parser.add_argument('--capture-specs', nargs='*', default=[
        'failure_73002_d39:anchor_73002_A_normal:39',
        'failure_73004_d26:anchor_73004_A_normal:26',
        'failure_73003_d39:anchor_73003_A_normal:39',
        'failure_73007_d29:anchor_73007_A_normal:29',
        'failure_73012_d28:anchor_73012_A_normal:28'])
    parser.add_argument('--min-hinge-excursion', type=float, default=0.005)
    parser.add_argument('--rebuild-policy-history', action='store_true')
    parser.add_argument('--official-policy', action='store_true')
    args = parser.parse_args(argv)
    if args.post_event_action_chunks < 0:
        parser.error('--post-event-action-chunks must be non-negative')
    if args.controlled_reclosure and (args.phase != 'collect' or args.memory_mode != 'A_normal'
                                     or not (args.collection_protocol and args.trace_memory and args.remote_policy)):
        parser.error('Controlled reclosure requires fresh A_normal collection and remote memory tracing')
    if args.full_task_eval and (args.phase != 'collect' or args.controlled_reclosure or args.memory_mode != 'A_normal'
                               or not (args.collection_protocol and (args.trace_memory or args.official_policy) and args.remote_policy)):
        parser.error('Full task evaluation requires fresh A_normal collection and remote memory tracing')
    if args.capture_approach and (args.phase != 'collect' or args.full_task_eval or args.official_policy
                                 or not (args.collection_protocol and args.trace_memory and args.remote_policy)):
        parser.error('Approach capture requires fresh V1 collection with Task40 trace')
    if args.renderer_aa is not None and not args.renderer_audit:
        parser.error('--renderer-aa requires --renderer-audit')
    if args.phase == 'renderer_probe' and (not args.renderer_audit or not args.renderer_probe_actions):
        parser.error('renderer_probe requires --renderer-audit and --renderer-probe-actions')
    if args.compare_accumulation_reset and (args.phase != 'renderer_probe' or args.restore_repeats != 2):
        parser.error('Accumulation comparison requires a two-repeat renderer probe')
    if args.renderer_gi_history is not None and args.phase != 'renderer_probe':
        parser.error('GI history limit requires a renderer probe')
    if args.renderer_disable_indirect_diffuse and args.phase != 'renderer_probe':
        parser.error('Disabling indirect diffuse lighting requires a renderer probe')
    if args.restore_checkpoint is not None or args.parent_trial is not None:
        if (not args.restore_checkpoint or not args.parent_trial or args.phase not in ('restore', 'renderer_probe')
                or args.memory_mode != 'A_normal' or args.restore_repeats not in (1, 2)
                or (args.phase == 'restore' and not (
                    args.collection_protocol and (args.trace_memory or args.official_policy) and args.remote_policy))
                or args.post_event_action_chunks != 2):
            parser.error('Targeted restore requires both archived paths, A_normal, at most two repeats, '
                         'remote tracing, and two post-toggle chunks')
    return args


if __name__ == '__main__':
    main(parse_args())
