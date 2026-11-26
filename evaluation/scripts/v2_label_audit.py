"""Replay recorded Task40 states with the official wrapper; certify no labels by inference.

This is a workbench GPU audit, not a training runner. Predicate values and robot
proprioception are sampled at the same _process_obs callback as official videos.
All observations, including the extra terminal one, are kept to audit indexing.
"""
import argparse
import hashlib
import json
from pathlib import Path
import sys
import time
import traceback

ROOT = Path(__file__).resolve().parents[2]
BEHAVIOR = Path('/nvme3/projects/GR00T/BEHAVIOR-1K-task-progress')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--deserialize-only', action='store_true')
    parser.add_argument('--all-episodes', action='store_true')
    parser.add_argument('--reuse-dir', type=Path)
    parser.add_argument('--episode', type=int, choices=(400010,400020,403000), default=400010)
    args = parser.parse_args()
    from task40_restore_worker import lease_sim_cache
    cache_lease = lease_sim_cache(args.output.parent/'isaac_appdata', 0, 2)
    sys.path[:0] = [str(BEHAVIOR/'OmniGibson'), str(BEHAVIOR/'bddl')]
    import isaacsim  # noqa: F401: must initialize Isaac before OmniGibson
    import h5py
    import numpy as np
    import pandas as pd
    import torch
    torch.set_num_threads(2)
    import omnigibson as og
    from omnigibson import object_states
    from omnigibson.envs import DataPlaybackWrapper
    from omnigibson.macros import gm
    from omnigibson.learning.utils.eval_utils import PROPRIOCEPTION_INDICES

    gm.ENABLE_TRANSITION_RULES = False
    gm.RENDER_VIEWER_CAMERA = False
    summaries = []
    args.output.parent.mkdir(parents=True, exist_ok=True)
    source = ROOT/'assets/datasets/b1k2025_groot_task40_v1_200'
    slices = ((140,143),(143,146),(146,149),(149,150),(150,151),(151,152),
              (152,155),(155,158),(158,165),(165,172),(172,179),(186,189),
              (189,193),(193,195),(197,204),(204,211),(211,218),(225,228),
              (228,232),(232,234),(236,240))

    class AuditPlayback(DataPlaybackWrapper):
        def _process_obs(self, obs, info):
            robot = self.robots[0]
            scope = self.env.task.object_scope
            microwave = scope['microwave.n.02_1'].wrapped_obj
            bag = scope['popcorn__bag.n.01_1'].wrapped_obj
            # Invoke the same physical predicates used in evaluation. No offsets
            # into serialized floats or missing-object-as-false substitutions.
            values = [microwave.states[object_states.Open].get_value(),
                      bag.states[object_states.Inside].get_value(microwave),
                      microwave.states[object_states.ToggledOn].get_value()]
            proprio = obs['robot_r1::proprio'].detach().cpu().numpy()
            self.audit_proprio.append(np.concatenate([proprio[a:b] for a,b in slices]))
            self.audit_labels.append(values)
            self.audit_joints.append(microwave.get_joint_positions().cpu().numpy())
            if len(self.audit_labels) % 250 == 0:
                np.savez_compressed(self.audit_partial, labels=np.asarray(self.audit_labels),
                    proprio=np.asarray(self.audit_proprio), microwave_joints=np.asarray(self.audit_joints))
                print(json.dumps(dict(episode=self.audit_eid, observations=len(self.audit_labels))), flush=True)
            # Audit collects data separately; don't buffer large official files.
            return {}

    try:
        for eid in (args.episode,):
            raw = Path('/nvme3/datasets/b1k2025_raw_task40_v2_audit/task-0040')/f'episode_{eid:08d}.hdf5'
            parquet = next(source.glob(f'data/**/episode_{eid:08d}.parquet'))
            expected = np.stack(pd.read_parquet(parquet)['observation.state'])
            with h5py.File(raw) as f:
                scene = json.loads(f['data'].attrs['scene_file'])
            # Official replay_obs uses the complete scene and only the task rooms.
            scene_dir = BEHAVIOR/'joylo/sampled_task/make_microwave_popcorn'
            full = next(p for p in sorted(scene_dir.glob('*.json')) if 'partial_rooms' not in p.name)
            import csv
            with (Path(gm.DATA_PATH)/'2025-challenge-task-instances/metadata/B50_task_misc.csv').open() as f:
                rooms = next(row[2].strip().split('\n') for row in csv.reader(f)
                             if 'make_microwave_popcorn' in row[1])
            env = AuditPlayback.create_from_hdf5(input_path=str(raw),
                output_path=str(args.output.parent/f'audit_{eid}.hdf5'),
                robot_obs_modalities=['proprio'], robot_proprio_keys=list(PROPRIOCEPTION_INDICES['R1Pro']),
                robot_sensor_config={'VisionSensor': {'modalities': []}}, external_sensors_config={},
                n_render_iterations=1, full_scene_file=str(full), load_room_instances=rooms,
                include_robot_control=False, include_contacts=False, flush_every_n_traj=1)
            if args.deserialize_only:
                # Deserialize the actual recorded non-kinematic state, using
                # initialized official object schemas. Never scan float offsets.
                decoded = []
                summaries = decoded
                from v2_recording_alignment import matching_demo
                split = json.loads((ROOT/'Output/test_02/experiment_01/evidence/data_split.json').read_text())
                episode_ids = sorted(split['train']+split['validation']) if args.all_episodes else (400010,400020,403000)
                metadata = {r['episode_index']:r for r in map(json.loads,(source/'meta/episodes.jsonl').read_text().splitlines())}
                for decode_eid in episode_ids:
                    decode_raw = raw.with_name(f'episode_{decode_eid:08d}.hdf5')
                    deadline = time.monotonic()+3600
                    while not decode_raw.exists():
                        if time.monotonic() > deadline:
                            raise FileNotFoundError(f'Official data download not complete: {decode_raw}')
                        print(json.dumps(dict(waiting_for_official_recording=decode_eid)),flush=True)
                        time.sleep(30)
                    with h5py.File(decode_raw) as f:
                        decode_scene = json.loads(f['data'].attrs['scene_file'])
                        assert decode_scene['metadata']['task']['inst_to_name'] == scene['metadata']['task']['inst_to_name'], decode_eid
                        converted_path = next(source.glob(f'data/**/episode_{decode_eid:08d}.parquet'))
                        converted = pd.read_parquet(converted_path,columns=['action','timestamp','frame_index'])
                        converted_actions = np.stack(converted['action']).astype(np.float32)
                        demo_name, inventory = matching_demo(f['data'],converted_actions)
                        group = f['data'][demo_name]
                        assert len(converted) == metadata[decode_eid]['length'], decode_eid
                        assert np.array_equal(converted['frame_index'].to_numpy(),np.arange(len(converted))), decode_eid
                        assert np.max(np.abs(converted['timestamp'].to_numpy()-np.arange(len(converted))/30))<1e-4, decode_eid
                        transitions = json.loads(group.attrs['transitions'])
                        for transition in transitions.values():
                            for name in transition['systems']['add']:
                                env.scene.get_system(name, force_init=True)
                        name = decode_scene['metadata']['task']['inst_to_name']['microwave.n.02_1']
                        previous = decode_scene['state']['registry']['object_registry'][name]['non_kin']['ToggledOn']['value']
                        values, present = [], []
                        reuse = args.reuse_dir/f'episode_{decode_eid:08d}_decoded.npz' if args.reuse_dir else None
                        if reuse and reuse.is_file():
                            # The previous failed stage completed these demo_0
                            # records and exact action checks; never reuse a
                            # different attempt or a partial frame array.
                            assert demo_name == 'demo_0', (decode_eid,demo_name)
                            reuse_plan = json.loads((args.reuse_dir/'plan.json').read_text())
                            assert reuse_plan['config']['kind'] == 'v2_train'
                            with np.load(reuse,allow_pickle=False) as old:
                                assert old['labels'].shape == (len(converted),1)
                                values = old['labels'][:,0].astype(bool).tolist()
                                present = old['object_explicitly_recorded'].tolist()
                            print(json.dumps(dict(episode=decode_eid,reused_complete_frames=len(values))),flush=True)
                        else:
                            recorded_states = group['state'][:]
                            recorded_sizes = group['state_size'][:]
                        for frame in (range(len(group['action'])) if not values else []):
                            size = int(recorded_sizes[frame])
                            state, used = og.sim.deserialize(torch.from_numpy(recorded_states[frame,:size]))
                            assert used == size, (decode_eid,frame,used,size)
                            objects = state[0]['registry']['object_registry']
                            if name in objects:
                                previous = objects[name]['non_kin']['ToggledOn']['value']
                            assert isinstance(previous, bool), (decode_eid,frame,type(previous))
                            values.append(bool(previous))
                            present.append(name in objects)
                            if frame % 500 == 0:
                                print(json.dumps(dict(episode=decode_eid, decoded_frames=frame+1)),flush=True)
                        labels = np.asarray(values, dtype=np.float32)
                        known = np.ones(len(labels), dtype=bool)
                        changes = np.flatnonzero(labels[1:] != labels[:-1])+1
                        # Frames close to a toggle transition remain unknown
                        # until the tiny playback propagation is validated.
                        for frame in changes:
                            known[max(0,frame-8):min(len(known),frame+9)] = False
                        archive = args.output.parent/f'episode_{decode_eid:08d}_decoded.npz'
                        np.savez_compressed(archive, labels=labels[:,None], known=known[:,None],
                                            object_explicitly_recorded=np.asarray(present))
                        reference = ROOT/f'Output/test_02/experiment_01/evidence/episode_{decode_eid:08d}_audit.npz'
                        agreement = None
                        if reference.exists():
                            with np.load(reference) as ref:
                                target = ref['labels'][:len(labels),2]
                            agreement = dict(all_frame_disagreements=int(np.count_nonzero(labels!=target)),
                                known_frame_disagreements=int(np.count_nonzero(labels[known]!=target[known])))
                        decoded.append(dict(episode=decode_eid, frame_count=len(labels), label='started',
                            positive_valid=int(labels[known].sum()), negative_valid=int((1-labels[known]).sum()),
                            unknown=int((~known).sum()), transition_frames=changes.tolist(),
                            full_replay_comparison=agreement, labels_certified=False,
                            selected_raw_group=demo_name, raw_attempt_inventory=inventory,
                            actions_exact_match=True, reused_from=str(reuse) if reuse and reuse.exists() else None,
                            explicitly_recorded_frames=int(np.sum(present)),
                            artifact_sha256=hashlib.sha256(archive.read_bytes()).hexdigest(),
                            artifact=str(archive), raw_sha256=hashlib.sha256(decode_raw.read_bytes()).hexdigest()))
                        args.output.write_text(json.dumps(dict(passed=False, episodes=decoded,
                            training_steps=0, labels_certified=False, scope='Recorded ToggledOn deserialization audit'),indent=2)+'\n')
                args.output.write_text(json.dumps(dict(passed=True, episodes=decoded,
                    training_steps=0, labels_certified=False,
                    label_names=['started'], observation_mapping='raw state[t] -> observation[t], final extra state excluded',
                    boundary_unknown_radius_frames=8, split=str(ROOT/'Output/test_02/experiment_01/evidence/data_split.json'),
                    scope='Official deserializer consumed all raw vectors; observation/predicate audit still required'),indent=2)+'\n')
                return
            env.audit_eid = eid
            env.audit_partial = args.output.parent/f'episode_{eid:08d}_partial.npz'
            env.audit_labels, env.audit_proprio, env.audit_joints = [], [], []
            try:
                env.playback_episode(0, record_data=True)
            except Exception:
                np.savez_compressed(env.audit_partial, labels=np.asarray(env.audit_labels),
                    proprio=np.asarray(env.audit_proprio), expected=expected,
                    microwave_joints=np.asarray(env.audit_joints))
                args.output.write_text(json.dumps(dict(passed=False, episodes=summaries,
                    failed_episode=eid, observed_frames=len(env.audit_labels),
                    training_steps=0, labels_certified=False, error=traceback.format_exc()), indent=2)+'\n')
                raise
            actual = np.asarray(env.audit_proprio)
            labels = np.asarray(env.audit_labels, dtype=np.float32)
            assert len(actual) == len(expected)+1, (actual.shape, expected.shape)
            errors = {str(offset): dict(max_abs=float(np.abs(actual[offset:offset+len(expected)]-expected).max()),
                mean_abs=float(np.abs(actual[offset:offset+len(expected)]-expected).mean())) for offset in (0,1)}
            # Official storage drops the final observation. Do not accept t+1
            # simply because its error happens to be smaller under version drift.
            aligned = errors['0']['max_abs'] < 2e-4
            archive = args.output.parent/f'episode_{eid:08d}_audit.npz'
            np.savez_compressed(archive, labels=labels, proprio=actual, expected=expected,
                                microwave_joints=np.asarray(env.audit_joints))
            summaries.append(dict(episode=eid, passed=aligned, frame_count=len(expected),
                observation_count=len(actual), alignment_errors=errors,
                positive_counts=labels[:len(expected)].sum(0).astype(int).tolist(),
                label_names=['door_open','bag_inside','started'],
                artifact=str(archive), raw_sha256=hashlib.sha256(raw.read_bytes()).hexdigest(),
                raw_versions=scene['versions'], local_omnigibson=og.__version__,
                labels_certified=False, reason='Robot alignment audit; image/predicate checks still required'))
            args.output.write_text(json.dumps(dict(passed=False, episodes=summaries,
                training_steps=0, reason='Audit in progress; labels not yet certified'), indent=2)+'\n')
            env.hdf5_file.close()
            env.input_hdf5.close()
            # The official replay entry point uses one process per recording.
            # Clearing a scene containing popcorn particles raises KeyError in
            # this local simulator. Workbench stages isolate the three files.
        args.output.write_text(json.dumps(dict(passed=True,
            episodes=summaries, training_steps=0, labels_certified=False,
            reason='Proprioception alignment only; inspect physical transitions and RGB before training'), indent=2)+'\n')
    except BaseException:
        error = traceback.format_exc()
        print(error, flush=True)
        args.output.write_text(json.dumps(dict(passed=False, episodes=summaries,
            training_steps=0, labels_certified=False, error=error), indent=2)+'\n')
        raise
    finally:
        cache_lease.close()
        og.shutdown()


if __name__ == '__main__':
    main()
