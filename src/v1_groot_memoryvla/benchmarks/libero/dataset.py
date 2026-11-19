"""Native LIBERO labels with ordered, same-episode memory groups."""
import re

import h5py
import numpy as np

from .protocol import (CONTRACT, DATA_ROOT, action_dict, camera_image, manifest_path,
                       read_json, state_dict)
from v1_groot_memoryvla.configs.data.embodiment_configs import MODALITY_CONFIGS
from v1_groot_memoryvla.data.embodiment_tags import EmbodimentTag
from v1_groot_memoryvla.data.types import VLAStepData


def instruction(name):
    return re.sub(r'^.*SCENE\d+_', '', name).replace('_', ' ')


def statistics(values):
    array = np.concatenate(values).astype(np.float64)
    return {key: value.tolist() for key, value in dict(min=array.min(0), max=array.max(0),
        mean=array.mean(0), std=array.std(0), q01=np.quantile(array,.01,axis=0),
        q99=np.quantile(array,.99,axis=0)).items()}


def make_processor(checkpoint, manifests):
    from v1_groot_memoryvla.model.gr00t_n1d6.processing_gr00t_n1d6 import Gr00tN1d6Processor
    processor = Gr00tN1d6Processor.from_pretrained(checkpoint,
        modality_configs={'libero_panda': MODALITY_CONFIGS['libero_panda']},
        use_relative_action=False, language_instruction_mapping={})
    states, actions = [], []
    for manifest in manifests:
        with h5py.File(manifest['file'], 'r') as file:
            for episode in manifest['episodes']:
                demo = file['data/'+episode['name']]
                states.append(state_dict(demo['obs/ee_states'][:], demo['obs/gripper_states'][:]))
                actions.append(action_dict(demo['actions'][:]))
    stats = {modality: {key: statistics([row[key] for row in rows]) for key in rows[0]}
             for modality, rows in [('state', states), ('action', actions)]}
    processor.set_statistics({'libero_panda': stats}, override=True)
    return processor


class LiberoGroups:
    def __init__(self, suite, tasks, processor=None, group_size=16, seed=42, data_root=DATA_ROOT):
        self.processor, self.group_size, self.seed = processor, group_size, seed
        self.manifests = [read_json(manifest_path(suite, task, data_root)) for task in tasks]
        if not all(self.manifests):
            raise ValueError('Prepare the selected LIBERO demonstrations before training')
        self.tasks = sorted(tasks)
        self.by_task = {m['task']: m for m in self.manifests}

    def __getitem__(self, index):
        # Equal task weighting; all groups start with a fresh bank. Sampling is
        # determined by step, so resume reconstructs the exact next group.
        rng = np.random.default_rng(self.seed+index)
        task = self.tasks[index % len(self.tasks)]
        manifest = self.by_task[task]
        episodes = manifest['episodes']
        episode = episodes[int(rng.integers(len(episodes)))]
        length = episode['length']
        candidates = np.arange(0, length, CONTRACT['action_steps'])
        frames = np.sort(rng.choice(candidates, min(self.group_size, len(candidates)), replace=False))
        steps = []
        with h5py.File(manifest['file'], 'r') as file:
            demo = file['data/'+episode['name']]
            for frame in frames:
                end = min(frame+CONTRACT['action_horizon'], length)
                actions = demo['actions'][frame:end].astype(np.float32)
                actions = np.pad(actions, ((0,CONTRACT['action_horizon']-len(actions)),(0,0)), mode='edge')
                value = VLAStepData(
                    images={name: [camera_image(demo['obs/'+key][frame])] for name, key in
                            [('image','agentview_rgb'),('wrist_image','eye_in_hand_rgb')]},
                    states=state_dict(demo['obs/ee_states'][frame:frame+1], demo['obs/gripper_states'][frame:frame+1]),
                    actions=action_dict(actions), text=instruction(manifest['name']),
                    embodiment=EmbodimentTag.LIBERO_PANDA)
                processed = self.processor([dict(type='episode_step',content=value)])
                processed['action_mask'][end-frame:] = 0
                processed['action'][end-frame:] = 0
                steps.append(processed)
        return dict(steps=steps, group_id=index, episode_key=f'{manifest["suite"]}/{task}/{episode["name"]}',
                    episode_start=True, episode_end=True, task=task, episode=episode['name'],
                    timesteps=(frames//CONTRACT['action_steps']).tolist(), frame_indices=frames.tolist())
