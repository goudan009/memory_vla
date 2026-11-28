"""Explicit cursor recovery around the pinned native shard/processor pipeline.

Only the single-process, zero-worker engineering configuration is supported.
Native background preprocessing shares global RNG with training; synchronous,
seeded shard preparation isolates augmentation RNG and permits exact replay.
"""
from contextlib import contextmanager
import copy
import hashlib
import json
import random


@contextmanager
def preprocessing_rng(seed):
    import numpy as np
    import torch
    python_state, numpy_state = random.getstate(), np.random.get_state()
    with torch.random.fork_rng(devices=[]):
        random.seed(seed)
        np.random.seed(seed)
        torch.random.default_generator.manual_seed(seed)
        try:
            yield
        finally:
            random.setstate(python_state)
            np.random.set_state(numpy_state)


def resumable_type(native_type):
    import numpy as np

    class CursorMixture(native_type):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            if self.world_size != 1:
                raise ValueError('B1K-2026 strict cursor currently requires single GPU')
            self.cursor = dict(epoch=0, shard=0, offset=0, samples=0, order=None)
            self.cursor_rng = np.random.default_rng(self.seed)
            self._cursor_shard = None
            self._cursor_schedule = None

        def _schedule(self):
            if self._cursor_schedule is None:
                self.shard_sampling_schedule = self.generate_shard_sampling_schedule()
                self._cursor_schedule = self.filter_shard_sample_schedule()
            return self._cursor_schedule

        def state_dict(self):
            return dict(schema='b1k2026-cursor-v1', seed=self.seed, rank=self.rank, world_size=self.world_size,
                        cursor=copy.deepcopy(self.cursor), rng=copy.deepcopy(self.cursor_rng.bit_generator.state))

        def load_state_dict(self, state):
            if (state['schema'], state['seed'], state['rank'], state['world_size']) != (
                    'b1k2026-cursor-v1', self.seed, self.rank, self.world_size):
                raise ValueError('sampler identity changed on resume')
            self.cursor = copy.deepcopy(state['cursor'])
            self.epoch = self.cursor['epoch']
            self.cursor_rng.bit_generator.state = state['rng']
            self._cursor_shard = None
            self._cursor_schedule = None

        def __iter__(self):
            from torch.utils.data import get_worker_info
            if get_worker_info() is not None:
                raise ValueError('strict cursor requires dataloader_num_workers=0')
            while True:
                schedule = self._schedule()
                if self.cursor['shard'] >= len(schedule):
                    self.epoch += 1
                    self.cursor.update(epoch=self.epoch, shard=0, offset=0, order=None)
                    self.cursor_rng = np.random.default_rng(self.seed + self.epoch)
                    self._cursor_shard = None
                    self._cursor_schedule = None
                    continue
                dataset_index, shard_index = map(int, schedule[self.cursor['shard']])
                if self._cursor_shard is None:
                    token = json.dumps([self.seed, self.epoch, self.cursor['shard'], dataset_index, shard_index])
                    seed = int(hashlib.sha256(token.encode()).hexdigest()[:8], 16)
                    with preprocessing_rng(seed):
                        self._cursor_shard = self.datasets[dataset_index].get_shard(shard_index)
                    if self.cursor['order'] is None:
                        self.cursor['order'] = self.cursor_rng.permutation(len(self._cursor_shard)).tolist()
                if not self.cursor['order']:
                    raise ValueError('empty native shard')
                item = self._cursor_shard[self.cursor['order'][self.cursor['offset']]]
                self.cursor['offset'] += 1
                self.cursor['samples'] += 1
                if self.cursor['offset'] == len(self.cursor['order']):
                    self.cursor.update(shard=self.cursor['shard'] + 1, offset=0, order=None)
                    self._cursor_shard = None
                yield item

        def close(self):
            self._cursor_shard = None
            if hasattr(super(), 'close'):
                super().close()

    return CursorMixture
