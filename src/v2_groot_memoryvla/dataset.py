"""Reuse V1 image/language/label sampling, add aligned past-only Body data."""
import json
from pathlib import Path
import pandas as pd
import torch
from v1_groot_memoryvla.data.dataset.history_dataset import HistoryDataset,HistoryCollator


class V2HistoryDataset(HistoryDataset):
    def __init__(self, *args, mapper, raw_dataset=None, **kwargs):
        super().__init__(*args,**kwargs)
        self.mapper, self.raw_dataset = mapper,Path(raw_dataset) if raw_dataset else None
        self._body_cache_key, self._body_data = None,None

    def __getitem__(self, index):
        group = super().__getitem__(index)
        di,eid,_ = self.groups[index]
        loader = self.loaders[di]
        if self._body_cache_key!=(di,eid):
            filename = loader.data_path_pattern.format(episode_chunk=eid//loader.chunk_size,episode_index=eid)
            frame = pd.read_parquet(loader.dataset_path/filename)
            raw = pd.read_parquet(self.raw_dataset/filename) if self.raw_dataset else None
            modality = json.loads((loader.dataset_path/"meta/modality.json").read_text())
            self._body_data = self.mapper.episode(frame,modality,list(range(len(frame))),raw)
            self._body_cache_key=(di,eid)
        group["body"] = {k:v[group["frame_indices"]] for k,v in self._body_data.items()}
        return group


class V2HistoryCollator(HistoryCollator):
    def __call__(self, groups):
        batch=super().__call__(groups)
        batch["inputs"]["body"]={k:torch.cat([g["body"][k] for g in groups],dim=0)
                                    for k in groups[0]["body"]}
        return batch
