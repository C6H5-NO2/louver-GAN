from typing import List

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

from .config import HyperParam
from .util import ColumnMeta, SlatDim


class SplitTabularDataset(Dataset):
    def __init__(self, data: np.ndarray, split: List[SlatDim]):
        self._data = data
        self._split = split

    def __getitem__(self, index) -> List[torch.Tensor]:
        data = torch.tensor(self._data[index], dtype=torch.float32)
        bias = 0
        slices = []
        for slat in self._split:
            slices.append(data[..., bias: bias + slat.attr_dim])
            bias += slat.attr_dim
        return slices  # each with shape (#samples, #attr_dim)

    def __len__(self):
        return self._data.shape[0]


def get_sampler(opt: HyperParam, data: np.ndarray, meta: List[ColumnMeta], split: List[SlatDim], cont_as_cond=True):
    ds = SplitTabularDataset(data, split)
    dl = DataLoader(ds, batch_size=opt.batch_size, shuffle=True, num_workers=2, drop_last=True)
    return dl
