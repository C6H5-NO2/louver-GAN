from typing import Sequence

import numpy as np
import torch

from .util import ColumnMeta


# from torch.utils.data import DataLoader, Dataset, Sampler


class CondSampler:
    def __init__(self, data: np.ndarray, meta: Sequence[ColumnMeta]):
        assert data.ndim == 2
        self._n_cond_col = len(meta)
        max_nmode = max((m.nmode for m in meta), default=0)
        self._cum_mode_freq = np.zeros((self._n_cond_col, max_nmode))
        self._cum_mode_log_freq = np.zeros_like(self._cum_mode_freq)
        bias = 0
        for col_idx, col_meta in enumerate(meta):
            bias += int(not col_meta.discrete)
            span = col_meta.nmode
            freq = np.sum(data[:, bias: bias + span], axis=0)
            log_freq = np.log(1 + freq)
            self._cum_mode_freq[col_idx, :span] = freq / np.sum(freq)
            self._cum_mode_log_freq[col_idx, :span] = log_freq / np.sum(log_freq)
            bias += span
        self._cum_mode_freq = self._cum_mode_freq.cumsum(axis=1)
        self._cum_mode_log_freq = self._cum_mode_log_freq.cumsum(axis=1)

    def sample_freq(self, size: int):
        return self._sample(size, self._cum_mode_freq)

    def sample_log_freq(self, size: int):
        return self._sample(size, self._cum_mode_log_freq)

    def _sample(self, size: int, cum_freq: np.ndarray):
        sel_col: np.ndarray = np.random.randint(low=0, high=self._n_cond_col, size=size)
        mode_prob: np.ndarray = np.random.random(size=size).reshape(-1, 1)
        sel_mode: np.ndarray = (cum_freq[sel_col] > mode_prob).argmax(axis=1)  # shape (#size,)
        return sel_col, sel_mode


class CondDataLoader:
    def __init__(self, data: np.ndarray, meta: Sequence[ColumnMeta], batch_size: int):
        assert data.ndim == 2
        self._sampler = CondSampler(data, meta)
        self._data = data
        self._batch_size = batch_size
        self._training = True

        self._cond_dim = sum(m.nmode for m in meta)
        self._cond_bias_by_col = np.hstack((0, np.cumsum([m.nmode for m in meta])[:-1]))
        self._ridx_by_col_mode = []
        bias = 0
        for col_meta in meta:
            bias += int(not col_meta.discrete)
            span = col_meta.nmode
            assert span > 1  # fixme
            mode_per_row = np.nonzero(data[:, bias: bias + span])[1]
            sort_idx = np.argsort(mode_per_row)
            counts = np.bincount(mode_per_row, minlength=col_meta.nmode)
            ridx_by_mode = np.split(sort_idx, counts.cumsum()[:-1])
            self._ridx_by_col_mode.append(ridx_by_mode)
            bias += span

    def __iter__(self):
        for _ in range(len(self)):
            x, c = self.get_batch()
            yield x, c

    def __len__(self):
        return self._data.shape[0] // self._batch_size

    def get_batch(self):
        if self._training:
            sel_col, sel_mode = self._sampler.sample_log_freq(self._batch_size)
        else:
            sel_col, sel_mode = self._sampler.sample_freq(self._batch_size)

        ridx = []
        for col, mode in zip(sel_col, sel_mode):
            ridx.append(np.random.choice(self._ridx_by_col_mode[col][mode]))
        x = torch.tensor(self._data[ridx], dtype=torch.float32)

        c = torch.zeros(self._batch_size, self._cond_dim)
        c[np.arange(self._batch_size), self._cond_bias_by_col[sel_col] + sel_mode] = 1

        return x, c

    @property
    def dataset_size(self) -> int:
        return self._data.shape[0]

    def train(self, mode: bool = True):
        self._training = mode
        return self

    def eval(self):
        return self.train(False)
