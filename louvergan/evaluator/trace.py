from typing import Dict, Iterable

import numpy as np

from .plot import plot_line


class Trace:
    def __init__(self, names: Iterable[str]):
        self._trace: Dict[str, list] = {n: [] for n in names}

    def __len__(self):
        return len(next(iter(self._trace.values())))

    def collect(self, *args):
        assert len(args) == len(self._trace)
        for k, a in zip(self._trace, args):
            self._trace[k].append(a)

    def plot(self, plot_mean: bool = False, **kwargs):
        if not len(self):
            return
        data = np.array(list(self._trace.values())).transpose()
        legend = list(self._trace.keys())
        if plot_mean and data.shape[1] > 1:
            data = np.hstack([data, data.mean(axis=1, keepdims=True)])
            legend.append('mean')
        plot_line(data, legend=legend, **kwargs)
