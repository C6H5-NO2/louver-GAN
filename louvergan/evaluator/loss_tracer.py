from collections import OrderedDict
from typing import Dict, Sequence

import numpy as np

from .util import plot_line


class LossTracer:
    def __init__(self, names: Sequence[str]):
        self._loss_trace: Dict[str, list] = OrderedDict.fromkeys(names)
        for k in self._loss_trace:
            self._loss_trace[k] = []

    def collect(self, *args):
        assert len(args) == len(self._loss_trace)
        for k, a in zip(self._loss_trace, args):
            self._loss_trace[k].append(a)

    def plot(self, tick_step: int, **kwargs):
        xticks = np.arange(0, len(next(iter(self._loss_trace.values()))) + tick_step, tick_step)
        # for k, v in self._loss_trace.items():
        #     plot_line(v, legend=[k], xticks=xticks, **kwargs)
        plot_line(np.array(list(self._loss_trace.values())).transpose(),
                  legend=list(self._loss_trace.keys()), xticks=xticks, **kwargs)
