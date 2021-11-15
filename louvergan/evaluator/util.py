from typing import Optional, Sequence

import numpy as np
from matplotlib import pyplot as plt


def plot_line(data, legend: Sequence[str], *, figsize=(1200, 720),
              xticks: Optional[np.ndarray] = None, yticks: Optional[np.ndarray] = None,
              title: Optional[str] = None, savepath: Optional[str] = None):
    """
    :param data: shape (#samples, #lines)
    :param figsize: in pixel
    """
    data = np.array(data)
    assert data.shape[1] == len(legend)
    plt.figure(figsize=np.array(figsize) / plt.rcParams['figure.dpi'])
    plt.plot(data, '-+', alpha=.7)
    plt.legend(legend)
    if xticks is not None:
        plt.xticks(xticks)
    if yticks is not None:
        plt.yticks(yticks)
    if title is not None:
        plt.title(title)
    if savepath is not None:
        plt.savefig(savepath)
    plt.show()
