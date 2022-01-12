from typing import Optional, Sequence

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt


# todo: rewrite
def plot_line(data, legend: Sequence[str], *, figsize=(1200, 720), title: Optional[str] = None,
              xticks: Optional[np.ndarray] = None, yticks: Optional[np.ndarray] = None,
              savepath: Optional[str] = None):
    """
    :param data: shape (#samples, #lines)
    :param figsize: in pixel
    """
    data = np.asarray(data)
    assert data.shape[1] == len(legend)
    plt.figure(figsize=np.array(figsize) / plt.rcParams['figure.dpi'])
    plt.plot(data, '-+', alpha=.7)
    plt.legend(legend)
    if title is not None:
        plt.title(title)
    if xticks is not None:
        plt.xticks(xticks)
    if yticks is not None:
        plt.yticks(yticks)
    if savepath is not None:
        plt.savefig(savepath)
    plt.show()


# todo: rewrite
def plot_cdf(real_data: np.ndarray, fake_data: np.ndarray, column_name: str):
    plt.figure()
    plt.xlabel(column_name)
    plt.ylabel('Cumulative Probability')
    plt.xticks(rotation=-45)
    plt.grid()
    plt.margins(0.02)
    plt.plot(np.sort(real_data), np.arange(1, len(real_data) + 1) / len(real_data),
             marker='o', linestyle='none', label='real data', alpha=.7)
    plt.plot(np.sort(fake_data), np.arange(1, len(fake_data) + 1) / len(fake_data),
             marker='o', linestyle='none', label='fake data', alpha=.7)
    plt.legend(loc='upper center', bbox_to_anchor=(0.5, 1.1), ncol=3)
    plt.show()


# todo: rewrite
def plot_mi(mi: np.ndarray, columns: Optional[pd.Index] = None):
    plt.imshow(mi, cmap='Blues', vmin=0., vmax=1.)
    if columns is not None:
        plt.xticks(np.arange(len(columns)), columns, rotation=-90)
        plt.yticks([])  # fixme: yticks have bug
        # plt.yticks(np.arange(len(columns)), columns)
    else:
        plt.xticks([])
        plt.yticks([])
    plt.colorbar(extend='both', ticks=(0, 1))
    plt.show()
