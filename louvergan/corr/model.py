from typing import List, Sequence

import torch
import torch.nn as nn
from torch.nn.utils import spectral_norm


class SplitCorrGenerator(nn.Module):
    def __init__(self, a_dims: Sequence[int], b_dims: Sequence[int], latent_dim: int = 128):
        super().__init__()
        for a_idx, a_dim in enumerate(a_dims):
            setattr(self, f'_corr_gen_a_{a_idx}', nn.Sequential(
                nn.Linear(a_dim + latent_dim, 64),
                nn.ReLU(),
                nn.Linear(64, 16),
                nn.ReLU()
            ))
        for b_idx, b_dim in enumerate(b_dims):
            setattr(self, f'_corr_gen_b_{b_idx}', nn.Sequential(
                nn.Linear(16 * len(a_dims), b_dim)
            ))
        self._n_gen_b = len(b_dims)
        self._latent_dim = latent_dim

    def forward(self, corr_a: Sequence[torch.Tensor]) -> List[torch.Tensor]:
        z = torch.randn(corr_a[0].shape[0], self._latent_dim, device=corr_a[0].device)
        s = []
        for a_idx, a in enumerate(corr_a):
            gen = getattr(self, f'_corr_gen_a_{a_idx}')
            za = torch.cat([z, a], dim=1)
            s.append(gen(za))
        s = torch.cat(s, dim=1)
        corr_b = []
        for b_idx in range(self._n_gen_b):
            gen = getattr(self, f'_corr_gen_b_{b_idx}')
            corr_b.append(gen(s))
        return corr_b


class SplitCorrDiscriminator(nn.Module):
    def __init__(self, a_dims: Sequence[int], b_dims: Sequence[int]):
        super().__init__()
        for a_idx, a_dim in enumerate(a_dims):
            setattr(self, f'_corr_dis_a_{a_idx}', nn.Sequential(
                spectral_norm(nn.Linear(a_dim, 10)),
                nn.LeakyReLU(.2),
                nn.Dropout(.5)
            ))
        for b_idx, b_dim in enumerate(b_dims):
            setattr(self, f'_corr_dis_b_{b_idx}', nn.Sequential(
                spectral_norm(nn.Linear(b_dim, 10)),
                nn.LeakyReLU(.2),
                nn.Dropout(.5)
            ))
        self._corr_dis_s = nn.Sequential(
            spectral_norm(nn.Linear((len(a_dims) + len(b_dims)) * 10, 1))
        )

    def forward(self, corr_a: Sequence[torch.Tensor], corr_b: Sequence[torch.Tensor]):
        h = []
        for a_idx, a in enumerate(corr_a):
            dis = getattr(self, f'_corr_dis_a_{a_idx}')
            h.append(dis(a))
        for b_idx, b in enumerate(corr_b):
            dis = getattr(self, f'_corr_dis_b_{b_idx}')
            h.append(dis(b))
        h = torch.cat(h, dim=1)
        return self._corr_dis_s(h)
