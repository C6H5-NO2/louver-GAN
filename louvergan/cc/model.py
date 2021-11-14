from typing import List, Sequence

import torch
import torch.nn as nn
from torch.nn.utils import spectral_norm


class CondClampAE(nn.Module):
    def __init__(self):
        super().__init__()
        raise NotImplementedError


class SplitCondClampGenerator(nn.Module):
    def __init__(self, a_dims: Sequence[int], b_dims: Sequence[int], latent_dim: int = 128):
        super().__init__()
        self._latent_dim = latent_dim
        for a_idx, a_dim in enumerate(a_dims):
            setattr(self, f'_cc_g_a_{a_idx}', nn.Sequential(
                nn.Linear(a_dim + latent_dim, 64),
                nn.ReLU(True),
                nn.Linear(64, 16),
                nn.ReLU(True)
            ))
        self._n_b = len(b_dims)
        for b_idx, b_dim in enumerate(b_dims):
            setattr(self, f'_cc_g_b_{b_idx}', nn.Sequential(
                nn.Linear(16 * len(a_dims), b_dim)
            ))

    def forward(self, cca: Sequence[torch.Tensor]) -> List[torch.Tensor]:
        z = torch.randn(cca[0].shape[0], self._latent_dim, device=cca[0].device)
        s = []
        for idx, a in enumerate(cca):
            gen = getattr(self, f'_cc_g_a_{idx}')
            za = torch.cat([z, a], dim=1)
            s.append(gen(za))
        s = torch.cat(s, dim=1)
        ccb = []
        for idx in range(self._n_b):
            gen = getattr(self, f'_cc_g_b_{idx}')
            ccb.append(gen(s))
        return ccb


class SplitCondClampDiscriminator(nn.Module):
    def __init__(self, a_dims: Sequence[int], b_dims: Sequence[int]):
        super().__init__()
        for a_idx, a_dim in enumerate(a_dims):
            setattr(self, f'_cc_d_a_{a_idx}', nn.Sequential(
                spectral_norm(nn.Linear(a_dim, 10)),
                nn.LeakyReLU(.2, True),
                nn.Dropout(.5)
            ))
        for b_idx, b_dim in enumerate(b_dims):
            setattr(self, f'_cc_d_b_{b_idx}', nn.Sequential(
                spectral_norm(nn.Linear(b_dim, 10)),
                nn.LeakyReLU(.2, True),
                nn.Dropout(.5)
            ))
        self._cc_d_s = nn.Sequential(
            spectral_norm(nn.Linear((len(a_dims) + len(b_dims)) * 10, 1))
        )

    def forward(self, cca: Sequence[torch.Tensor], ccb: Sequence[torch.Tensor]):
        h = []
        for idx, a in enumerate(cca):
            dis = getattr(self, f'_cc_d_a_{idx}')
            h.append(dis(a))
        for idx, b in enumerate(ccb):
            dis = getattr(self, f'_cc_d_b_{idx}')
            h.append(dis(b))
        h = torch.cat(h, dim=1)
        return self._cc_d_s(h)

# class CondClampGenerator(nn.Module):
#     def __init__(self, a_dim: int, b_dim: int, latent_dim: int = 128):
#         super().__init__()
#         self._latent_dim = latent_dim
#         self._generator = nn.Sequential(
#             nn.Linear(a_dim + latent_dim, 64),
#             nn.ReLU(True),
#             nn.Linear(64, 32),
#             nn.ReLU(True),
#             nn.Linear(32, b_dim)
#         )
# 
#     def forward(self, a):
#         z = torch.randn(a.shape[0], self._latent_dim, device=a.device)
#         za = torch.cat([z, a], dim=1)
#         b = self._generator(za)
#         return b
# 
# 
# class CondClampDiscriminator(nn.Module):
#     def __init__(self, a_dim: int, b_dim: int):
#         super().__init__()
#         self._discriminator = nn.Sequential(
#             spectral_norm(nn.Linear(a_dim + b_dim, 10)),
#             nn.LeakyReLU(.2, True),
#             nn.Dropout(.5),
#             # spectral_norm(nn.Linear(10, 10)),
#             # nn.LeakyReLU(.2, True),
#             # nn.Dropout(.5),
#             spectral_norm(nn.Linear(10, 1))
#         )
# 
#     def forward(self, a, b):
#         ab = torch.cat([a, b], dim=1)
#         return self._discriminator(ab)
