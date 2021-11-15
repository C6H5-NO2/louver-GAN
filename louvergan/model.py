from typing import List, Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils import spectral_norm

from .config import HyperParam
from .util import ColumnMeta, SlatDim


# || Generator

def gen_blk(in_features: int, out_features: int):
    return nn.Sequential(nn.Linear(in_features, out_features), nn.BatchNorm1d(out_features), nn.ReLU())


class Residual(nn.Module):
    def __init__(self, in_features: int, hidden_features: int):
        super().__init__()
        self._net = gen_blk(in_features, hidden_features)

    def forward(self, i):
        o = self._net(i)
        return torch.cat([o, i], dim=1)

    @property
    def out_features(self):
        return self._net[0].in_features + self._net[0].out_features


class Generator(nn.Module):
    def __init__(self, opt: HyperParam, split: Sequence[SlatDim], meta: Sequence[ColumnMeta]):
        super().__init__()
        cond_dim = sum(s.cond_dim for s in split)
        g_s = [Residual(opt.latent_dim + cond_dim, 256)]
        g_s.append(gen_blk(g_s[-1].out_features, 256))
        self._gen_s = nn.Sequential(*g_s)

        for idx, slat in enumerate(split):
            setattr(self, f'_gen_{idx}', nn.Sequential(
                nn.Linear(256, slat.attr_dim)
            ))
        self._n_slat = len(split)

        self._meta = meta
        self._x_noact = None
        self._moving_avg_w = opt.moving_avg_w
        self._shadow_info = [torch.tensor(0., device=opt.device), torch.tensor(1., device=opt.device),
                             torch.tensor(0., device=opt.device), torch.tensor(1., device=opt.device), ]

    def forward(self, z: torch.Tensor, c: torch.Tensor) -> torch.Tensor:
        s = self._gen_s(torch.cat([z, c], dim=1))
        x = []
        for idx in range(self._n_slat):
            gen = getattr(self, f'_gen_{idx}')
            x.append(gen(s))
        x = torch.cat(x, dim=1)
        self._x_noact = x
        return self.activate(x)

    @property
    def x_no_activation(self) -> torch.Tensor:
        return self._x_noact

    def move_average(self, *args) -> List[torch.Tensor]:
        """
        :param args: [fake mean, fake std, real mean, real std]
        """
        assert len(args) == len(self._shadow_info)
        w = self._moving_avg_w
        for idx in range(len(self._shadow_info)):
            shadow = self._shadow_info[idx].detach()
            self._shadow_info[idx] = (1. - w) * args[idx] + w * shadow
        return self._shadow_info

    def activate(self, x: torch.Tensor) -> torch.Tensor:
        a = []
        bias = 0
        for meta in self._meta:
            if not meta.discrete:
                a.append(torch.tanh(x[:, bias]).reshape(-1, 1))
                bias += 1
            span = meta.nmode
            a.append(F.gumbel_softmax(x[:, bias: bias + span], tau=.2, hard=True))
            bias += span
        return torch.cat(a, dim=1)


# || Discriminator

def dis_blk(in_features: int, out_features: int):
    return nn.Sequential(spectral_norm(nn.Linear(in_features, out_features)), nn.LeakyReLU(.2), nn.Dropout(.5))


class ServerDiscriminator(nn.Module):
    def __init__(self, in_features: int):
        super().__init__()
        self.layers = dis_blk(in_features, 256)
        self.last_layer = spectral_norm(nn.Linear(256, 1))
        self.features = None

    def forward(self, h):
        f = self.layers(h)
        self.features = f
        o = self.last_layer(f)
        return o


class Discriminator(nn.Module):
    def __init__(self, opt: HyperParam, split: Sequence[SlatDim]):
        super().__init__()
        for idx, slat in enumerate(split):
            in_features = (slat.attr_dim + slat.cond_dim) * opt.dis_pac_size
            setattr(self, f'_dis_{idx}', dis_blk(in_features, 256))
        self._dis_s = ServerDiscriminator(len(split) * 256)
        self._attr_split_size = [slat.attr_dim for slat in split]
        self._cond_split_size = [slat.cond_dim for slat in split]
        self._pac_batch = opt.batch_size // opt.dis_pac_size

    def forward(self, x: torch.Tensor, c: torch.Tensor) -> torch.Tensor:
        xi = x.split(self._attr_split_size, dim=1)
        ci = c.split(self._cond_split_size, dim=1)
        h = []
        for idx in range(len(self._attr_split_size)):
            xc = torch.cat([xi[idx], ci[idx]], dim=1).reshape(self._pac_batch, -1)
            dis = getattr(self, f'_dis_{idx}')
            h.append(dis(xc))
        h = torch.cat(h, dim=1)
        return self._dis_s(h)

    @property
    def features(self) -> torch.Tensor:
        return self._dis_s.features
