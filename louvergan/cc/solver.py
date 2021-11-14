from typing import List, Optional, Sequence, Type

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

from .config import CondClampSolverConfig
from .model import SplitCondClampDiscriminator, SplitCondClampGenerator
from .util import CondClamp, CondClampSolverType
from ..split import SplitOptimizer
from ..util import BiasSpan, get_slices


class CondClampSolver:
    def __init__(self, cc: CondClamp):
        self._cc = cc
        self._model: Optional[nn.Module] = None

    def fit(self, dl: DataLoader, conf: CondClampSolverConfig, verbose: bool = True):
        return self

    @torch.no_grad()
    def predict(self, batch: torch.Tensor) -> List[torch.Tensor]:
        raise NotImplementedError

    def cc_loss(self, batch: torch.Tensor) -> torch.Tensor:
        return torch.tensor(0., device=batch.device, requires_grad=True)

    @classmethod
    def from_type(cls, stype: CondClampSolverType, cc: CondClamp):
        scls = get_cc_solver_from_type(stype)
        return scls(cc)


class CondClampSolverCGAN(CondClampSolver):
    def fit(self, dl: DataLoader, conf: CondClampSolverConfig, verbose: bool = True):
        loss_trace = []

        a_dims = [sum(span for _, span in bs_slat) for bs_slat in self._cc.bias_span_a_per_slat]
        b_dims = [sum(span for _, span in bs_slat) for bs_slat in self._cc.bias_span_b_per_slat]

        generator = SplitCondClampGenerator(a_dims, b_dims).to(conf.device).train()
        discriminator = SplitCondClampDiscriminator(a_dims, b_dims).to(conf.device).train()

        optim_g = SplitOptimizer.from_module(generator, torch.optim.Adam, lr=conf.lr, betas=(.5, .999))
        optim_d = SplitOptimizer.from_module(discriminator, torch.optim.Adam, lr=conf.lr, betas=(.5, .999))

        for i_epoch in range(conf.n_epoch):
            if verbose:
                print(f'    cc: epoch {i_epoch:04d};', end='')
            for data in dl:
                # step D
                data = torch.cat(data, dim=1).to(conf.device)
                cc_a = [get_slices(data, bs_slat) for bs_slat in self._cc.bias_span_a_per_slat]
                cc_b_true = [get_slices(data, bs_slat) for bs_slat in self._cc.bias_span_b_per_slat]
                cc_b_true = [
                    (ori + torch.randn_like(ori) * 1e-4).clamp(0., 1.) for ori in cc_b_true
                ]  # apply noise
                y_real = discriminator(cc_a, cc_b_true)

                cc_b_pred: List[torch.Tensor] = generator(cc_a)
                cc_b_pred = self.active_modes(cc_b_pred, self._cc.bias_span_b_per_slat)
                y_fake = discriminator(cc_a, cc_b_pred)

                loss_d = F.relu(1. - y_real).mean() + F.relu(1. + y_fake).mean()
                optim_d.zero_grad()
                loss_d.backward()
                optim_d.step()

                # step G
                cc_b_pred = generator(cc_a)
                cc_b_pred = self.active_modes(cc_b_pred, self._cc.bias_span_b_per_slat)
                y_fake = discriminator(cc_a, cc_b_pred)

                loss_g = -torch.mean(y_fake)
                optim_g.zero_grad()
                loss_g.backward()
                optim_g.step()

            loss_trace.append((loss_d.item(), loss_g.item()))
            if verbose:
                print(f' loss D: {loss_d.item():.4f}, loss G: {loss_g.item():.4f}')
            # if i_epoch % conf.save_step == 0 or i_epoch == conf.n_epoch-1:
            #     torch.save(generator.state_dict(), path_join(conf.checkpoint_path, f'/cc/g_{i_epoch:04d}.pt'))

        # todo: plot

        self._model = generator.eval()
        return self

    @torch.no_grad()
    def predict(self, batch: torch.Tensor) -> List[torch.Tensor]:
        cc_a = [get_slices(batch, bs_slat) for bs_slat in self._cc.bias_span_a_per_slat]
        cc_b_pred = self._model(cc_a)
        cc_b_pred = self.active_modes(cc_b_pred, self._cc.bias_span_b_per_slat, hard=True)
        return cc_b_pred

    def cc_loss(self, batch: torch.Tensor) -> torch.Tensor:
        cc_b_pred = self.predict(batch)
        cc_b_fake = [get_slices(batch, bs_slat) for bs_slat in self._cc.bias_span_b_per_slat]
        loss = []
        for bs_slat, pred, fake in zip(self._cc.bias_span_b_per_slat, cc_b_pred, cc_b_fake):
            bias = 0
            for _, span in bs_slat:
                target = torch.argmax(pred[:, bias: bias + span], dim=1)
                loss.append(F.cross_entropy(fake[:, bias: bias + span], target=target))
                bias += span
        return torch.cat(loss).mean()

    @staticmethod
    def active_modes(logits: Sequence[torch.Tensor], bias_span_per_slat: Sequence[Sequence[BiasSpan]],
                     tau: float = .2, hard: bool = False) -> List[torch.Tensor]:
        activated = []
        for logit, bs_slat in zip(logits, bias_span_per_slat):
            slat = []
            bias = 0
            for _, span in bs_slat:
                slat.append(F.gumbel_softmax(logit[:, bias: bias + span], tau=tau, hard=hard))
                bias += span
            activated.append(torch.cat(slat, dim=1))
        return activated


def get_cc_solver_from_type(stype: CondClampSolverType) -> Type[CondClampSolver]:
    if stype == CondClampSolverType.NONE:
        return CondClampSolver
    elif stype == CondClampSolverType.AE:
        raise NotImplementedError
    elif stype == CondClampSolverType.CGAN:
        return CondClampSolverCGAN
    else:
        raise ValueError
