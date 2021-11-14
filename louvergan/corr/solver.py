from typing import List, Optional, Sequence, Type

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

from .config import CorrSolverConfig
from .model import SplitCorrDiscriminator, SplitCorrGenerator
from .util import Corr, CorrSolverType
from ..optimizer import SplitOptimizer
from ..sampler import CondDataLoader
from ..util import BiasSpan, get_slices, path_join


class CorrSolver:
    def __init__(self, corr: Corr):
        self._corr = corr
        self._model: Optional[nn.Module] = None

    def fit(self, loader: DataLoader, conf: CorrSolverConfig, verbose: bool = True):
        return self

    def load(self, state_dict):
        self._model.load_state_dict(state_dict)

    @torch.no_grad()
    def predict(self, batch: torch.Tensor) -> List[torch.Tensor]:
        raise NotImplementedError

    def corr_loss(self, batch: torch.Tensor) -> torch.Tensor:
        return torch.tensor(0., device=batch.device, requires_grad=True)

    @classmethod
    def from_type(cls, stype: CorrSolverType, corr: Corr):
        scls = get_corr_solver_from_type(stype)
        return scls(corr)


class CorrSolverCGAN(CorrSolver):
    def fit(self, loader: CondDataLoader, conf: CorrSolverConfig, verbose: bool = True):
        loss_trace = []

        a_dims = [sum(span for _, span in bs_slat) for bs_slat in self._corr.bias_span_a_per_slat]
        b_dims = [sum(span for _, span in bs_slat) for bs_slat in self._corr.bias_span_b_per_slat]

        generator = SplitCorrGenerator(a_dims, b_dims).to(conf.device).train()
        discriminator = SplitCorrDiscriminator(a_dims, b_dims).to(conf.device).train()

        optim_g = SplitOptimizer.from_module(generator, torch.optim.Adam, lr=conf.lr, betas=(.5, .9))
        optim_d = SplitOptimizer.from_module(discriminator, torch.optim.Adam, lr=conf.lr, betas=(.5, .999))

        for i_epoch in range(conf.n_epoch):
            if verbose:
                print(f'    corr: epoch {i_epoch:04d};', end='')
            for data, _ in loader:
                data = data.to(conf.device)
                corr_a = [get_slices(data, bs_slat) for bs_slat in self._corr.bias_span_a_per_slat]

                # step D
                corr_b_real = [get_slices(data, bs_slat) for bs_slat in self._corr.bias_span_b_per_slat]
                corr_b_real = [
                    (ori + torch.randn_like(ori) * 1e-4).clamp(0., 1.) for ori in corr_b_real
                ]  # apply noise
                y_real = discriminator(corr_a, corr_b_real)

                corr_b_fake: List[torch.Tensor] = generator(corr_a)
                corr_b_fake = self.activate(corr_b_fake, self._corr.bias_span_b_per_slat)
                y_fake = discriminator(corr_a, corr_b_fake)

                loss_d = F.relu(1. - y_real).mean() + F.relu(1. + y_fake).mean()
                optim_d.zero_grad()
                loss_d.backward()
                optim_d.step()

                # step G
                corr_b_fake = generator(corr_a)
                corr_b_fake = self.activate(corr_b_fake, self._corr.bias_span_b_per_slat)
                y_fake = discriminator(corr_a, corr_b_fake)

                loss_g = -torch.mean(y_fake)
                optim_g.zero_grad()
                loss_g.backward()
                optim_g.step()

            loss_trace.append((loss_d.item(), loss_g.item()))
            if verbose:
                print(f' loss D: {loss_d.item():.4f}, loss G: {loss_g.item():.4f}')
            if i_epoch % conf.save_step == 0 or i_epoch == conf.n_epoch - 1:
                name = f'corr={"~".join(self._corr.a_names)}={"~".join(self._corr.b_names)}=cgan-g-{i_epoch:04d}.pt'
                path = path_join(conf.checkpoint_path, name)
                torch.save(generator.state_dict(), path)
        # todo: plot
        self._model = generator.eval()
        return self

    @torch.no_grad()
    def predict(self, batch: torch.Tensor) -> List[torch.Tensor]:
        corr_a = [get_slices(batch, bs_slat) for bs_slat in self._corr.bias_span_a_per_slat]
        corr_b_pred = self._model(corr_a)
        corr_b_pred = self.activate(corr_b_pred, self._corr.bias_span_b_per_slat, hard=True)
        return corr_b_pred

    def corr_loss(self, batch: torch.Tensor) -> torch.Tensor:
        corr_b_pred = self.predict(batch)
        corr_b_fake = [get_slices(batch, bs_slat) for bs_slat in self._corr.bias_span_b_per_slat]
        loss = []
        for bs_slat, pred, fake in zip(self._corr.bias_span_b_per_slat, corr_b_pred, corr_b_fake):
            bias = 0
            for _, span in bs_slat:
                target = torch.argmax(pred[:, bias: bias + span], dim=1)
                loss.append(F.cross_entropy(fake[:, bias: bias + span], target=target))
                bias += span
        return torch.cat(loss).mean()

    @staticmethod
    def activate(logits: Sequence[torch.Tensor], bias_span_per_slat: Sequence[Sequence[BiasSpan]],
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


def get_corr_solver_from_type(stype: CorrSolverType) -> Type[CorrSolver]:
    if stype == CorrSolverType.NONE:
        return CorrSolver
    elif stype == CorrSolverType.AE:
        raise NotImplementedError
    elif stype == CorrSolverType.CGAN:
        return CorrSolverCGAN
    else:
        raise ValueError
