from typing import Iterable, Optional, Tuple, Type

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

from .config import CorrSolverConfig
from .model import SplitCorrAutoEncoder, SplitCorrDiscriminator, SplitCorrGenerator
from .util import BiasSpan, Corr, CorrSolverType, get_slices
from ..evaluator import Trace
from ..optimizer import SplitOptimizer
from ..polyfill import zip_strict
from ..sampler import CondDataLoader
from ..util import path_join


class CorrSolver:
    def __init__(self, corr: Corr):
        self._corr = corr
        self._model: Optional[nn.Module] = None

    @classmethod
    def from_type(cls, stype: CorrSolverType, corr: Corr):
        scls = get_corr_solver_from_type(stype)
        return scls(corr)

    def fit(self, loader: DataLoader, conf: CorrSolverConfig, verbose: bool = False):
        raise NotImplementedError

    @torch.no_grad()
    def predict(self, batch: torch.Tensor) -> Tuple[torch.Tensor, ...]:
        raise NotImplementedError

    def corr_loss(self, batch: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError

    def load(self, state_dict: dict):
        raise NotImplementedError

    @staticmethod
    def activate(logits: Iterable[torch.Tensor], bias_span_per_slat: Iterable[Iterable[BiasSpan]],
                 tau: float = .2, hard: bool = False) -> Tuple[torch.Tensor, ...]:
        activated = []
        for logit, bs_slat in zip_strict(logits, bias_span_per_slat):
            slat = []
            bias = 0
            for _, span in bs_slat:
                slat.append(F.gumbel_softmax(logit[:, bias: bias + span], tau=tau, hard=hard))
                bias += span
            activated.append(torch.cat(slat, dim=1))
        return tuple(activated)


class CorrSolverNone(CorrSolver):
    def fit(self, loader: DataLoader, conf: CorrSolverConfig, verbose: bool = False):
        return self

    @torch.no_grad()
    def predict(self, batch: torch.Tensor) -> Tuple[torch.Tensor, ...]:
        pass

    def corr_loss(self, batch: torch.Tensor) -> torch.Tensor:
        return torch.tensor(0., device=batch.device, requires_grad=True)

    def load(self, state_dict: dict):
        raise NotImplementedError


class CorrSolverAE(CorrSolver):
    def fit(self, loader: DataLoader, conf: CorrSolverConfig, verbose: bool = False):
        loss_trace = Trace(['loss ae'])

        a_dims = [sum(span for _, span in bs_slat) for bs_slat in self._corr.bias_span_a_per_slat]
        b_dims = [sum(span for _, span in bs_slat) for bs_slat in self._corr.bias_span_b_per_slat]

        autoencoder = SplitCorrAutoEncoder(a_dims, b_dims).to(conf.device).train()

        optim = SplitOptimizer.from_module(autoencoder, torch.optim.Adam, lr=conf.lr, betas=(.5, .999))

        for i_epoch in range(conf.n_epoch):
            if verbose:
                print(f'    corr: epoch {i_epoch:04d};', end='')
            for x_real, _ in loader:
                x_real = x_real.to(conf.device)
                corr_a = [get_slices(x_real, bs_slat) for bs_slat in self._corr.bias_span_a_per_slat]
                corr_b_true = [get_slices(x_real, bs_slat) for bs_slat in self._corr.bias_span_b_per_slat]
                corr_b_pred = autoencoder(corr_a)
                corr_b_pred = self.activate(corr_b_pred, self._corr.bias_span_b_per_slat)
                corr_b_true = torch.cat(corr_b_true, dim=1)
                corr_b_pred = torch.cat(corr_b_pred, dim=1)
                loss = (corr_b_true - corr_b_pred).norm(p=1, dim=1).mean()
                optim.zero_grad()
                loss.backward()
                optim.step()
            if verbose:
                print(f' loss AE: {loss.item():.4f}')
            loss_trace.collect(loss.item())
            if i_epoch % conf.save_step == 0 or i_epoch == conf.n_epoch - 1:
                name = f'corr={"~".join(self._corr.a_names)}={"~".join(self._corr.b_names)}=ae-{i_epoch:04d}.pt'
                path = path_join(conf.checkpoint_path, name)
                torch.save(autoencoder.state_dict(), path)

        loss_trace.plot(figsize=(900, 540), title=f'corr pretrain: {self._corr.a_names} => {self._corr.b_names}',
                        xticks=np.arange(0, conf.n_epoch + conf.save_step, conf.save_step))
        self._model = autoencoder.eval()
        return self

    @torch.no_grad()
    def predict(self, batch: torch.Tensor) -> Tuple[torch.Tensor, ...]:
        corr_a = [get_slices(batch, bs_slat) for bs_slat in self._corr.bias_span_a_per_slat]
        corr_b_pred = self._model(corr_a)
        corr_b_pred = self.activate(corr_b_pred, self._corr.bias_span_b_per_slat, hard=True)
        return corr_b_pred

    def corr_loss(self, batch: torch.Tensor) -> torch.Tensor:
        corr_b_pred = self.predict(batch)
        corr_b_fake = [get_slices(batch, bs_slat) for bs_slat in self._corr.bias_span_b_per_slat]
        loss = []
        for bs_slat, pred, fake in zip_strict(self._corr.bias_span_b_per_slat, corr_b_pred, corr_b_fake):
            bias = 0
            for _, span in bs_slat:
                loss.append(
                    (pred[:, bias: bias + span] - fake[:, bias: bias + span]).norm(p=1, dim=1, keepdim=True)
                )
                bias += span
        return torch.stack(loss, dim=0).mean()

    def load(self, state_dict: dict, conf: Optional[CorrSolverConfig] = None):
        if self._model is None:
            a_dims = [sum(span for _, span in bs_slat) for bs_slat in self._corr.bias_span_a_per_slat]
            b_dims = [sum(span for _, span in bs_slat) for bs_slat in self._corr.bias_span_b_per_slat]
            self._model = SplitCorrAutoEncoder(a_dims, b_dims).to(conf.device).eval()
        self._model.load_state_dict(state_dict)


class CorrSolverCGAN(CorrSolver):
    def fit(self, loader: CondDataLoader, conf: CorrSolverConfig, verbose: bool = False):
        loss_trace = Trace(['loss corr d', 'loss corr g'])

        a_dims = [sum(span for _, span in bs_slat) for bs_slat in self._corr.bias_span_a_per_slat]
        b_dims = [sum(span for _, span in bs_slat) for bs_slat in self._corr.bias_span_b_per_slat]

        generator = SplitCorrGenerator(a_dims, b_dims, conf.latent_dim).to(conf.device).train()
        discriminator = SplitCorrDiscriminator(a_dims, b_dims).to(conf.device).train()

        optim_g = SplitOptimizer.from_module(generator, torch.optim.Adam, lr=conf.lr, betas=(.5, .9))
        optim_d = SplitOptimizer.from_module(discriminator, torch.optim.Adam, lr=conf.lr, betas=(.5, .999))

        for i_epoch in range(conf.n_epoch):
            if verbose:
                print(f'    corr: epoch {i_epoch:04d};', end='')
            for x_real, _ in loader:
                x_real = x_real.to(conf.device)
                corr_a = [get_slices(x_real, bs_slat) for bs_slat in self._corr.bias_span_a_per_slat]

                # step D
                corr_b_real = [get_slices(x_real, bs_slat) for bs_slat in self._corr.bias_span_b_per_slat]
                corr_b_real = [
                    # apply noise
                    (ori + torch.randn_like(ori) * 2e-5).clamp(0., 1.) for ori in corr_b_real
                ]
                y_real = discriminator(corr_a, corr_b_real)

                corr_b_fake: Tuple[torch.Tensor, ...] = generator(corr_a)
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

            if verbose:
                print(f' loss D: {loss_d.item():.4f}; loss G: {loss_g.item():.4f}')
            loss_trace.collect(loss_d.item(), loss_g.item())
            if i_epoch % conf.save_step == 0 or i_epoch == conf.n_epoch - 1:
                name = f'corr={"~".join(self._corr.a_names)}={"~".join(self._corr.b_names)}=cgan-g-{i_epoch:04d}.pt'
                path = path_join(conf.checkpoint_path, name)
                torch.save(generator.state_dict(), path)

        loss_trace.plot(figsize=(900, 540), title=f'corr pretrain: {self._corr.a_names} => {self._corr.b_names}',
                        xticks=np.arange(0, conf.n_epoch + conf.save_step, conf.save_step))
        self._model = generator.eval()
        return self

    @torch.no_grad()
    def predict(self, batch: torch.Tensor) -> Tuple[torch.Tensor, ...]:
        corr_a = [get_slices(batch, bs_slat) for bs_slat in self._corr.bias_span_a_per_slat]
        corr_b_pred = self._model(corr_a)
        corr_b_pred = self.activate(corr_b_pred, self._corr.bias_span_b_per_slat, hard=True)
        return corr_b_pred

    def corr_loss(self, batch: torch.Tensor) -> torch.Tensor:
        corr_b_pred = self.predict(batch)
        corr_b_fake = [get_slices(batch, bs_slat) for bs_slat in self._corr.bias_span_b_per_slat]
        loss = []
        for bs_slat, pred, fake in zip_strict(self._corr.bias_span_b_per_slat, corr_b_pred, corr_b_fake):
            bias = 0
            for _, span in bs_slat:
                target = torch.argmax(pred[:, bias: bias + span], dim=1)
                loss.append(F.cross_entropy(fake[:, bias: bias + span], target=target))
                bias += span
        return torch.stack(loss, dim=0).mean()

    def load(self, state_dict: dict, conf: Optional[CorrSolverConfig] = None):
        if self._model is None:
            a_dims = [sum(span for _, span in bs_slat) for bs_slat in self._corr.bias_span_a_per_slat]
            b_dims = [sum(span for _, span in bs_slat) for bs_slat in self._corr.bias_span_b_per_slat]
            self._model = SplitCorrGenerator(a_dims, b_dims, conf.latent_dim).to(conf.device).eval()
        self._model.load_state_dict(state_dict)


def get_corr_solver_from_type(stype: CorrSolverType) -> Type[CorrSolver]:
    if stype == CorrSolverType.NONE:
        return CorrSolverNone
    elif stype == CorrSolverType.AE:
        return CorrSolverAE
    elif stype == CorrSolverType.CGAN:
        return CorrSolverCGAN
    else:
        raise ValueError
