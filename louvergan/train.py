from typing import Sequence

import numpy as np
import torch
import torch.nn.functional as F

from .config import HyperParam
from .corr import CorrSolver
from .model import Discriminator, Generator
from .optimizer import SplitOptimizer
from .sampler import CondDataLoader
from .util import ColumnMeta, path_join


def cond_loss(meta: Sequence[ColumnMeta], x_fake_noact: torch.Tensor, cond: torch.Tensor) -> torch.Tensor:
    bias = 0
    loss = []
    cond_slices = cond.split([m.nmode for m in meta], dim=1)
    for col_meta, col_cond in zip(meta, cond_slices):
        bias += int(not col_meta.discrete)
        span = col_meta.nmode
        target = torch.argmax(col_cond, dim=1)  # i.e. relative bias
        # shape (#batch_size,)
        loss.append(F.cross_entropy(x_fake_noact[:, bias: bias + span], target=target, reduction='none'))
        bias += span
    loss = torch.stack(loss, dim=1)  # shape (#batch_size, #data_cols)

    col_by_cond_bias = np.hstack([np.repeat(i, repeats=m.nmode) for i, m in enumerate(meta)])
    cond_bias = torch.argmax(cond, dim=1).cpu().numpy()
    active_col = col_by_cond_bias[cond_bias]
    mask = torch.zeros_like(loss, requires_grad=False)
    mask[np.arange(mask.shape[0]), active_col] = 1
    return (loss * mask).sum() / loss.shape[0]


def train(opt: HyperParam, meta: Sequence[ColumnMeta], loader: CondDataLoader,
          generator: Generator, discriminator: Discriminator, evaluator,
          corr_solvers: Sequence[CorrSolver]):
    optim_g = SplitOptimizer.from_module(generator, torch.optim.Adam,
                                         lr=opt.gen_lr, betas=(.5, .9), weight_decay=opt.gen_weight_decay)
    optim_d = SplitOptimizer.from_module(discriminator, torch.optim.Adam, lr=opt.dis_lr, betas=(.5, .999))

    for i_epoch in range(opt.n_epoch):
        print(f'epoch {i_epoch:04d}')
        for _ in range(len(loader)):
            x_real, cond = loader.get_batch()
            x_real = x_real.to(opt.device)
            cond = cond.to(opt.device)

            # step D
            z = torch.randn(opt.batch_size, opt.latent_dim, device=opt.device)
            x_fake = generator(z, cond)
            y_fake = discriminator(x_fake, cond)
            y_real = discriminator(x_real, cond)
            loss_d_adv = F.relu(1. - y_real).mean() + F.relu(1. + y_fake).mean()

            critic_fake = torch.mean(y_fake).item()
            critic_real = torch.mean(y_real).item()

            optim_d.zero_grad()
            loss_d = loss_d_adv
            loss_d.backward()
            optim_d.step()

            # step G
            z = torch.randn(opt.batch_size, opt.latent_dim, device=opt.device)
            x_fake = generator(z, cond)
            x_fake_noact = generator.x_no_activation
            y_fake = discriminator(x_fake, cond)
            loss_g_adv = -torch.mean(y_fake)

            feat_fake = discriminator.features
            _ = discriminator(x_real, cond)
            feat_real = discriminator.features
            feat_info = generator.move_average(feat_fake.mean(dim=0), feat_fake.std(dim=0),
                                               feat_real.mean(dim=0), feat_real.std(dim=0))
            loss_g_info = (feat_info[0] - feat_info[2]).norm(p=2) + (feat_info[1] - feat_info[3]).norm(p=2)

            loss_g_cond = cond_loss(meta, x_fake_noact, cond)

            loss_g_corr = torch.stack([solver.corr_loss(x_fake_noact) for solver in corr_solvers], dim=0).sum()

            optim_g.zero_grad()
            loss_g = loss_g_adv + opt.lambda_cond * loss_g_cond \
                     + opt.lambda_corr * loss_g_corr + opt.lambda_info * loss_g_info
            loss_g.backward()
            optim_g.step()

        print(f'loss D: {loss_d.item():.4f}; R/F: {critic_real:.4f}/{critic_fake:.4f}')
        print(f'loss G: {loss_g_adv.item():.4f} (adv); all: {loss_g.item():.4f}, '
              f'cond: {loss_g_cond.item():.4f}, corr: {loss_g_corr.item():.4f}, info:{loss_g_info.item():.4f}')

        # evaluator.collect_loss(loss_d.item(), loss_g_adv.item())

        if i_epoch % opt.save_step == 0 or i_epoch == opt.n_epoch - 1:
            torch.save(generator.state_dict(), path_join(opt.checkpoint_path, f'louver-g-{i_epoch:04d}.pt'))

            # syn_df = synthesize(opt, meta, meta.dataset_rows, generator.state_dict(), sampler, transformer)
            # syn_df.to_csv(path_join(opt.checkpoint_path, f'{DATASET_NAME}-louver-{i_epoch:04d}.csv'), index=False)
            # evaluator.analyse_ml_performance(syn_df)

    # evaluator.plot_loss()
    # evaluator.plot_ml_info()

    return generator, discriminator
