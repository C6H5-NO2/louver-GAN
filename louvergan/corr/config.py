from dataclasses import dataclass


@dataclass
class CorrSolverConfig:
    checkpoint_path: str
    device: str

    n_epoch: int = 1
    save_step: int = 100

    latent_dim: int = 128
    batch_size: int = 256
    lr: float = 1e-4
