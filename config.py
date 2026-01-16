from __future__ import annotations
from dataclasses import dataclass, field
from typing import List

@dataclass
class DataConfig:
    root: str = "data/celeba"
    attribute: str = "Smiling"
    image_size: int = 128
    batch_size: int = 64
    num_workers: int = 4

@dataclass
class TrainConfig:
    epochs: int = 5
    lr: float = 1e-4
    weight_decay: float = 1e-4
    amp: bool = True
    seed: int = 42

@dataclass
class AttackConfig:
    eps_list: List[float] = field(default_factory=lambda: [
        0.0, 1/255, 2/255, 4/255, 8/255, 16/255
    ])
    pgd_steps: int = 10
    pgd_alpha: float = 2/255

@dataclass
class Paths:
    checkpoints_dir: str = "results/checkpoints"
    metrics_dir: str = "results/metrics"
    figures_dir: str = "results/figures"
    adv_examples_dir: str = "results/adv_examples"