from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Tuple

import torch
from torch.utils.data import DataLoader
from torchvision.datasets import CelebA
from torchvision import transforms


# Normalize to stable range for attacks/defenses
CELEBA_MEAN = [0.5, 0.5, 0.5]
CELEBA_STD = [0.5, 0.5, 0.5]


@dataclass
class CelebAConfig:
    data_root: str = "data"          # repo_root/data
    subdir: str = "celeba"           # repo_root/data/celeba
    attr: str = "Smiling"
    image_size: int = 128
    batch_size: int = 128
    num_workers: int = 2


def _celeba_root(cfg: CelebAConfig) -> str:
    # torchvision CelebA expects: root/<base_folder>/...
    # We'll pass root=data and keep dataset under data/celeba/...
    # Easiest is: root="data", and ensure files live under data/celeba/
    return str(Path(cfg.data_root))


def _train_tfms(image_size: int):
    return transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.ToTensor(),
        transforms.Normalize(CELEBA_MEAN, CELEBA_STD),
    ])


def _eval_tfms(image_size: int):
    return transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
        transforms.Normalize(CELEBA_MEAN, CELEBA_STD),
    ])


class CelebABinary(torch.utils.data.Dataset):
    """
    Wrapper around torchvision CelebA for ONE attribute.
    Returns (image, y) where y in {0,1}.
    """

    def __init__(self, split: str, attr: str, transform, root: str = "data"):
        self.base = CelebA(
            root=root,
            split=split,              # train | valid | test
            target_type="attr",
            download=False,           # manual placement (avoids Google Drive quota issues)
            transform=transform,
        )
        self.attr = attr
        try:
            self.idx = self.base.attr_names.index(attr)
        except ValueError as e:
            raise ValueError(
                f"Unknown attribute '{attr}'. Example attrs: Smiling, Eyeglasses, Male, Young"
            ) from e

    def __len__(self) -> int:
        return len(self.base)

    def __getitem__(self, i: int):
        x, attrs = self.base[i]
        y = (attrs[self.idx] == 1).long()  # -1/+1 -> 0/1
        return x, y


def make_loaders(cfg: CelebAConfig) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """
    Professor requirement: keep train/val/test separated.
    - train: training
    - valid: tuning
    - test: final clean + attack + defense evaluation
    """
    root = _celeba_root(cfg)

    train_ds = CelebABinary("train", cfg.attr, _train_tfms(cfg.image_size), root=root)
    val_ds   = CelebABinary("valid", cfg.attr, _eval_tfms(cfg.image_size), root=root)
    test_ds  = CelebABinary("test",  cfg.attr, _eval_tfms(cfg.image_size), root=root)

    train_loader = DataLoader(train_ds, batch_size=cfg.batch_size, shuffle=True,
                              num_workers=cfg.num_workers, pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size=cfg.batch_size, shuffle=False,
                            num_workers=cfg.num_workers, pin_memory=True)
    test_loader = DataLoader(test_ds, batch_size=cfg.batch_size, shuffle=False,
                             num_workers=cfg.num_workers, pin_memory=True)
    return train_loader, val_loader, test_loader
