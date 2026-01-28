from __future__ import annotations
from dataclasses import dataclass
from typing import Tuple
import torch
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.datasets import CelebA

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)

@dataclass
class CelebAConfig:
    root: str = "data/celeba"
    attribute: str = "Smiling"
    image_size: int = 128
    batch_size: int = 64
    num_workers: int = 4
    pin_memory: bool = True


def _build_transforms(image_size: int) -> transforms.Compose:
    return transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
        transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
    ])

class CelebAAttributeDataset(torch.utils.data.Dataset):
    """
    Wraps torchvision CelebA so labels are 0/1 for a single chosen attribute.
    CelebA attr labels are {-1, +1}. We map: -1 -> 0, +1 -> 1.
    """
    def __init__(self, base: CelebA, attr_index: int):
        self.base = base
        self.attr_index = attr_index

    def __len__(self) -> int:
        return len(self.base)

    def __getitem__(self, idx: int):
        x, attrs = self.base[idx]  # attrs shape [40]
        y = attrs[self.attr_index].item()
        y = 1 if y == 1 else 0
        return x, torch.tensor(y, dtype=torch.long)


def get_celeba_dataloaders(cfg: CelebAConfig) -> Tuple[DataLoader, DataLoader, DataLoader]:
    tfm = _build_transforms(cfg.image_size)

    train_base = CelebA(root=cfg.root, split="train", target_type="attr", transform=tfm, download=True)
    val_base   = CelebA(root=cfg.root, split="valid", target_type="attr", transform=tfm, download=False)
    test_base  = CelebA(root=cfg.root, split="test",  target_type="attr", transform=tfm, download=False)

    # Find attribute index
    attr_names = train_base.attr_names
    if cfg.attribute not in attr_names:
        raise ValueError(f"Attribute '{cfg.attribute}' not found. Available: {attr_names}")

    attr_idx = attr_names.index(cfg.attribute)

    train_ds = CelebAAttributeDataset(train_base, attr_idx)
    val_ds   = CelebAAttributeDataset(val_base, attr_idx)
    test_ds  = CelebAAttributeDataset(test_base, attr_idx)

    train_loader = DataLoader(
        train_ds, batch_size=cfg.batch_size, shuffle=True,
        num_workers=cfg.num_workers, pin_memory=cfg.pin_memory
    )
    val_loader = DataLoader(
        val_ds, batch_size=cfg.batch_size, shuffle=False,
        num_workers=cfg.num_workers, pin_memory=cfg.pin_memory
    )
    test_loader = DataLoader(
        test_ds, batch_size=cfg.batch_size, shuffle=False,
        num_workers=cfg.num_workers, pin_memory=cfg.pin_memory
    )
    return train_loader, val_loader, test_loader