from typing import Tuple

from torch.utils.data import DataLoader

from src.data.celeba import CelebABinary
from src.data.transforms import eval_tfms, train_tfms


def make_loaders(
    attr: str,
    image_size: int = 128,
    batch_size: int = 128,
    num_workers: int = 2,
) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """
    We use official CelebA splits:
      - train: training
      - valid: validation/tuning
      - test: final evaluation (clean + attack + defense)
    """
    train_ds = CelebABinary(split="train", attr=attr, transform=train_tfms(image_size))
    val_ds = CelebABinary(split="valid", attr=attr, transform=eval_tfms(image_size))
    test_ds = CelebABinary(split="test", attr=attr, transform=eval_tfms(image_size))

    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
    )
    test_loader = DataLoader(
        test_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
    )
    return train_loader, val_loader, test_loader
