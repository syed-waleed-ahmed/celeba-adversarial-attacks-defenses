import torch
from torchvision.datasets import CelebA

from src.utils.paths import data_dir


def _attr_index(dataset: CelebA, attr_name: str) -> int:
    try:
        return dataset.attr_names.index(attr_name)
    except ValueError as e:
        raise ValueError(
            f"Unknown CelebA attribute '{attr_name}'. "
            "Examples: Smiling, Eyeglasses, Male, Young, Blond_Hair"
        ) from e


class CelebABinary(torch.utils.data.Dataset):
    """
    CelebA wrapper returning (image, y) for one attribute.
    CelebA attributes are in {-1, +1}; we convert to {0, 1}.
    """

    def __init__(self, split: str, attr: str, transform):
        self.base = CelebA(
            root=str(data_dir()),
            split=split,  # 'train' | 'valid' | 'test'
            target_type="attr",
            download=False,  # IMPORTANT: avoid GDrive quota issues; user places files locally
            transform=transform,
        )
        self.attr = attr
        self.idx = _attr_index(self.base, attr)

    def __len__(self) -> int:
        return len(self.base)

    def __getitem__(self, i: int):
        x, attrs = self.base[i]  # attrs shape [40], values -1/+1
        y = (attrs[self.idx] == 1).long()
        return x, y
