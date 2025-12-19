import torch


def get_device(force: str | None = None) -> str:
    if force is not None:
        return force
    return "cuda" if torch.cuda.is_available() else "cpu"


def set_seed(seed: int = 42) -> None:
    import os
    import random
    import numpy as np

    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)

    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
