from __future__ import annotations
import os
import random
from typing import Any, Dict
import numpy as np
import torch

def get_device() -> torch.device:
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")

def set_seed(seed: int = 42) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = True

def ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)

def save_checkpoint(path: str, model: torch.nn.Module, optimizer: torch.optim.Optimizer,
                    epoch: int, best_val_acc: float, extra: Dict[str, Any] | None = None) -> None:
    ensure_dir(os.path.dirname(path))
    ckpt = {
        "model_state": model.state_dict(),
        "optimizer_state": optimizer.state_dict(),
        "epoch": epoch,
        "best_val_acc": best_val_acc
    }
    if extra:
        ckpt.update(extra)
    torch.save(ckpt, path)

def load_checkpoint(path: str, model: torch.nn.Module, device: torch.device) -> Dict[str, Any]:
    ckpt = torch.load(path, map_location=device)
    model.load_state_dict(ckpt["model_state"])
    return ckpt