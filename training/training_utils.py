from __future__ import annotations
import os
import random
from dataclasses import dataclass
from typing import Dict, Any
import numpy as np
import torch

@dataclass
class TrainConfig:
    epochs: int = 5
    lr: float = 1e-4
    weight_decay: float = 1e-4
    amp: bool = True
    save_dir: str = "results/checkpoints"
    run_name: str = "baseline"


def set_seed(seed: int = 42) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = True  # speed
    # torch.backends.cudnn.deterministic = True  # set True only if you need strict determinism


def get_device() -> torch.device:
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def save_checkpoint(path: str, model: torch.nn.Module, optimizer: torch.optim.Optimizer,
                    epoch: int, best_val_acc: float, extra: Dict[str, Any] | None = None) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    payload = {
        "model_state": model.state_dict(),
        "optimizer_state": optimizer.state_dict(),
        "epoch": epoch,
        "best_val_acc": best_val_acc,
    }
    if extra:
        payload.update(extra)
    torch.save(payload, path)


@torch.no_grad()
def evaluate_accuracy(model: torch.nn.Module, loader, device: torch.device) -> float:
    model.eval()
    correct, total = 0, 0
    for x, y in loader:
        x = x.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)
        logits = model(x)
        preds = logits.argmax(dim=1)
        correct += (preds == y).sum().item()
        total += y.numel()
    return correct / max(total, 1)