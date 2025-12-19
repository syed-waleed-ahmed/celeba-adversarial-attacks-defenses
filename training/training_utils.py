from __future__ import annotations

from pathlib import Path
from typing import Dict, Tuple

import torch
from tqdm import tqdm

from evaluation.metrics import accuracy_from_logits


def get_loss():
    return torch.nn.BCEWithLogitsLoss()


def save_checkpoint(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(payload, path)


def train_one_epoch(model, loader, criterion, optimizer, device: str) -> Dict[str, float]:
    model.train()
    total_loss, total_acc, n = 0.0, 0.0, 0

    for x, y in tqdm(loader, desc="train", leave=False):
        x = x.to(device)
        y = y.float().to(device)

        logits = model(x).squeeze(1)
        loss = criterion(logits, y)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += float(loss.item())
        total_acc += accuracy_from_logits(logits, y)
        n += 1

    return {"loss": total_loss / max(1, n), "acc": total_acc / max(1, n)}


@torch.no_grad()
def evaluate(model, loader, criterion, device: str) -> Dict[str, float]:
    model.eval()
    total_loss, total_acc, n = 0.0, 0.0, 0

    for x, y in tqdm(loader, desc="eval", leave=False):
        x = x.to(device)
        y = y.float().to(device)

        logits = model(x).squeeze(1)
        loss = criterion(logits, y)

        total_loss += float(loss.item())
        total_acc += accuracy_from_logits(logits, y)
        n += 1

    return {"loss": total_loss / max(1, n), "acc": total_acc / max(1, n)}
