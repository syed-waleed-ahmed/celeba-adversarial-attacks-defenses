from __future__ import annotations
import torch
import torch.nn as nn
from torch.amp import autocast, GradScaler
from models.model_utils import get_device, save_checkpoint

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


def train_baseline(
    model: torch.nn.Module,
    train_loader,
    val_loader,
    epochs: int,
    lr: float,
    weight_decay: float,
    amp: bool,
    save_path: str
) -> None:
    device = get_device()
    model.to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    scaler = GradScaler("cuda", enabled=(amp and device.type == "cuda"))

    best_val_acc = -1.0

    for epoch in range(1, epochs + 1):
        model.train()
        running_loss = 0.0
        correct, total = 0, 0

        for x, y in train_loader:
            x = x.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)

            optimizer.zero_grad(set_to_none=True)

            with autocast("cuda", enabled=(amp and device.type == "cuda")):
                logits = model(x)
                loss = criterion(logits, y)

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            running_loss += loss.item()
            preds = logits.argmax(dim=1)
            correct += (preds == y).sum().item()
            total += y.numel()

        train_acc = correct / max(total, 1)
        val_acc = evaluate_accuracy(model, val_loader, device)
        avg_loss = running_loss / max(len(train_loader), 1)

        print(f"[Baseline] Epoch {epoch}/{epochs} | loss={avg_loss:.4f} | train_acc={train_acc:.4f} | val_acc={val_acc:.4f}")

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            save_checkpoint(save_path, model, optimizer, epoch, best_val_acc)

    print(f"[Baseline] Best val acc: {best_val_acc:.4f}")
    print(f"[Baseline] Saved: {save_path}")