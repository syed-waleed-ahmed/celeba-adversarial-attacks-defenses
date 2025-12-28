from __future__ import annotations
import torch
import torch.nn as nn
from torch.cuda.amp import autocast, GradScaler
from training.training_utils import TrainConfig, get_device, evaluate_accuracy, save_checkpoint

def train_baseline(model: torch.nn.Module,
                   train_loader,
                   val_loader,
                   cfg: TrainConfig) -> str:
    device = get_device()
    model = model.to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    scaler = GradScaler(enabled=(cfg.amp and device.type == "cuda"))

    best_val_acc = -1.0
    best_path = f"{cfg.save_dir}/{cfg.run_name}_best.pt"

    for epoch in range(1, cfg.epochs + 1):
        model.train()
        running_loss = 0.0

        for x, y in train_loader:
            x = x.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)

            optimizer.zero_grad(set_to_none=True)

            with autocast(enabled=(cfg.amp and device.type == "cuda")):
                logits = model(x)
                loss = criterion(logits, y)

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            running_loss += loss.item()

        val_acc = evaluate_accuracy(model, val_loader, device)
        avg_loss = running_loss / max(len(train_loader), 1)

        print(f"Epoch {epoch}/{cfg.epochs} | loss={avg_loss:.4f} | val_acc={val_acc:.4f}")

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            save_checkpoint(best_path, model, optimizer, epoch, best_val_acc)

    print(f"Best val acc: {best_val_acc:.4f}")
    return best_path