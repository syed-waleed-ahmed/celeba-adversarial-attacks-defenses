from __future__ import annotations
import torch
import torch.nn as nn
from torch.amp import autocast, GradScaler
from models.model_utils import get_device, save_checkpoint
from training.train_baseline import evaluate_accuracy
from attacks.fgsm import fgsm_attack
from attacks.pgd import pgd_attack

def train_adversarial(
    model: torch.nn.Module,
    train_loader,
    val_loader,
    epochs: int,
    lr: float,
    weight_decay: float,
    amp: bool,
    save_path: str,
    method: str,
    eps_pixel: float,
    pgd_steps: int = 10,
    pgd_alpha_pixel: float = 2 / 255,
    mix_clean: float = 0.5
) -> None:
    """
    Adversarial training: generate adversarial examples on-the-fly and train on a mix.
    mix_clean=0.5 means ~50% clean + ~50% adversarial (randomized selection each batch).
    """
    device = get_device()
    model.to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    scaler = GradScaler("cuda", enabled=(amp and device.type == "cuda"))

    best_val_acc = -1.0
    method_l = method.lower()

    for epoch in range(1, epochs + 1):
        model.train()
        running_loss = 0.0

        for x, y in train_loader:
            x = x.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)

            # Generate adversarial examples in eval mode
            model.eval()
            if method_l == "fgsm":
                x_adv = fgsm_attack(model, x, y, eps_pixel=eps_pixel)
            elif method_l == "pgd":
                with torch.enable_grad():
                    x_adv = pgd_attack(
                        model, x, y,
                        eps_pixel=eps_pixel,
                        alpha_pixel=pgd_alpha_pixel,
                        steps=pgd_steps,
                        random_start=True
                    )
            else:
                raise ValueError("method must be 'fgsm' or 'pgd'")

            # Back to train mode for optimization
            model.train()

            # Random mix clean & adv
            if 0.0 < mix_clean < 1.0:
                n = x.size(0)
                k = int(round(n * mix_clean))
                perm = torch.randperm(n, device=device)
                clean_idx = perm[:k]
                adv_idx = perm[k:]

                x_mix = x.clone()
                x_mix[adv_idx] = x_adv[adv_idx]
                y_mix = y
            elif mix_clean <= 0.0:
                x_mix, y_mix = x_adv, y
            else:
                x_mix, y_mix = x, y

            optimizer.zero_grad(set_to_none=True)

            with autocast("cuda", enabled=(amp and device.type == "cuda")):
                logits = model(x_mix)
                loss = criterion(logits, y_mix)

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            running_loss += loss.item()

        val_acc = evaluate_accuracy(model, val_loader, device)
        avg_loss = running_loss / max(len(train_loader), 1)
        print(f"[Defense-{method.upper()}] Epoch {epoch}/{epochs} | loss={avg_loss:.4f} | val_acc={val_acc:.4f}")

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            save_checkpoint(
                save_path, model, optimizer, epoch, best_val_acc,
                extra={"adv_method": method, "train_eps_pixel": eps_pixel}
            )

    print(f"[Defense-{method.upper()}] Best val acc: {best_val_acc:.4f}")
    print(f"[Defense-{method.upper()}] Saved: {save_path}")