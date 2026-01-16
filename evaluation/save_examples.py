from __future__ import annotations
import os
import torch
from torchvision.utils import save_image

from models.model_utils import ensure_dir
from attacks.fgsm import fgsm_attack
from attacks.pgd import pgd_attack
from attacks.attack_utils import to_device_stats

def denormalize(x_norm: torch.Tensor) -> torch.Tensor:
    device = x_norm.device
    mean, std = to_device_stats(device)
    x = x_norm * std + mean
    return torch.clamp(x, 0.0, 1.0)

def save_adversarial_examples(model, loader, out_dir: str, eps_pixel: float, method: str, max_batches: int = 1):
    ensure_dir(out_dir)
    device = next(model.parameters()).device
    model.eval()

    batches = 0
    for x, y in loader:
        x = x.to(device)
        y = y.to(device)

        if method == "fgsm":
            x_adv = fgsm_attack(model, x, y, eps_pixel=eps_pixel)
        elif method == "pgd":
            x_adv = pgd_attack(model, x, y, eps_pixel=eps_pixel, alpha_pixel=2/255, steps=10)
        else:
            raise ValueError("method must be fgsm or pgd")

        x_vis = denormalize(x)
        adv_vis = denormalize(x_adv)

        save_image(x_vis[:16], os.path.join(out_dir, f"clean_eps{eps_pixel:.6f}.png"), nrow=4)
        save_image(adv_vis[:16], os.path.join(out_dir, f"{method}_eps{eps_pixel:.6f}.png"), nrow=4)
        save_image(torch.abs(adv_vis[:16]-x_vis[:16]),
                   os.path.join(out_dir, f"delta_{method}_eps{eps_pixel:.6f}.png"), nrow=4)

        batches += 1
        if batches >= max_batches:
            break