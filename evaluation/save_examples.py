from __future__ import annotations
import os
import torch
from torchvision.utils import save_image
from models.model_utils import ensure_dir
from attacks.fgsm import fgsm_attack
from attacks.pgd import pgd_attack
from attacks.attack_utils import to_device_stats

def denormalize(x_norm: torch.Tensor) -> torch.Tensor:
    """
    Invert ImageNet normalization:
      x_norm = (x - mean) / std
      x      = x_norm * std + mean
    Returns x in [0,1] for visualization/saving.
    """
    device = x_norm.device
    mean, std = to_device_stats(device)
    x = x_norm * std + mean
    return torch.clamp(x, 0.0, 1.0)

def _scale_for_visibility(delta: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    """
    Scale a delta image to [0,1] for visualization.
    This does NOT change the attack; it only makes the difference map visible.
    """
    # Normalize per-batch (global max) to preserve relative patterns
    m = delta.max().clamp(min=eps)
    return torch.clamp(delta / m, 0.0, 1.0)

def save_adversarial_examples(
    model: torch.nn.Module,
    loader,
    out_dir: str,
    eps_pixel: float,
    method: str,
    max_batches: int = 1,
    pgd_steps: int = 10,
    pgd_alpha_pixel: float = 2 / 255,
    n_save: int = 16,
    nrow: int = 4
) -> None:
    """
    Saves clean, adversarial, and delta images for a few batches.
    - clean_eps*.png: clean images
    - {method}_eps*.png: adversarial images
    - delta_{method}_raw_eps*.png: raw absolute difference (often looks dark)
    - delta_{method}_scaled_eps*.png: scaled difference (good for report)
    """
    ensure_dir(out_dir)
    device = next(model.parameters()).device
    model.eval()

    method_l = method.lower()
    batches = 0

    for x, y in loader:
        x = x.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)

        # Generate adversarial examples
        if method_l == "fgsm":
            x_adv = fgsm_attack(model, x, y, eps_pixel=eps_pixel)
        elif method_l == "pgd":
            # Extra safety: ensure grads are enabled
            with torch.enable_grad():
                x_adv = pgd_attack(
                    model,
                    x,
                    y,
                    eps_pixel=eps_pixel,
                    alpha_pixel=pgd_alpha_pixel,
                    steps=pgd_steps,
                    random_start=True
                )
        else:
            raise ValueError("method must be 'fgsm' or 'pgd'")

        # Denormalize for saving
        x_vis = denormalize(x)
        adv_vis = denormalize(x_adv)

        # Compute delta in pixel space
        delta_raw = torch.abs(adv_vis - x_vis)
        delta_scaled = _scale_for_visibility(delta_raw)

        # Save a grid
        save_image(
            x_vis[:n_save],
            os.path.join(out_dir, f"clean_eps{eps_pixel:.6f}.png"),
            nrow=nrow
        )
        save_image(
            adv_vis[:n_save],
            os.path.join(out_dir, f"{method_l}_eps{eps_pixel:.6f}.png"),
            nrow=nrow
        )
        # Raw delta
        save_image(
            delta_raw[:n_save],
            os.path.join(out_dir, f"delta_{method_l}_raw_eps{eps_pixel:.6f}.png"),
            nrow=nrow
        )
        # Scaled delta
        save_image(
            delta_scaled[:n_save],
            os.path.join(out_dir, f"delta_{method_l}_scaled_eps{eps_pixel:.6f}.png"),
            nrow=nrow
        )

        batches += 1
        if batches >= max_batches:
            break