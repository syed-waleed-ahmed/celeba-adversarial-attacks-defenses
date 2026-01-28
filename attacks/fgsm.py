from __future__ import annotations
import torch
import torch.nn.functional as F
from attacks.attack_utils import clamp_normalized, eps_pixel_to_norm


def fgsm_attack(
    model: torch.nn.Module,
    x_norm: torch.Tensor,
    y: torch.Tensor,
    eps_pixel: float
) -> torch.Tensor:
    """
    FGSM on normalized inputs.
    x_norm is already normalized (ImageNet mean/std).
    eps_pixel is in pixel space (e.g., 8/255).
    """
    if eps_pixel <= 0:
        return x_norm

    device = x_norm.device
    eps_norm = eps_pixel_to_norm(eps_pixel, device)

    model.eval()
    
    with torch.enable_grad():
        x_adv = x_norm.detach().clone().requires_grad_(True)
        logits = model(x_adv)
        loss = F.cross_entropy(logits, y)

        model.zero_grad(set_to_none=True)
        loss.backward()

        grad_sign = x_adv.grad.detach().sign()
        x_adv = x_adv + eps_norm * grad_sign

    x_adv = clamp_normalized(x_adv.detach(), device)
    return x_adv