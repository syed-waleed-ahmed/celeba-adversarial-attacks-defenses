from __future__ import annotations
import torch
import torch.nn.functional as F
from attacks.attack_utils import clamp_normalized, eps_pixel_to_norm

def pgd_attack(
    model: torch.nn.Module,
    x: torch.Tensor,
    y: torch.Tensor,
    eps_pixel: float,
    alpha_pixel: float,
    steps: int,
    random_start: bool = True
) -> torch.Tensor:
    """
    L-infinity PGD in normalized space (x is already normalized).
    eps_pixel/alpha_pixel are specified in pixel space [0,1] and converted per-channel.
    """
    if eps_pixel <= 0 or steps <= 0:
        return x

    device = x.device
    eps_norm = eps_pixel_to_norm(eps_pixel, device)
    alpha_norm = eps_pixel_to_norm(alpha_pixel, device)

    model.eval()

    x0 = x.detach()
    if random_start:
        x_adv = x0 + (2 * torch.rand_like(x0) - 1.0) * eps_norm
        x_adv = clamp_normalized(x_adv, device)
    else:
        x_adv = x0.clone()

    # Bulletproof: works even if caller is under no_grad()
    with torch.enable_grad():
        for _ in range(steps):
            x_adv = x_adv.detach().requires_grad_(True)
            logits = model(x_adv)
            loss = F.cross_entropy(logits, y)

            model.zero_grad(set_to_none=True)
            loss.backward()

            grad_sign = x_adv.grad.detach().sign()
            x_adv = x_adv + alpha_norm * grad_sign

            # Project to eps-ball around x0 (normalized space, per-channel)
            x_adv = torch.max(torch.min(x_adv, x0 + eps_norm), x0 - eps_norm)
            x_adv = clamp_normalized(x_adv.detach(), device)

    return x_adv