from __future__ import annotations
import torch

IMAGENET_MEAN = torch.tensor([0.485, 0.456, 0.406]).view(1,3,1,1)
IMAGENET_STD  = torch.tensor([0.229, 0.224, 0.225]).view(1,3,1,1)

def to_device_stats(device: torch.device):
    return IMAGENET_MEAN.to(device), IMAGENET_STD.to(device)

def clamp_normalized(x_norm: torch.Tensor, device: torch.device) -> torch.Tensor:
    """
    x_norm = (x - mean)/std where x in [0,1].
    Clamp x to [0,1] by clamping x_norm to [(0-mean)/std, (1-mean)/std].
    """
    mean, std = to_device_stats(device)
    lo = (0.0 - mean) / std
    hi = (1.0 - mean) / std
    return torch.max(torch.min(x_norm, hi), lo)

def eps_pixel_to_norm(eps_pixel: float, device: torch.device) -> torch.Tensor:
    """Convert scalar eps in pixel-space [0,1] to per-channel eps in normalized space."""
    _, std = to_device_stats(device)
    return (torch.tensor(eps_pixel, device=device).view(1,1,1,1) / std)