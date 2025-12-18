import torch
import torch.nn as nn
from torchvision import models


class ResNetBinary(nn.Module):
    """
    ImageNet-pretrained ResNet (18 or 50) with a binary head.
    Output: single logit per image.
    """

    def __init__(self, variant: str = "resnet18", pretrained: bool = True):
        super().__init__()
        v = variant.lower()

        if v == "resnet18":
            weights = models.ResNet18_Weights.DEFAULT if pretrained else None
            backbone = models.resnet18(weights=weights)
        elif v == "resnet50":
            weights = models.ResNet50_Weights.DEFAULT if pretrained else None
            backbone = models.resnet50(weights=weights)
        else:
            raise ValueError("variant must be 'resnet18' or 'resnet50'")

        in_features = backbone.fc.in_features
        backbone.fc = nn.Linear(in_features, 1)
        self.backbone = backbone

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.backbone(x)  # [B, 1]
