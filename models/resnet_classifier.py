import torch
import torch.nn as nn
from torchvision import models


class ResNetAttributeClassifier(nn.Module):
    """
    ImageNet-pretrained ResNet (18/50) for binary attribute classification.
    Output: single logit per image.
    """

    def __init__(self, backbone: str = "resnet18", pretrained: bool = True):
        super().__init__()
        b = backbone.lower()

        if b == "resnet18":
            weights = models.ResNet18_Weights.DEFAULT if pretrained else None
            net = models.resnet18(weights=weights)
        elif b == "resnet50":
            weights = models.ResNet50_Weights.DEFAULT if pretrained else None
            net = models.resnet50(weights=weights)
        else:
            raise ValueError("backbone must be 'resnet18' or 'resnet50'")

        in_features = net.fc.in_features
        net.fc = nn.Linear(in_features, 1)  # binary logit
        self.net = net

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)  # [B, 1]
