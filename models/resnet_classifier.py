from __future__ import annotations
import torch
import torch.nn as nn
from torchvision.models import resnet18, ResNet18_Weights

class ResNet18Binary(nn.Module):
    def __init__(self, pretrained: bool = True):
        super().__init__()
        weights = ResNet18_Weights.IMAGENET1K_V1 if pretrained else None
        self.backbone = resnet18(weights=weights)
        in_features = self.backbone.fc.in_features
        self.backbone.fc = nn.Linear(in_features, 2)  # binary classification via logits for 2 classes

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.backbone(x)