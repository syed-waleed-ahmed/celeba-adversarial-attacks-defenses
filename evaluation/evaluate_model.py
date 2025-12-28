from __future__ import annotations
import argparse
from pathlib import Path
import torch
from data.data_loader import CelebAConfig, make_loaders
from models.model_utils import get_device
from models.resnet_classifier import ResNetAttributeClassifier
from evaluation.metrics import accuracy_from_logits

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt", type=str, required=True)
    p.add_argument("--batch_size", type=int, default=128)
    p.add_argument("--image_size", type=int, default=128)
    p.add_argument("--num_workers", type=int, default=2)
    p.add_argument("--device", type=str, default=None)
    return p.parse_args()

@torch.no_grad()
def main():
    args = parse_args()
    device = get_device(args.device)
    print(f"Device: {device}")

    ckpt = torch.load(Path(args.ckpt), map_location="cpu")
    attr = ckpt["attr"]
    backbone = ckpt.get("backbone", "resnet18")

    cfg = CelebAConfig(attr=attr, image_size=args.image_size, batch_size=args.batch_size, num_workers=args.num_workers)
    _train, _val, test_loader = make_loaders(cfg)

    model = ResNetAttributeClassifier(backbone=backbone, pretrained=False).to(device)
    model.load_state_dict(ckpt["model_state"])
    model.eval()

    accs = []
    for x, y in test_loader:
        x = x.to(device)
        y = y.to(device)
        logits = model(x).squeeze(1)
        accs.append(accuracy_from_logits(logits, y))

    acc = sum(accs) / max(1, len(accs))
    print(f"Clean TEST accuracy ({attr}) = {acc:.4f}")

if __name__ == "__main__":
    main()