from __future__ import annotations

import argparse
from pathlib import Path

import torch

from data.data_loader import CelebAConfig, make_loaders
from models.model_utils import get_device, set_seed
from models.resnet_classifier import ResNetAttributeClassifier
from training.training_utils import get_loss, save_checkpoint, train_one_epoch, evaluate


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--attr", type=str, default="Smiling")
    p.add_argument("--epochs", type=int, default=3)
    p.add_argument("--batch_size", type=int, default=128)
    p.add_argument("--image_size", type=int, default=128)
    p.add_argument("--num_workers", type=int, default=2)
    p.add_argument("--backbone", type=str, default="resnet18")  # resnet18/resnet50
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--device", type=str, default=None)          # cpu/cuda
    return p.parse_args()


def main():
    args = parse_args()
    set_seed(args.seed)

    device = get_device(args.device)
    print(f"Device: {device}")

    cfg = CelebAConfig(
        attr=args.attr,
        image_size=args.image_size,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
    )
    train_loader, val_loader, _test_loader = make_loaders(cfg)

    model = ResNetAttributeClassifier(backbone=args.backbone, pretrained=True).to(device)
    criterion = get_loss()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    best_val_acc = -1.0
    ckpt_path = Path("results/checkpoints") / f"baseline_{args.attr.lower()}_{args.backbone}.pt"

    for epoch in range(1, args.epochs + 1):
        tr = train_one_epoch(model, train_loader, criterion, optimizer, device)
        va = evaluate(model, val_loader, criterion, device)
        print(
            f"Epoch {epoch}: "
            f"train loss={tr['loss']:.4f} acc={tr['acc']:.4f} | "
            f"val loss={va['loss']:.4f} acc={va['acc']:.4f}"
        )

        if va["acc"] > best_val_acc:
            best_val_acc = va["acc"]
            save_checkpoint(
                ckpt_path,
                {
                    "attr": args.attr,
                    "backbone": args.backbone,
                    "model_state": model.state_dict(),
                },
            )
            print(f"Saved best checkpoint -> {ckpt_path}")

    print(f"Done. Best val acc: {best_val_acc:.4f}")


if __name__ == "__main__":
    main()
