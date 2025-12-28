from __future__ import annotations
from data.celeba_dataset import CelebAConfig, get_celeba_dataloaders
from models.resnet_classifier import ResNet18Binary
from training.training_utils import TrainConfig, set_seed, get_device, evaluate_accuracy
from training.train_baseline import train_baseline

import torch

def main():
    set_seed(42)

    data_cfg = CelebAConfig(
        root="data/celeba",
        attribute="Smiling",
        image_size=128,
        batch_size=64,
        num_workers=4,
        pin_memory=True,
    )
    train_loader, val_loader, test_loader = get_celeba_dataloaders(data_cfg)

    model = ResNet18Binary(pretrained=True)

    train_cfg = TrainConfig(
        epochs=5,
        lr=1e-4,
        weight_decay=1e-4,
        amp=True,
        save_dir="results/checkpoints",
        run_name="resnet18_smiling",
    )

    best_path = train_baseline(model, train_loader, val_loader, train_cfg)

    # Load best and evaluate test
    device = get_device()
    ckpt = torch.load(best_path, map_location=device)
    model.load_state_dict(ckpt["model_state"])
    model = model.to(device)

    test_acc = evaluate_accuracy(model, test_loader, device)
    print(f"Clean test accuracy: {test_acc:.4f}")
    print(f"Saved best checkpoint: {best_path}")

if __name__ == "__main__":
    main()