from __future__ import annotations
import os
from config import DataConfig, TrainConfig, Paths
from data.data_loader import CelebAConfig, make_loaders
from models.resnet_classifier import ResNetAttributeClassifier
from models.model_utils import set_seed, get_device, ensure_dir, load_checkpoint
from training.train_baseline import train_baseline, evaluate_accuracy

def main():
    paths = Paths()
    ensure_dir(paths.checkpoints_dir)

    data_cfg = DataConfig()
    train_cfg = TrainConfig()
    set_seed(train_cfg.seed)

    loaders_cfg = CelebAConfig(
        root=data_cfg.root,
        attribute=data_cfg.attribute,
        image_size=data_cfg.image_size,
        batch_size=data_cfg.batch_size,
        num_workers=data_cfg.num_workers,
    )
    train_loader, val_loader, test_loader = make_loaders(loaders_cfg)

    model = ResNetAttributeClassifier(pretrained=True)

    ckpt_path = os.path.join(paths.checkpoints_dir, f"baseline_resnet18_{data_cfg.attribute.lower()}_best.pt")
    train_baseline(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        epochs=train_cfg.epochs,
        lr=train_cfg.lr,
        weight_decay=train_cfg.weight_decay,
        amp=train_cfg.amp,
        save_path=ckpt_path,
    )

    # Evaluate clean test with best ckpt
    device = get_device()
    load_checkpoint(ckpt_path, model, device)
    model.to(device)
    test_acc = evaluate_accuracy(model, test_loader, device)
    print(f"[Baseline] Clean test accuracy: {test_acc:.4f}")
    print(f"[Baseline] Checkpoint: {ckpt_path}")

if __name__ == "__main__":
    main()