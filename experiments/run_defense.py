from __future__ import annotations
import os

from config import DataConfig, TrainConfig, AttackConfig, Paths
from data.data_loader import CelebAConfig, make_loaders
from models.resnet_classifier import ResNetAttributeClassifier
from models.model_utils import set_seed, get_device, ensure_dir, load_checkpoint
from defenses.adversarial_training import train_adversarial
from evaluation.evaluate_model import eval_clean, eval_under_attack, save_metrics_json
from evaluation.visualization import plot_acc_vs_eps

def main():
    paths = Paths()
    ensure_dir(paths.checkpoints_dir)
    ensure_dir(paths.metrics_dir)
    ensure_dir(paths.figures_dir)

    data_cfg = DataConfig()
    train_cfg = TrainConfig()
    atk_cfg = AttackConfig()
    set_seed(train_cfg.seed)

    loaders_cfg = CelebAConfig(
        root=data_cfg.root,
        attribute=data_cfg.attribute,
        image_size=data_cfg.image_size,
        batch_size=data_cfg.batch_size,
        num_workers=data_cfg.num_workers,
    )
    train_loader, val_loader, test_loader = make_loaders(loaders_cfg)

    device = get_device()
    model = ResNetAttributeClassifier(pretrained=True).to(device)

    # Defense training config
    eps_train = 8/255
    defense_method = "fgsm"  # change to "pgd" for stronger defense (slower)

    defended_ckpt = os.path.join(paths.checkpoints_dir, f"defended_{defense_method}_resnet18_{data_cfg.attribute.lower()}_best.pt")
    train_adversarial(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        epochs=train_cfg.epochs,
        lr=train_cfg.lr,
        weight_decay=train_cfg.weight_decay,
        amp=train_cfg.amp,
        save_path=defended_ckpt,
        method=defense_method,
        eps_pixel=eps_train,
        pgd_steps=atk_cfg.pgd_steps,
        pgd_alpha_pixel=atk_cfg.pgd_alpha,
        mix_clean=0.5,
    )

    # Evaluate defended model
    load_checkpoint(defended_ckpt, model, device)
    model.to(device)

    clean_acc = eval_clean(model, test_loader)
    fgsm_curve = eval_under_attack(model, test_loader, atk_cfg.eps_list, attack="fgsm")
    pgd_curve  = eval_under_attack(model, test_loader, atk_cfg.eps_list, attack="pgd",
                                   pgd_steps=atk_cfg.pgd_steps, pgd_alpha=atk_cfg.pgd_alpha)

    metrics_path = os.path.join(paths.metrics_dir, f"defended_{defense_method}_{data_cfg.attribute.lower()}.json")
    save_metrics_json(metrics_path, {
        "model": f"defended_{defense_method}",
        "attribute": data_cfg.attribute,
        "train_eps": eps_train,
        "clean_test_acc": clean_acc,
        "fgsm": fgsm_curve,
        "pgd": pgd_curve,
        "eps_list": atk_cfg.eps_list,
    })

    plot_path = os.path.join(paths.figures_dir, f"defended_{defense_method}_acc_vs_eps_{data_cfg.attribute.lower()}.png")
    plot_acc_vs_eps(
        {f"Defended({defense_method.upper()})-FGSM": fgsm_curve,
         f"Defended({defense_method.upper()})-PGD": pgd_curve},
        title=f"Defended Robustness ({data_cfg.attribute})",
        save_path=plot_path
    )

    print(f"[Defense] Metrics: {metrics_path}")
    print(f"[Defense] Plot: {plot_path}")
    print(f"[Defense] Checkpoint: {defended_ckpt}")

if __name__ == "__main__":
    main()
