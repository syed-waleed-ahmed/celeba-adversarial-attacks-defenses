from __future__ import annotations
import os

from config import DataConfig, AttackConfig, Paths
from data.data_loader import CelebAConfig, make_loaders
from models.resnet_classifier import ResNetAttributeClassifier
from models.model_utils import get_device, load_checkpoint, ensure_dir
from evaluation.evaluate_model import eval_clean, eval_under_attack, save_metrics_json
from evaluation.visualization import plot_acc_vs_eps
from evaluation.save_examples import save_adversarial_examples

def main():
    paths = Paths()
    ensure_dir(paths.metrics_dir)
    ensure_dir(paths.figures_dir)
    ensure_dir(paths.adv_examples_dir)

    data_cfg = DataConfig()
    atk_cfg = AttackConfig()

    loaders_cfg = CelebAConfig(
        root=data_cfg.root,
        attribute=data_cfg.attribute,
        image_size=data_cfg.image_size,
        batch_size=data_cfg.batch_size,
        num_workers=data_cfg.num_workers,
    )
    _, _, test_loader = make_loaders(loaders_cfg)

    device = get_device()
    model = ResNetAttributeClassifier(pretrained=False).to(device)

    baseline_ckpt = os.path.join(paths.checkpoints_dir, f"baseline_resnet18_{data_cfg.attribute.lower()}_best.pt")
    load_checkpoint(baseline_ckpt, model, device)

    clean_acc = eval_clean(model, test_loader)
    fgsm_curve = eval_under_attack(model, test_loader, atk_cfg.eps_list, attack="fgsm")
    pgd_curve  = eval_under_attack(model, test_loader, atk_cfg.eps_list, attack="pgd",
                                   pgd_steps=atk_cfg.pgd_steps, pgd_alpha=atk_cfg.pgd_alpha)

    metrics_path = os.path.join(paths.metrics_dir, f"baseline_attack_{data_cfg.attribute.lower()}.json")
    save_metrics_json(metrics_path, {
        "model": "baseline",
        "attribute": data_cfg.attribute,
        "clean_test_acc": clean_acc,
        "fgsm": fgsm_curve,
        "pgd": pgd_curve,
        "eps_list": atk_cfg.eps_list,
    })

    plot_path = os.path.join(paths.figures_dir, f"baseline_acc_vs_eps_{data_cfg.attribute.lower()}.png")
    plot_acc_vs_eps(
        {"FGSM": fgsm_curve, "PGD": pgd_curve},
        title=f"Baseline Robustness ({data_cfg.attribute})",
        save_path=plot_path
    )

    # save some example images at a representative epsilon
    eps_show = 8/255
    save_adversarial_examples(model, test_loader,
                              out_dir=os.path.join(paths.adv_examples_dir, "baseline"),
                              eps_pixel=eps_show, method="fgsm", max_batches=1)
    save_adversarial_examples(model, test_loader,
                              out_dir=os.path.join(paths.adv_examples_dir, "baseline"),
                              eps_pixel=eps_show, method="pgd", max_batches=1)

    print(f"[Attack] Metrics: {metrics_path}")
    print(f"[Attack] Plot: {plot_path}")

if __name__ == "__main__":
    main()