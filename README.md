# Adversarial Attacks & Defenses on CelebA (Attribute Classification)

This project studies adversarial robustness on the **CelebA (HOME)** dataset by training an **attribute classifier**
(e.g., Smiling vs Not Smiling) using an **ImageNet-pretrained ResNet (ResNet-18)**, applying **white-box attacks**
(**FGSM**, optional **PGD**), and evaluating a defense via **adversarial training**.

## Project scope (agreed)
- Dataset: **CelebA HOME** (aligned & cropped images + attribute labels + official train/val/test split)
- Task: **Binary attribute classification** (one attribute at a time)
- Model: **ImageNet-pretrained ResNet-18** (torchvision)
- Attack: **FGSM** (baseline), optional **PGD** (creative extension)
- Defense: **Adversarial training**
- Evaluation: Clean accuracy + accuracy under attack vs epsilon, and defense trade-off

## Repo layout
- `data/` dataset helper scripts (dataset itself is local, not committed)
- `models/` ResNet classifier + model utilities
- `training/` baseline + defended training pipelines
- `attacks/` FGSM (and optional PGD) implementations
- `defenses/` adversarial training implementation
- `evaluation/` metrics + evaluation + visualizations
- `experiments/` runnable scripts to reproduce baseline/attack/defense results
- `results/` generated figures/logs/checkpoints (ignored by git)
- `report/` report files
- `presentation/` slides material

## Setup (Windows PowerShell)
```bash
python -m venv .venv
.venv\Scripts\Activate.ps1
pip install -r requirements.txt
