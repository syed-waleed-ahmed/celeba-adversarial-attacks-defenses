# CelebA Adversarial Attacks & Defenses (Group 16)

We fine-tune an ImageNet-pretrained CNN (ResNet-18) on a CelebA **binary attribute classification** task (e.g., Smiling),
then evaluate adversarial **attacks** (FGSM, optional PGD) and a **defense** (adversarial training).

## Team split
- **Waleed**: core pipeline (data/model/training/clean evaluation)
- **Rosen**: attacks (FGSM/PGD + attacked evaluation)
- **Alan**: defenses (adversarial training + defended evaluation)
- **Matthijs**: experiments (plots, sweeps, result aggregation)

## Repo structure
- `src/` reusable code (data, models, training, evaluation, attacks, defenses)
- `scripts/` runnable entry points
- `outputs/` generated plots/tables (kept local, not committed)
- `report/` report files (tracked)

## Setup (Windows PowerShell)
```bash
python -m venv .venv
.venv\Scripts\Activate.ps1
pip install -r requirements.txt
