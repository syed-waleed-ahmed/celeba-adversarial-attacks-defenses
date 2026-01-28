# Adversarial Attacks and Defenses on the CelebA Dataset

This repository contains the code and experiments for the **Deep Learning final project** on adversarial attacks and defenses. We study the vulnerability of a convolutional neural network to adversarial examples and evaluate adversarial training as a defense strategy.

The project focuses on **binary facial attribute classification (Smiling)** using the **CelebA dataset** and a **ResNet-18** model.

---

## Project Overview

* **Dataset**: CelebA (Smiling attribute)
* **Model**: ResNet-18 pretrained on ImageNet
* **Attacks**:

  * Fast Gradient Sign Method (FGSM)
  * Projected Gradient Descent (PGD)
* **Defense**:

  * Adversarial training (FGSM-based)
* **Evaluation**:

  * Accuracy under clean inputs
  * Robustness under increasing perturbation strength (ε)
  * Qualitative visualization of adversarial examples

---

## Repository Structure

```
.
├── attacks/                 # FGSM and PGD implementations
├── defenses/                # Adversarial training
├── data/                    # CelebA dataset loader
├── models/                  # ResNet classifier
├── training/                # Training loops
├── evaluation/              # Evaluation, plotting, visualization
├── experiments/             # Entry-point scripts
├── results/
│   ├── figures/             # Accuracy vs epsilon plots
│   ├── metrics/             # JSON metrics
│   └── adv_examples/        # Example adversarial images
├── config.py                # Experiment configuration
├── requirements.txt         # Python dependencies
└── README.md
```

---

## Setup Instructions

### 1. Create and activate a virtual environment

**Windows (PowerShell):**

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

**macOS / Linux:**

```bash
python3 -m venv .venv
source .venv/bin/activate
```

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

---

## Running the Experiments

All commands should be run **from the project root directory**.

### 1. Train the baseline model

```bash
python -m experiments.run_baseline
```

This trains a ResNet-18 model on clean data and saves the best checkpoint to:

```
results/checkpoints/baseline_resnet18_smiling_best.pt
```

---

### 2. Evaluate baseline robustness (FGSM & PGD)

```bash
python -m experiments.run_attacks
```

This generates:

* Accuracy vs ε plots
* Robustness metrics (JSON)
* Example adversarial images

Outputs are saved in:

```
results/figures/
results/metrics/
results/adv_examples/baseline/
```

---

### 3. Train defended model (adversarial training)

```bash
python -m experiments.run_defense
```

By default, FGSM adversarial training is used.
The defended model checkpoint is saved to:

```
results/checkpoints/defended_fgsm_resnet18_smiling_best.pt
```

The script also evaluates robustness under FGSM and PGD and saves comparison plots.

---

### 4. Compare baseline vs defense

If included:

```bash
python -m experiments.compare_results
```

This generates a direct comparison plot between baseline and defended models.

---

## Results

Key results are visualized as **accuracy vs epsilon** plots:

* Baseline robustness
* Defended model robustness
* Baseline vs defense comparison

Qualitative examples show that adversarial perturbations are visually imperceptible but highly effective.

---

## Notes on Implementation

* Attacks are performed in **normalized input space**, with ε specified in **pixel space** and converted appropriately.
* FGSM and PGD are implemented as **white-box attacks**.
* Adversarial training mixes clean and adversarial samples during training.
* Evaluation strictly uses the **test split** to avoid data leakage.

---

## Reproducibility

All experiments are deterministic given a fixed random seed (defined in `config.py`).
CelebA is downloaded automatically using `torchvision.datasets.CelebA`.

---

## References

* Szegedy et al., *Intriguing Properties of Neural Networks*, ICLR 2014
* Goodfellow et al., *Explaining and Harnessing Adversarial Examples*, ICLR 2015
* Madry et al., *Towards Deep Learning Models Resistant to Adversarial Attacks*, ICLR 2018

---

## Authors

Group 16

* Rosen Stoev
* Syed Waleed Ahmed
* Alan Nessipbayev
* Matthijs van der Bent
