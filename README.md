# Adversarial Attacks & Defenses on CelebA

### Attribute Classification with Pretrained ResNet-18

This project studies **adversarial robustness of deep neural networks** using the **CelebA** dataset.
We train an **ImageNet-pretrained ResNet-18** for **binary facial attribute classification** (e.g., *Smiling vs Not Smiling*), evaluate its vulnerability to **white-box adversarial attacks** (FGSM), and analyze defenses via **adversarial training**.

The project is developed as part of the course **“Deep Learning: From Theory to Practice”**.

---

## Project Scope

* **Dataset**: CelebA (aligned & cropped images, official train/val/test split)
* **Task**: Binary attribute classification (one attribute at a time)
* **Model**: ImageNet-pretrained ResNet-18 (torchvision)
* **Attacks**:

  * FGSM (baseline, required)
  * PGD (optional extension)
* **Defense**: Adversarial training
* **Evaluation**:

  * Clean accuracy
  * Accuracy under attack vs ε
  * Robustness vs clean-accuracy trade-off

---

## Repository Structure

```
.
├── data/                 # Dataset loaders & helpers (CelebA not committed)
├── models/               # Model definitions (ResNet-18 classifier)
├── training/             # Baseline and defended training pipelines
├── attacks/              # FGSM / PGD implementations
├── defenses/             # Defense methods (adversarial training)
├── evaluation/            # Metrics, evaluation scripts, visualizations
├── experiments/           # Runnable scripts for experiments
├── results/               # Checkpoints, logs, figures (ignored by git)
├── report/                # Report files (LaTeX / figures)
├── requirements.txt
└── README.md
```

---

## Environment Setup (Windows + NVIDIA GPU)

### 1) Create virtual environment

```bash
python -m venv .venv
.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
```

### 2) Install PyTorch with CUDA (recommended)

For NVIDIA GPUs:

```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
```

### 3) Install remaining dependencies

```bash
pip install -r requirements.txt
```

### 4) Verify CUDA is available

```bash
python -c "import torch; print('CUDA available:', torch.cuda.is_available()); print(torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU')"
```

---

## Dataset: CelebA

* The CelebA dataset is **automatically downloaded** using `torchvision.datasets.CelebA`.
* Dataset files are stored locally under:

  ```
  data/celeba/
  ```
* **Dataset files are NOT committed** to the repository (see `.gitignore`).

Official dataset:

* https://mmlab.ie.cuhk.edu.hk/projects/CelebA.html

---

## Running the Baseline (Clean Training)

This trains a **ResNet-18** classifier on a chosen attribute (default: *Smiling*) and evaluates clean accuracy.

```bash
python -m experiments.run_baseline
```

Expected output:

* Training logs per epoch
* Best validation accuracy
* Clean test accuracy
* Checkpoint saved to:

  ```
  results/checkpoints/resnet18_smiling_best.pt
  ```

---

## Experiments Workflow

### Baseline

* Train clean model
* Evaluate clean accuracy

### Adversarial Attacks

* Generate adversarial examples (FGSM / PGD)
* Evaluate accuracy vs ε
* Save adversarial image samples and plots

### Defense

* Train model with adversarial training
* Compare clean vs attacked performance
* Analyze robustness–accuracy trade-offs

---

## Notes & Best Practices

* Images are resized to **128×128** for efficiency.
* Epsilon (ε) values are defined in **pixel scale** (e.g., 8/255).
* Official CelebA splits (train/val/test) are strictly respected.
* All experiments are reproducible via scripts in `experiments/`.