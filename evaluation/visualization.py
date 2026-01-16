from __future__ import annotations
import os
from typing import Dict

import matplotlib.pyplot as plt

from models.model_utils import ensure_dir

def plot_acc_vs_eps(curves: Dict[str, Dict[float, float]], title: str, save_path: str) -> None:
    """
    curves: {"baseline_fgsm": {eps:acc, ...}, "defended_fgsm": {...}}
    """
    ensure_dir(os.path.dirname(save_path))
    plt.figure()
    for name, data in curves.items():
        xs = sorted(data.keys())
        ys = [data[x] for x in xs]
        plt.plot(xs, ys, marker="o", label=name)
    plt.xlabel("epsilon")
    plt.ylabel("accuracy")
    plt.title(title)
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(save_path, dpi=200)
    plt.close()