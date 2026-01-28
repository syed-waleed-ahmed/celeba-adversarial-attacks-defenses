from __future__ import annotations
import json
import os

from config import DataConfig, Paths
from models.model_utils import ensure_dir
from evaluation.visualization import plot_acc_vs_eps

def load_json(path: str) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)

def main():
    paths = Paths()
    ensure_dir(paths.figures_dir)

    data_cfg = DataConfig()

    baseline_metrics = os.path.join(paths.metrics_dir, f"baseline_attack_{data_cfg.attribute.lower()}.json")
    defended_metrics = os.path.join(paths.metrics_dir, f"defended_fgsm_{data_cfg.attribute.lower()}.json")

    b = load_json(baseline_metrics)
    d = load_json(defended_metrics)

    # Curves are stored with string keys by json; convert to float
    def to_float_keys(curve: dict) -> dict:
        return {float(k): float(v) for k, v in curve.items()}

    curves = {
        "Baseline-FGSM": to_float_keys(b["fgsm"]),
        "Defended-FGSM": to_float_keys(d["fgsm"]),
        "Baseline-PGD": to_float_keys(b["pgd"]),
        "Defended-PGD": to_float_keys(d["pgd"]),
    }

    save_path = os.path.join(paths.figures_dir, f"compare_baseline_vs_defense_{data_cfg.attribute.lower()}.png")
    plot_acc_vs_eps(curves, title=f"Baseline vs Defense ({data_cfg.attribute})", save_path=save_path)
    print(f"[Compare] Saved: {save_path}")

if __name__ == "__main__":
    main()