from __future__ import annotations
import json
from typing import Dict, List, Literal
import torch
from models.model_utils import get_device, ensure_dir
from attacks.fgsm import fgsm_attack
from attacks.pgd import pgd_attack

AttackType = Literal["clean", "fgsm", "pgd"]

@torch.no_grad()
def eval_clean(model: torch.nn.Module, loader) -> float:
    device = get_device()
    model.eval()
    correct, total = 0, 0
    for x, y in loader:
        x = x.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)
        logits = model(x)
        pred = logits.argmax(dim=1)
        correct += (pred == y).sum().item()
        total += y.numel()
    return correct / max(total, 1)

def eval_under_attack(
    model: torch.nn.Module,
    loader,
    eps_list: List[float],
    attack: AttackType,
    pgd_steps: int = 10,
    pgd_alpha: float = 2/255
) -> Dict[float, float]:
    device = get_device()
    model.eval()
    results: Dict[float, float] = {}

    for eps in eps_list:
        correct, total = 0, 0

        for x, y in loader:
            x = x.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)

            if attack == "clean":
                x_adv = x
            elif attack == "fgsm":
                # fgsm_attack already enables grads internally
                x_adv = fgsm_attack(model, x, y, eps_pixel=eps)
            elif attack == "pgd":
                # pgd_attack is now bulletproof too, but keeping explicit grad is fine
                with torch.enable_grad():
                    x_adv = pgd_attack(
                        model, x, y,
                        eps_pixel=eps,
                        alpha_pixel=pgd_alpha,
                        steps=pgd_steps,
                        random_start=True
                    )
            else:
                raise ValueError("attack must be clean/fgsm/pgd")

            # Inference under no_grad
            with torch.no_grad():
                logits = model(x_adv)
                pred = logits.argmax(dim=1)
                correct += (pred == y).sum().item()
                total += y.numel()

        results[float(eps)] = correct / max(total, 1)
        print(f"[Eval {attack}] eps={eps:.6f} acc={results[float(eps)]:.4f}")

    return results

def save_metrics_json(path: str, payload: dict) -> None:
    out_dir = path.rsplit("/", 1)[0] if "/" in path else "."
    ensure_dir(out_dir)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)