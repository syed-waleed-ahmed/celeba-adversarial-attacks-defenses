import torch


@torch.no_grad()
def accuracy_from_logits(logits: torch.Tensor, y: torch.Tensor) -> float:
    """
    logits: [B] or [B,1]
    y: {0,1}
    """
    if logits.ndim == 2 and logits.size(1) == 1:
        logits = logits.squeeze(1)
    probs = torch.sigmoid(logits)
    preds = (probs >= 0.5).long()
    return (preds == y.long()).float().mean().item()
