from typing import Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F


@torch.no_grad()
def mc_dropout_predict(
    model: nn.Module,
    x: torch.Tensor,
    n_samples: int = 20
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Monte Carlo Dropout predictive uncertainty estimation:
    Maintains Dropout active while keeping BatchNorm layers locked in evaluation mode.
    Returns:
      - mean_probs: Tensor [B, C] of empirical predictive probabilities
      - uncertainty: Tensor [B] of predictive standard deviation across classes
    """
    model.train()
    for m in model.modules():
        if isinstance(m, nn.BatchNorm2d):
            m.eval()

    probs_list = []
    for _ in range(n_samples):
        logits = model(x)
        probs_list.append(F.softmax(logits, dim=1).unsqueeze(0))

    probs_stack = torch.cat(probs_list, dim=0)  # (n_samples, B, C)
    mean_probs = probs_stack.mean(dim=0)
    uncertainty = probs_stack.std(dim=0).mean(dim=1)
    model.eval()

    return mean_probs.cpu(), uncertainty.cpu()
