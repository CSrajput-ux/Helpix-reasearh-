from typing import Optional
import torch
import torch.nn as nn
from torch.utils.data import DataLoader


class TemperatureScaler(nn.Module):
    """
    Post-hoc temperature scaling calibration module.
    Scales logits by a single learnable scalar parameter T:
      p_calibrated = softmax(logits / T)
    Preserves top-1 predictions and ranking (AUROC invariant) while aligning confidence with true empirical likelihood.
    """
    def __init__(self, initial_temperature: float = 1.5):
        super().__init__()
        self.temperature = nn.Parameter(torch.ones(1) * initial_temperature)

    def forward(self, logits: torch.Tensor) -> torch.Tensor:
        return logits / self.temperature.clamp(min=0.05)


def fit_temperature(
    model: nn.Module,
    val_loader: DataLoader,
    device: Optional[torch.device] = None,
    lr: float = 0.01,
    max_iter: int = 50
) -> TemperatureScaler:
    """
    Fit temperature parameter T exclusively on the validation set using L-BFGS.
    Never fit or adjust temperature scaling on the internal or external test sets.
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model.eval()
    scaler = TemperatureScaler().to(device)
    logits_list, labels_list = [], []

    with torch.no_grad():
        for x, y in val_loader:
            logits_list.append(model(x.to(device)).cpu())
            labels_list.append(y)

    logits = torch.cat(logits_list).to(device)
    labels = torch.cat(labels_list).to(device)

    optimizer = torch.optim.LBFGS([scaler.temperature], lr=lr, max_iter=max_iter)
    criterion = nn.CrossEntropyLoss()

    def closure():
        optimizer.zero_grad()
        loss = criterion(scaler(logits), labels)
        loss.backward()
        return loss

    optimizer.step(closure)
    return scaler
