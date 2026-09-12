from pathlib import Path
from typing import Tuple, Dict, Any, Optional
import time
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from sklearn.metrics import roc_auc_score
from .losses import FocalLoss


def train_epoch(
    model: nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    criterion: nn.Module,
    scaler: torch.amp.GradScaler,
    device: torch.device
) -> float:
    """Execute one training epoch with automated mixed precision (AMP)."""
    model.train()
    total_loss = 0.0
    total_samples = 0
    is_cuda = (device.type == 'cuda')

    for x, y in loader:
        x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
        optimizer.zero_grad(set_to_none=True)
        with torch.amp.autocast('cuda', enabled=is_cuda):
            out = model(x)
            loss = criterion(out, y)
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        bs = x.size(0)
        total_loss += loss.item() * bs
        total_samples += bs

    return total_loss / max(total_samples, 1)


@torch.no_grad()
def eval_epoch(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device
) -> Tuple[float, np.ndarray, np.ndarray]:
    """Execute evaluation pass; returns (mean_loss, softmax_probabilities [N, 2], ground_truth_labels [N])."""
    model.eval()
    total_loss = 0.0
    all_probs = []
    all_labels = []
    is_cuda = (device.type == 'cuda')

    for x, y in loader:
        x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
        with torch.amp.autocast('cuda', enabled=is_cuda):
            out = model(x)
            loss = criterion(out, y)
        total_loss += loss.item() * x.size(0)
        probs = F.softmax(out, dim=1)
        all_probs.extend(probs.cpu().numpy())
        all_labels.extend(y.cpu().numpy())

    avg_loss = total_loss / max(len(loader.dataset), 1)
    return avg_loss, np.array(all_probs), np.array(all_labels)


def fit_model(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    epochs: int = 15,
    lr: float = 1e-4,
    patience: int = 3,
    ckpt_path: str | Path = "model_best.pt",
    device: Optional[torch.device] = None
) -> nn.Module:
    """
    Standard model training loop with AdamW, ReduceLROnPlateau, FocalLoss, and early stopping.
    Saves only the best checkpoint based on validation loss.
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = model.to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', patience=2, factor=0.5
    )
    criterion = FocalLoss()
    scaler = torch.amp.GradScaler('cuda', enabled=(device.type == 'cuda'))

    best_val_loss = float('inf')
    no_improve = 0

    for epoch in range(epochs):
        train_loss = train_epoch(model, train_loader, optimizer, criterion, scaler, device)
        if device.type == 'cuda':
            torch.cuda.empty_cache()

        val_loss, val_probs, val_labels = eval_epoch(model, val_loader, criterion, device)
        scheduler.step(val_loss)

        if len(np.unique(val_labels)) > 1:
            val_auroc = roc_auc_score(val_labels, val_probs[:, 1])
        else:
            val_auroc = float('nan')

        print(f"Epoch {epoch+1:02d}/{epochs:02d} | Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f} | Val AUROC: {val_auroc:.4f}")

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            no_improve = 0
            torch.save(model.state_dict(), ckpt_path)
        else:
            no_improve += 1
            if no_improve >= patience:
                print("Early stopping triggered.")
                break

        if device.type == 'cuda':
            torch.cuda.empty_cache()

    if Path(ckpt_path).exists():
        model.load_state_dict(torch.load(ckpt_path, map_location=device, weights_only=True))
    return model
