from typing import Dict, Any
import numpy as np
from sklearn.metrics import roc_auc_score, f1_score, confusion_matrix


def expected_calibration_error(probs: np.ndarray, labels: np.ndarray, n_bins: int = 10) -> float:
    """
    Expected Calibration Error (ECE):
    Computes area-weighted mean absolute difference between accuracy and mean confidence across uniform confidence bins.
    """
    boundaries = np.linspace(0, 1, n_bins + 1)
    conf = np.max(probs, axis=1)
    preds = np.argmax(probs, axis=1)
    accs = (preds == labels).astype(float)
    ece = 0.0
    for lo_b, hi_b in zip(boundaries[:-1], boundaries[1:]):
        mask = (conf > lo_b) & (conf <= hi_b)
        if mask.sum() > 0:
            ece += (mask.sum() / len(labels)) * abs(accs[mask].mean() - conf[mask].mean())
    return float(ece)


def brier_score(probs: np.ndarray, labels: np.ndarray) -> float:
    """
    Multi-class Brier score:
    Mean squared error between predicted class probability distribution and one-hot true indicator vector.
    """
    onehot = np.eye(probs.shape[1])[labels]
    return float(np.mean(np.sum((probs - onehot) ** 2, axis=1)))


def calculate_metrics(probs: np.ndarray, labels: np.ndarray) -> Dict[str, Any]:
    """
    Compute full diagnostic metric suite for binary classification:
    - AUROC (calculated strictly on malignant class probability column probs[:, 1])
    - Sensitivity (Recall)
    - Specificity
    - F1-score
    - Expected Calibration Error (ECE)
    - Brier score
    - Sample size (N)
    """
    preds = np.argmax(probs, axis=1)
    cm = confusion_matrix(labels, preds, labels=[0, 1])
    tn, fp, fn, tp = cm.ravel()

    auroc = (
        float(roc_auc_score(labels, probs[:, 1]))
        if len(np.unique(labels)) > 1 else float('nan')
    )

    return {
        "AUROC": auroc,
        "Sensitivity": float(tp / (tp + fn)) if (tp + fn) > 0 else 0.0,
        "Specificity": float(tn / (tn + fp)) if (tn + fp) > 0 else 0.0,
        "F1": float(f1_score(labels, preds, zero_division=0)),
        "ECE": expected_calibration_error(probs, labels),
        "Brier": brier_score(probs, labels),
        "N": int(len(labels)),
    }
