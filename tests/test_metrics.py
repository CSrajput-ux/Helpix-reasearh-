import pytest
import numpy as np
from src.evaluation.metrics import expected_calibration_error, brier_score, calculate_metrics
from src.evaluation.bootstrap import bootstrap_auroc


def test_ece_perfect_calibration():
    # Construct synthetic data where confidence perfectly matches accuracy
    probs = np.array([
        [0.9, 0.1],  # pred 0, conf 0.9
        [0.9, 0.1],  # pred 0, conf 0.9
        [0.1, 0.9],  # pred 1, conf 0.9
        [0.1, 0.9],  # pred 1, conf 0.9
    ])
    labels = np.array([0, 0, 1, 1])
    ece = expected_calibration_error(probs, labels, n_bins=10)
    assert ece >= 0.0
    assert ece <= 0.2


def test_brier_score():
    probs = np.array([[1.0, 0.0], [0.0, 1.0]])
    labels = np.array([0, 1])
    # Perfect score is 0.0
    assert brier_score(probs, labels) == 0.0

    # Inverted predictions
    probs_bad = np.array([[0.0, 1.0], [1.0, 0.0]])
    assert brier_score(probs_bad, labels) == 2.0


def test_calculate_metrics():
    probs = np.array([
        [0.9, 0.1],
        [0.8, 0.2],
        [0.2, 0.8],
        [0.1, 0.9]
    ])
    labels = np.array([0, 0, 1, 1])
    metrics = calculate_metrics(probs, labels)
    assert metrics["AUROC"] == 1.0
    assert metrics["Sensitivity"] == 1.0
    assert metrics["Specificity"] == 1.0
    assert metrics["F1"] == 1.0
    assert metrics["N"] == 4


def test_bootstrap_auroc():
    probs_1d = np.array([0.1, 0.2, 0.3, 0.7, 0.8, 0.9])
    labels = np.array([0, 0, 0, 1, 1, 1])
    mean_auc, lo, hi = bootstrap_auroc(probs_1d, labels, n_bootstrap=100, seed=42)
    assert 0.0 <= lo <= mean_auc <= hi <= 1.0
