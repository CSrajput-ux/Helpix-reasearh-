from typing import Tuple
import numpy as np
from sklearn.metrics import roc_auc_score


def bootstrap_auroc(
    probs_1d: np.ndarray,
    labels: np.ndarray,
    n_bootstrap: int = 1000,
    seed: int = 42
) -> Tuple[float, float, float]:
    """
    Non-parametric bootstrap estimation of 95% Confidence Interval for AUROC.
    probs_1d: 1-D array of predicted malignant-class probabilities (probs[:, 1]).
    labels: binary ground truth labels.
    """
    rng = np.random.RandomState(seed)
    scores = []
    n = len(labels)

    for _ in range(n_bootstrap):
        idx = rng.randint(0, n, n)
        if len(np.unique(labels[idx])) < 2:
            continue
        scores.append(roc_auc_score(labels[idx], probs_1d[idx]))

    if len(scores) == 0:
        return float('nan'), float('nan'), float('nan')

    mean_score = float(np.mean(scores))
    ci_lower = float(np.percentile(scores, 2.5))
    ci_upper = float(np.percentile(scores, 97.5))

    return mean_score, ci_lower, ci_upper
