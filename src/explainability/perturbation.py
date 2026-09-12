"""
Perturbation / Deletion Test for Attribution Sensitivity
=========================================================
NOTE ON SCIENTIFIC INTERPRETATION:
The quantitative deletion analysis evaluates attribution sensitivity by masking the
highest-salience regions and measuring the resulting model output confidence variation.
This procedure does NOT establish causal biological validity or clinically reasoned judgment.
"""

from typing import Tuple
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
from .gradcam import get_gradcam_explainer


def run_deletion_test(
    model: nn.Module,
    dataset: Dataset,
    n_samples: int = 30,
    mask_percentile: float = 60.0,
    device: torch.device | None = None
) -> Tuple[float, float]:
    """
    Evaluate attribution sensitivity via quantitative deletion test:
      1. Compute Grad-CAM attribution map.
      2. Mask top-40% salient pixels (pixels above 60th percentile) to zero.
      3. Measure class probability drop (orig_p - masked_p).
    Returns (mean_confidence_drop, median_confidence_drop).
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model.eval()
    cam = get_gradcam_explainer(model)
    drops = []

    for i in range(min(n_samples, len(dataset))):
        img_t, label = dataset[i]
        inp = img_t.unsqueeze(0).to(device)
        targets = [ClassifierOutputTarget(label)]

        try:
            grayscale_cam = cam(input_tensor=inp, targets=targets)[0]
        except Exception:
            grayscale_cam = cam(input_tensor=inp, targets=None)[0]

        with torch.no_grad():
            orig_p = F.softmax(model(inp), dim=1)[0, label].item()

        # Mask top-40% salient pixels (above 60th percentile)
        thresh = np.percentile(grayscale_cam, mask_percentile)
        mask = torch.from_numpy((grayscale_cam > thresh).astype('float32')).unsqueeze(0).repeat(3, 1, 1).to(device)
        masked_inp = (img_t.to(device) * (1.0 - mask)).unsqueeze(0)

        with torch.no_grad():
            masked_p = F.softmax(model(masked_inp), dim=1)[0, label].item()

        # Mathematical drop (positive value signifies confidence reduction upon masking salient region)
        drops.append(orig_p - masked_p)

    del cam
    if device.type == 'cuda':
        torch.cuda.empty_cache()

    mean_drop = float(np.mean(drops))
    median_drop = float(np.median(drops))
    return mean_drop, median_drop
