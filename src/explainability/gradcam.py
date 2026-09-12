from typing import Optional, List
import numpy as np
import torch
import torch.nn as nn
from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
from pytorch_grad_cam.utils.image import show_cam_on_image


def get_gradcam_explainer(model: nn.Module) -> GradCAM:
    """
    Instantiate Grad-CAM on the penal-ultimate convolutional layer of the backbone.
    Target layer: model.backbone.features[-2]
    """
    if hasattr(model, 'backbone') and hasattr(model.backbone, 'features'):
        target_layers = [model.backbone.features[-2]]
    else:
        target_layers = [list(model.children())[-2]]
    return GradCAM(model=model, target_layers=target_layers)


def generate_cam(
    cam: GradCAM,
    input_tensor: torch.Tensor,
    target_class: Optional[int] = None
) -> np.ndarray:
    """
    Compute 2D Grad-CAM attribution heatmap for a given input tensor.
    Attribution map reflects gradient-weighted activation sensitivity.
    """
    targets = [ClassifierOutputTarget(target_class)] if target_class is not None else None
    grayscale_cam = cam(input_tensor=input_tensor, targets=targets)[0]
    return grayscale_cam
