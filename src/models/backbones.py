from typing import List
import torch
import torch.nn as nn
import torchvision.models as tv_models


class TVEfficientNetFeatureExtractor(nn.Module):
    """
    Multi-scale intermediate feature extractor based on torchvision's EfficientNet-B0.
    Extracts feature representations across five distinct receptive scales:
      - Stage 1 (C1): index 1, 112x112, 16 channels
      - Stage 2 (C2): index 2, 56x56,   24 channels
      - Stage 3 (C3): index 3, 28x28,   40 channels
      - Stage 4 (C4): index 5, 14x14,   112 channels
      - Stage 5 (C5): index 7, 7x7,     320 channels
    Total concatenated feature channels = 512.
    """
    def __init__(self, pretrained: bool = True):
        super().__init__()
        weights = tv_models.EfficientNet_B0_Weights.DEFAULT if pretrained else None
        eff = tv_models.efficientnet_b0(weights=weights)
        self.features = eff.features
        self.stage_indices = [1, 2, 3, 5, 7]
        self.channels = [16, 24, 40, 112, 320]

    def forward(self, x: torch.Tensor) -> List[torch.Tensor]:
        outputs = []
        for i, layer in enumerate(self.features):
            x = layer(x)
            if i in self.stage_indices:
                outputs.append(x)
        return outputs
