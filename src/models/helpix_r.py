from typing import Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F
from .se_attention import SEBlock
from .backbones import TVEfficientNetFeatureExtractor


class HELPixR(nn.Module):
    """
    HELPix-R: Multi-scale feature fusion with per-scale Squeeze-and-Excitation channel attention.
    
    Architecture:
      1. EfficientNet-B0 backbone extracting 5 intermediate scales (C1-C5: 16, 24, 40, 112, 320 channels).
      2. Independent SEBlock attention applied to each intermediate scale.
      3. Adaptive average pooling to 1x1 spatial resolution per scale.
      4. Flattening and concatenation across all 5 scales -> 512-dimensional unified representation.
      5. Classification head: Dropout(p) -> Linear(512, 256) -> ReLU -> Dropout(p) -> Linear(256, num_classes).
    """
    def __init__(self, num_classes: int = 2, backbone_name: str = 'efficientnet_b0', dropout_p: float = 0.3):
        super().__init__()
        self.backbone = TVEfficientNetFeatureExtractor()
        feat_channels = self.backbone.channels

        self.se_blocks = nn.ModuleList([SEBlock(c) for c in feat_channels])
        self.pools = nn.ModuleList([nn.AdaptiveAvgPool2d(1) for _ in feat_channels])

        fusion_dim = sum(feat_channels)  # 16 + 24 + 40 + 112 + 320 = 512
        self.dropout = nn.Dropout(dropout_p)
        self.classifier = nn.Sequential(
            nn.Linear(fusion_dim, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout_p),
            nn.Linear(256, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        feats = self.backbone(x)
        pooled = [
            pool(se(f)).flatten(1)
            for f, se, pool in zip(feats, self.se_blocks, self.pools)
        ]
        fused = torch.cat(pooled, dim=1)
        return self.classifier(self.dropout(fused))

    @torch.no_grad()
    def mc_dropout_predict(self, x: torch.Tensor, n_samples: int = 20) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Monte Carlo Dropout predictive uncertainty estimation:
        Enables Dropout layers while locking BatchNorm layers in evaluation mode.
        Computes predictive mean probability vector and standard deviation (uncertainty) across n_samples.
        """
        self.train()
        for m in self.modules():
            if isinstance(m, nn.BatchNorm2d):
                m.eval()
        probs_list = []
        for _ in range(n_samples):
            logits = self.forward(x)
            probs_list.append(F.softmax(logits, dim=1).unsqueeze(0))
        probs_stack = torch.cat(probs_list, dim=0)  # (n_samples, B, C)
        mean_probs = probs_stack.mean(dim=0)
        uncertainty = probs_stack.std(dim=0).mean(dim=1)
        self.eval()
        return mean_probs.cpu(), uncertainty.cpu()
