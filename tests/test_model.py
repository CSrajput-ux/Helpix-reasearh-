import pytest
import torch
from src.models.se_attention import SEBlock
from src.models.backbones import TVEfficientNetFeatureExtractor
from src.models.helpix_r import HELPixR
from src.training.losses import FocalLoss
from src.calibration.temperature_scaling import TemperatureScaler


def test_se_block():
    se = SEBlock(channels=24, reduction=16)
    x = torch.randn(2, 24, 14, 14)
    out = se(x)
    assert out.shape == x.shape


def test_feature_extractor():
    fe = TVEfficientNetFeatureExtractor(pretrained=False)
    x = torch.randn(1, 3, 224, 224)
    feats = fe(x)
    assert len(feats) == 5
    assert feats[0].shape[1] == 16
    assert feats[1].shape[1] == 24
    assert feats[2].shape[1] == 40
    assert feats[3].shape[1] == 112
    assert feats[4].shape[1] == 320


def test_helpix_r_forward():
    model = HELPixR(num_classes=2, dropout_p=0.3)
    x = torch.randn(2, 3, 224, 224)
    out = model(x)
    assert out.shape == (2, 2)


def test_mc_dropout():
    model = HELPixR(num_classes=2, dropout_p=0.3)
    x = torch.randn(2, 3, 224, 224)
    mean_probs, uncertainty = model.mc_dropout_predict(x, n_samples=5)
    assert mean_probs.shape == (2, 2)
    assert uncertainty.shape == (2,)
    # Probabilities sum to 1
    assert torch.allclose(mean_probs.sum(dim=1), torch.ones(2), atol=1e-4)


def test_focal_loss():
    loss_fn = FocalLoss(alpha=1.0, gamma=2.0)
    logits = torch.tensor([[2.0, -1.0], [-1.0, 2.0]], dtype=torch.float32)
    targets = torch.tensor([0, 1], dtype=torch.long)
    loss = loss_fn(logits, targets)
    assert loss.item() > 0.0
    assert torch.isfinite(loss)


def test_temperature_scaler():
    scaler = TemperatureScaler(initial_temperature=1.5)
    logits = torch.tensor([[2.0, -2.0]], dtype=torch.float32)
    scaled = scaler(logits)
    assert torch.allclose(scaled, logits / 1.5)
