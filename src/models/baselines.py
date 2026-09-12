import torch.nn as nn
import torchvision.models as tv_models
import timm


def create_baseline_model(model_name: str, num_classes: int = 2, pretrained: bool = True) -> nn.Module:
    """
    Factory function for baseline comparison models matching the experimental protocol.
    Available architectures:
      - resnet50
      - efficientnet_b0
      - mobilenetv3 (MobileNetV3-Large)
      - convnext_tiny
    """
    b_name = model_name.lower().strip()
    try:
        if b_name == 'resnet50':
            weights = tv_models.ResNet50_Weights.DEFAULT if pretrained else None
            m = tv_models.resnet50(weights=weights)
            m.fc = nn.Linear(m.fc.in_features, num_classes)
            return m
        elif b_name in ('efficientnet_b0', 'efficientnet'):
            weights = tv_models.EfficientNet_B0_Weights.DEFAULT if pretrained else None
            m = tv_models.efficientnet_b0(weights=weights)
            m.classifier[1] = nn.Linear(m.classifier[1].in_features, num_classes)
            return m
        elif b_name in ('mobilenetv3', 'mobilenet_v3_large'):
            weights = tv_models.MobileNet_V3_Large_Weights.DEFAULT if pretrained else None
            m = tv_models.mobilenet_v3_large(weights=weights)
            m.classifier[3] = nn.Linear(m.classifier[3].in_features, num_classes)
            return m
        elif b_name in ('convnext_tiny', 'convnext'):
            weights = tv_models.ConvNeXt_Tiny_Weights.DEFAULT if pretrained else None
            m = tv_models.convnext_tiny(weights=weights)
            m.classifier[2] = nn.Linear(m.classifier[2].in_features, num_classes)
            return m
        else:
            return timm.create_model(model_name, pretrained=pretrained, num_classes=num_classes)
    except Exception:
        return timm.create_model(model_name, pretrained=pretrained, num_classes=num_classes)
