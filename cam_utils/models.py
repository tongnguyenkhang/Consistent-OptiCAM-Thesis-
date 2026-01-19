import torch
import torch.nn as nn
from util import Preprocessing_Layer

try:
    from torchvision.models import (
        resnet50, ResNet50_Weights,
        resnet34, ResNet34_Weights,
        resnet18, ResNet18_Weights,
    )
    _HAS_NEW_TORCHVISION = True
except Exception:
    from torchvision import models as _legacy_models
    _HAS_NEW_TORCHVISION = False

def count_parameters(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def get_backbone(name: str = "resnet50", pretrained: bool = True) -> nn.Module:
    name = name.lower()
    if name in ("resnet50", "r50", "res50"):
        if _HAS_NEW_TORCHVISION:
            weights = ResNet50_Weights.IMAGENET1K_V1 if pretrained else None
            return resnet50(weights=weights)
        return _legacy_models.resnet50(pretrained=pretrained)
    if name in ("resnet34", "r34", "res34"):
        if _HAS_NEW_TORCHVISION:
            weights = ResNet34_Weights.IMAGENET1K_V1 if pretrained else None
            return resnet34(weights=weights)
        return _legacy_models.resnet34(pretrained=pretrained)
    if name in ("resnet18", "r18", "res18"):
        if _HAS_NEW_TORCHVISION:
            weights = ResNet18_Weights.IMAGENET1K_V1 if pretrained else None
            return resnet18(weights=weights)
        return _legacy_models.resnet18(pretrained=pretrained)
    raise ValueError(f"Unsupported backbone: {name}")

def get_preprocess(mean=None, std=None) -> Preprocessing_Layer:
    mean = mean or [0.485, 0.456, 0.406]
    std  = std  or [0.229, 0.224, 0.225]
    return Preprocessing_Layer(mean, std)

def build_model_seq(backbone: nn.Module,
                    device: torch.device,
                    mean=None, std=None,
                    eval_mode: bool = True) -> nn.Sequential:
    preprocess = get_preprocess(mean, std)
    seq = nn.Sequential(preprocess, backbone).to(device)
    return seq.eval() if eval_mode else seq.train()