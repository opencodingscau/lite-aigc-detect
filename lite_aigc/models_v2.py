"""V2 compact backbones for Pilot A (does not change paper model names)."""
from __future__ import annotations

import torch.nn as nn
from torchvision import models


def _timm_classifier(name: str, num_classes: int = 2) -> nn.Module:
    import timm

    return timm.create_model(name, pretrained=False, num_classes=num_classes)


def build_v2_model(name: str, num_classes: int = 2) -> nn.Module:
    """Pilot-A / v2-only architectures."""
    key = name.lower().replace("-", "_")

    timm_aliases = {
        "repvit": "repvit_m0_9",
        "repvit_m0_9": "repvit_m0_9",
        "repvit_m1_0": "repvit_m1_0",
        "shvit": "shvit_s4",
        "shvit_s4": "shvit_s4",
        "efficientvit": "efficientvit_b0",
        "efficientvit_b0": "efficientvit_b0",
    }
    if key in timm_aliases:
        tname = timm_aliases[key]
        try:
            return _timm_classifier(tname, num_classes=num_classes)
        except Exception as e:  # noqa: BLE001
            raise ImportError(
                f"timm model '{tname}' unavailable ({e}). pip install -U timm"
            ) from e

    if key in ("convnext_tiny", "mambaout_proxy"):
        # Wave-1 proxy for gated-CNN / MambaOut-style capacity until vendor code is added.
        m = models.convnext_tiny(weights=None)
        m.classifier[2] = nn.Linear(m.classifier[2].in_features, num_classes)
        return m
    if key in ("efficientnet_v2_s", "efficientnetv2_s"):
        m = models.efficientnet_v2_s(weights=None)
        m.classifier[1] = nn.Linear(m.classifier[1].in_features, num_classes)
        return m

    if key in ("efficientvim", "efficient_vim"):
        raise ImportError("EfficientViM not vendored yet under v2/vendor/.")
    if key in ("efficientvmamba", "efficient_vmamba"):
        raise ImportError("EfficientVMamba not vendored yet under v2/vendor/.")
    if key == "mambaout":
        raise ImportError("Use mambaout_proxy for wave-1, or vendor official MambaOut.")

    raise ValueError(f"Unknown v2 model: {name}")
