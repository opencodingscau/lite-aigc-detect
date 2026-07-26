"""Models: baselines + LiteFreqNet (lightweight CNN + frequency branch)."""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models


def build_model(name: str, num_classes: int = 2) -> nn.Module:
    name = name.lower()
    if name == "efficientnet_b0":
        m = models.efficientnet_b0(weights=None)
        m.classifier[1] = nn.Linear(m.classifier[1].in_features, num_classes)
        return m
    if name == "mobilenet_v3_small":
        m = models.mobilenet_v3_small(weights=None)
        m.classifier[-1] = nn.Linear(m.classifier[-1].in_features, num_classes)
        return m
    if name in ("shufflenet_v2_x0_5", "shufflenetv2_x0_5"):
        m = models.shufflenet_v2_x0_5(weights=None)
        m.fc = nn.Linear(m.fc.in_features, num_classes)
        return m
    if name == "lite_freq_net":
        return LiteFreqNet(num_classes=num_classes, fusion="concat")
    if name == "lite_freq_net_nofreq":
        return LiteFreqNet(num_classes=num_classes, use_freq=False)
    if name in ("lite_freq_net_v2", "lite_freq_net_gated"):
        # v2: mid-band prior + gated residual fusion (not naive concat)
        return LiteFreqNet(num_classes=num_classes, fusion="gated_add", mid_prior=True)
    if name in (
        "mambapsa_cls",
        "mambapsa_cls_freq",
        "mobilemamba_lite",
        "mobilemamba_lite_freq",
    ):
        from mamba_backbones import DualBranchDetector, MambaPSACls, MobileMambaLite

        if name == "mambapsa_cls":
            return MambaPSACls(embed_dim=192, depth=4, num_classes=num_classes)
        if name == "mobilemamba_lite":
            return MobileMambaLite(num_classes=num_classes)
        if name == "mambapsa_cls_freq":
            bb = MambaPSACls(embed_dim=192, depth=4, num_classes=num_classes)
            return DualBranchDetector(
                bb, emb=192, freq_branch=FreqBranch(out_dim=192, mid_prior=True), num_classes=num_classes
            )
        if name == "mobilemamba_lite_freq":
            bb = MobileMambaLite(num_classes=num_classes)
            return DualBranchDetector(
                bb, emb=192, freq_branch=FreqBranch(out_dim=192, mid_prior=True), num_classes=num_classes
            )
    raise ValueError(f"Unknown model: {name}")


class FreqBranch(nn.Module):
    """DCT-like mid/high band energy via FFT magnitude + small CNN."""

    def __init__(self, out_dim: int = 128, mid_prior: bool = False):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(1, 32, 3, stride=2, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 64, 3, stride=2, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 128, 3, stride=2, padding=1),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d(1),
        )
        self.fc = nn.Linear(128, out_dim)
        # learnable radial band gate (low/mid/high); mid_prior biases toward mid-band
        if mid_prior:
            self.band_logits = nn.Parameter(torch.tensor([-2.0, 2.0, -0.5]))
        else:
            self.band_logits = nn.Parameter(torch.tensor([0.0, 1.0, 0.5]))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: B,3,H,W in normalized RGB -> gray
        gray = (0.2989 * x[:, 0] + 0.5870 * x[:, 1] + 0.1140 * x[:, 2]).unsqueeze(1)
        # FFT magnitude
        spec = torch.fft.fftshift(torch.fft.fft2(gray), dim=(-2, -1))
        mag = torch.log1p(spec.abs())
        b, _, h, w = mag.shape
        yy = torch.linspace(-1, 1, h, device=mag.device).view(1, 1, h, 1)
        xx = torch.linspace(-1, 1, w, device=mag.device).view(1, 1, 1, w)
        r = torch.sqrt(xx * xx + yy * yy)
        gates = torch.softmax(self.band_logits, dim=0)
        low = (r < 0.3).float()
        mid = ((r >= 0.3) & (r < 0.7)).float()
        high = (r >= 0.7).float()
        mag = mag * (gates[0] * low + gates[1] * mid + gates[2] * high)
        feat = self.net(mag).flatten(1)
        return self.fc(feat)


class SpatialBranch(nn.Module):
    def __init__(self, out_dim: int = 128):
        super().__init__()
        backbone = models.mobilenet_v3_small(weights=None)
        self.features = backbone.features
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(576, out_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        f = self.features(x)
        f = self.pool(f).flatten(1)
        return self.fc(f)


class LiteFreqNet(nn.Module):
    """Lightweight dual-branch detector (spatial MobileNet-small + freq FFT).

    fusion:
      - concat: original v1 (often hurts)
      - gated_add: z = LN(s + sigmoid(alpha) * f)  then MLP head
    """

    def __init__(
        self,
        num_classes: int = 2,
        emb: int = 128,
        use_freq: bool = True,
        fusion: str = "concat",
        mid_prior: bool = False,
    ):
        super().__init__()
        self.use_freq = use_freq
        self.fusion = fusion if use_freq else "none"
        self.spatial = SpatialBranch(out_dim=emb)
        self.freq = FreqBranch(out_dim=emb, mid_prior=mid_prior) if use_freq else None
        if use_freq and self.fusion == "gated_add":
            self.fuse_alpha = nn.Parameter(torch.tensor(-1.0))  # sigmoid~0.27 start conservative
            self.fuse_norm = nn.LayerNorm(emb)
            in_dim = emb
        elif use_freq:
            in_dim = emb * 2
            self.fuse_alpha = None
            self.fuse_norm = None
        else:
            in_dim = emb
            self.fuse_alpha = None
            self.fuse_norm = None
        self.head = nn.Sequential(
            nn.Linear(in_dim, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(0.2),
            nn.Linear(128, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        s = self.spatial(x)
        if self.use_freq and self.freq is not None:
            f = self.freq(x)
            if self.fusion == "gated_add":
                gate = torch.sigmoid(self.fuse_alpha)
                z = self.fuse_norm(s + gate * f)
            else:
                z = torch.cat([s, f], dim=1)
        else:
            z = s
        return self.head(z)


def count_params(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
