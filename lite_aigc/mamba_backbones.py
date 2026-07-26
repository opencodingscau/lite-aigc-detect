#!/usr/bin/env python3
"""Pure-PyTorch Mamba-style backbones for bake-off (no mamba_ssm required).

- MobileMambaLite: multi-receptive DWConv + selective SSM (MobileMamba-inspired)
- MambaPSACls: patch embed + BiViM (forward/backward SSM), MambaPSA-inspired

Honest labeling: these are protocol-faithful reimplementations for controlled bake-off,
not bit-exact official MobileMamba / YOLO-MambaPSA weights.
"""
from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


class SelectiveSSM(nn.Module):
    """Lightweight selective state-space scan (sequential, pure PyTorch)."""

    def __init__(self, d_model: int, d_state: int = 16, expand: int = 2):
        super().__init__()
        self.d_model = d_model
        self.d_state = d_state
        self.d_inner = d_model * expand
        self.in_proj = nn.Linear(d_model, self.d_inner * 2, bias=False)
        self.conv1d = nn.Conv1d(
            self.d_inner, self.d_inner, kernel_size=3, padding=1, groups=self.d_inner
        )
        self.x_proj = nn.Linear(self.d_inner, d_state * 2 + self.d_inner, bias=False)
        self.dt_proj = nn.Linear(self.d_inner, self.d_inner, bias=True)
        self.A_log = nn.Parameter(torch.log(torch.arange(1, d_state + 1, dtype=torch.float32)).repeat(self.d_inner, 1))
        self.D = nn.Parameter(torch.ones(self.d_inner))
        self.out_proj = nn.Linear(self.d_inner, d_model, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: B, L, D
        b, l, _ = x.shape
        xz = self.in_proj(x)
        x_in, z = xz.chunk(2, dim=-1)
        x_in = self.conv1d(x_in.transpose(1, 2)).transpose(1, 2)
        x_in = F.silu(x_in)
        x_dbl = self.x_proj(x_in)
        delta, B, C = torch.split(x_dbl, [self.d_inner, self.d_state, self.d_state], dim=-1)
        delta = F.softplus(self.dt_proj(delta))  # B L d_inner
        A = -torch.exp(self.A_log.float())  # d_inner, d_state

        # sequential scan
        h = torch.zeros(b, self.d_inner, self.d_state, device=x.device, dtype=x_in.dtype)
        ys = []
        for t in range(l):
            dt = delta[:, t].unsqueeze(-1)  # B d_inner 1
            Bt = B[:, t].unsqueeze(1)  # B 1 d_state
            dA = torch.exp(dt * A)  # B d_inner d_state
            dB = dt * Bt  # B d_inner d_state
            h = dA * h + dB * x_in[:, t].unsqueeze(-1)
            y = torch.einsum("bdn,bn->bd", h, C[:, t]) + self.D * x_in[:, t]
            ys.append(y)
        y = torch.stack(ys, dim=1)
        y = y * F.silu(z)
        return self.out_proj(y)


class BiViMBlock(nn.Module):
    """Bidirectional vision Mamba block (MambaPSA BiViM spirit)."""

    def __init__(self, dim: int, d_state: int = 16):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.fwd = SelectiveSSM(dim, d_state=d_state, expand=1)
        self.bwd = SelectiveSSM(dim, d_state=d_state, expand=1)
        self.mix = nn.Linear(dim * 2, dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.norm(x)
        yf = self.fwd(h)
        yb = self.bwd(torch.flip(h, dims=[1]))
        yb = torch.flip(yb, dims=[1])
        return x + self.mix(torch.cat([yf, yb], dim=-1))


class MRFFILite(nn.Module):
    """MobileMamba-inspired multi-receptive + SSM interaction (channel-last tokens)."""

    def __init__(self, dim: int):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.dw3 = nn.Conv1d(dim, dim, 3, padding=1, groups=dim)
        self.dw5 = nn.Conv1d(dim, dim, 5, padding=2, groups=dim)
        self.dw7 = nn.Conv1d(dim, dim, 7, padding=3, groups=dim)
        self.ssm = SelectiveSSM(dim, d_state=8, expand=1)
        self.proj = nn.Linear(dim * 2, dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: B L C
        h = self.norm(x)
        t = h.transpose(1, 2)
        local = self.dw3(t) + self.dw5(t) + self.dw7(t)
        local = local.transpose(1, 2)
        global_ = self.ssm(h)
        return x + self.proj(torch.cat([local, global_], dim=-1))


class MambaPSACls(nn.Module):
    """Classification backbone adapted from MambaPSA BiViM idea."""

    def __init__(self, img_size=224, embed_dim=192, depth=4, num_classes=2):
        super().__init__()
        # 224 -> 14 (four stride-2) so SSM seq len = 196
        self.patch = nn.Sequential(
            nn.Conv2d(3, embed_dim // 2, 3, stride=2, padding=1),
            nn.BatchNorm2d(embed_dim // 2),
            nn.GELU(),
            nn.Conv2d(embed_dim // 2, embed_dim // 2, 3, stride=2, padding=1),
            nn.BatchNorm2d(embed_dim // 2),
            nn.GELU(),
            nn.Conv2d(embed_dim // 2, embed_dim, 3, stride=2, padding=1),
            nn.BatchNorm2d(embed_dim),
            nn.GELU(),
            nn.Conv2d(embed_dim, embed_dim, 3, stride=2, padding=1),
            nn.BatchNorm2d(embed_dim),
            nn.GELU(),
        )
        n = (img_size // 16) ** 2
        self.pos = nn.Parameter(torch.zeros(1, n, embed_dim))
        self.blocks = nn.ModuleList([BiViMBlock(embed_dim) for _ in range(depth)])
        self.norm = nn.LayerNorm(embed_dim)
        self.head = nn.Linear(embed_dim, num_classes)
        nn.init.trunc_normal_(self.pos, std=0.02)

    def forward_features(self, x: torch.Tensor) -> torch.Tensor:
        x = self.patch(x)
        x = x.flatten(2).transpose(1, 2)
        if x.size(1) != self.pos.size(1):
            pos = F.interpolate(
                self.pos.transpose(1, 2), size=x.size(1), mode="linear", align_corners=False
            ).transpose(1, 2)
        else:
            pos = self.pos
        x = x + pos
        for blk in self.blocks:
            x = blk(x)
        return self.norm(x).mean(dim=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.head(self.forward_features(x))


class MobileMambaLite(nn.Module):
    """Lightweight MobileMamba-inspired classifier (SSM only at 14x14)."""

    def __init__(self, img_size=224, dim=192, depth=4, num_classes=2):
        super().__init__()
        self.stem = nn.Sequential(
            nn.Conv2d(3, 64, 3, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.GELU(),
            nn.Conv2d(64, 128, 3, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.GELU(),
            nn.Conv2d(128, dim, 3, stride=2, padding=1),
            nn.BatchNorm2d(dim),
            nn.GELU(),
            nn.Conv2d(dim, dim, 3, stride=2, padding=1),
            nn.BatchNorm2d(dim),
            nn.GELU(),
        )
        n = (img_size // 16) ** 2
        self.pos = nn.Parameter(torch.zeros(1, n, dim))
        self.blocks = nn.ModuleList([MRFFILite(dim) for _ in range(depth)])
        self.norm = nn.LayerNorm(dim)
        self.head = nn.Linear(dim, num_classes)
        nn.init.trunc_normal_(self.pos, std=0.02)

    def forward_features(self, x: torch.Tensor) -> torch.Tensor:
        x = self.stem(x)
        x = x.flatten(2).transpose(1, 2)
        if x.size(1) != self.pos.size(1):
            pos = F.interpolate(
                self.pos.transpose(1, 2), size=x.size(1), mode="linear", align_corners=False
            ).transpose(1, 2)
        else:
            pos = self.pos
        x = x + pos
        for blk in self.blocks:
            x = blk(x)
        return self.norm(x).mean(dim=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.head(self.forward_features(x))


class DualBranchDetector(nn.Module):
    """Wrap spatial backbone (+ optional gated freq). FreqBranch injected by caller."""

    def __init__(
        self,
        backbone: nn.Module,
        emb: int,
        freq_branch: nn.Module | None,
        num_classes: int = 2,
    ):
        super().__init__()
        self.backbone = backbone
        self.freq = freq_branch
        self.use_freq = freq_branch is not None
        if self.use_freq:
            self.fuse_alpha = nn.Parameter(torch.tensor(-1.0))
            self.fuse_norm = nn.LayerNorm(emb)
        self.head = nn.Sequential(
            nn.Linear(emb, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(0.2),
            nn.Linear(128, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if hasattr(self.backbone, "forward_features"):
            s = self.backbone.forward_features(x)
        else:
            s = self.backbone(x)
        if self.use_freq and self.freq is not None:
            f = self.freq(x)
            gate = torch.sigmoid(self.fuse_alpha)
            z = self.fuse_norm(s + gate * f)
        else:
            z = s
        return self.head(z)


def build_mamba_backbone(name: str, num_classes: int = 2) -> nn.Module:
    name = name.lower()
    if name == "mambapsa_cls":
        return MambaPSACls(embed_dim=192, depth=4, num_classes=num_classes)
    if name == "mobilemamba_lite":
        return MobileMambaLite(num_classes=num_classes)
    raise ValueError(name)
