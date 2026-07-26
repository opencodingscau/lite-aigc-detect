#!/usr/bin/env python3
"""Render study-specific SSM architecture overview figure for the paper."""
from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch

ROOT = Path(__file__).resolve().parents[1]
OUTS = [
    ROOT / "docs" / "fig_ssm_architecture.png",
    ROOT / "latex" / "figures" / "fig_ssm_architecture.png",
    ROOT / "freeze" / "figures" / "fig_ssm_architecture.png",
]


def box(ax, xy, w, h, text, fc="#e8eef5", ec="#1f4e79"):
    x, y = xy
    p = FancyBboxPatch(
        (x, y),
        w,
        h,
        boxstyle="round,pad=0.02,rounding_size=0.02",
        linewidth=1.2,
        edgecolor=ec,
        facecolor=fc,
    )
    ax.add_patch(p)
    ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", fontsize=8.5, color="#111")


def arrow(ax, x1, y1, x2, y2):
    ax.annotate(
        "",
        xy=(x2, y2),
        xytext=(x1, y1),
        arrowprops=dict(arrowstyle="->", color="#333", lw=1.2),
    )


def main():
    fig, axes = plt.subplots(1, 2, figsize=(11.2, 5.6))
    for ax, title, block_label, note in [
        (
            axes[0],
            "LiteSSM-A",
            "×4 MRFFILite\nLN → DWConv{3,5,7}+SSM\nconcat → Linear + residual",
            "d_state=8 · expand=1 · sequential PyTorch scan",
        ),
        (
            axes[1],
            "LiteSSM-B",
            "×4 BiViMBlock\nLN → SSM_fwd ∥ SSM_bwd\nconcat → Linear + residual",
            "d_state=16 · bidirectional SelectiveSSM · sequential PyTorch scan",
        ),
    ]:
        ax.set_xlim(0, 10)
        ax.set_ylim(0, 12)
        ax.axis("off")
        ax.set_title(title, fontsize=11, pad=8)

        levels = [
            (10.6, "Input\n224×224×3"),
            (9.2, "Stem (4× Conv3×3/s2, BN, GELU)\n→ 14×14×192"),
            (7.6, "Flatten + pos emb\nL=196, C=192"),
            (5.5, block_label),
            (3.4, "LayerNorm + mean pool"),
            (1.8, "Linear → 2 logits"),
        ]
        w, h = 7.2, 1.1
        x = 1.4
        for i, (y, text) in enumerate(levels):
            fc = "#dfeaf7" if i not in (3,) else "#f7e8d4"
            box(ax, (x, y), w, h, text, fc=fc)
            if i < len(levels) - 1:
                y_next = levels[i + 1][0]
                arrow(ax, 5, y, 5, y_next + h)
        ax.text(5, 0.55, note, ha="center", va="center", fontsize=7.5, color="#444")

    fig.suptitle(
        "Study-specific pure-PyTorch SSM classifiers (LiteSSM-A / LiteSSM-B)",
        fontsize=11,
        y=0.98,
    )
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    for out in OUTS:
        out.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(out, dpi=220, bbox_inches="tight")
        print("wrote", out)
    plt.close(fig)


if __name__ == "__main__":
    main()
