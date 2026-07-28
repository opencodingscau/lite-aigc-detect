#!/usr/bin/env python3
"""Render the full paper figure set with a unified visual language.

Outputs to latex/figures/ and freeze/figures/.
All numeric values are read from freeze / jpeg_results / docs (no invented science).
"""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.lines import Line2D
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch, Rectangle

ROOT = Path(__file__).resolve().parents[1]
OUT_DIRS = [ROOT / "latex" / "figures", ROOT / "freeze" / "figures"]

# Unified palette (print-safe; avoid purple/glow defaults)
C = {
    "ink": "#1C1917",
    "muted": "#57534E",
    "rule": "#D6D3D1",
    "bg": "#FAFAF9",
    "ssm": "#0F766E",
    "ssm_soft": "#CCFBF1",
    "cnn": "#1D4ED8",
    "cnn_soft": "#DBEAFE",
    "freq": "#C2410C",
    "freq_soft": "#FFEDD5",
    "ref": "#44403C",
    "accent": "#A16207",
    "accent_soft": "#FEF3C7",
    "bad": "#9F1239",
    "good": "#047857",
    "shared": "#E7E5E4",
    "shared_edge": "#78716C",
}

ORDER = [
    "mobilemamba_lite",
    "mambapsa_cls",
    "efficientnet_b0",
    "lite_freq_net_v2",
    "mobilenet_v3_small",
    "shufflenet_v2_x0_5",
]
LABELS = {
    "mobilemamba_lite": "LiteSSM-A",
    "mambapsa_cls": "LiteSSM-B",
    "efficientnet_b0": "EfficientNet-B0",
    "lite_freq_net_v2": "LiteFreqNet v2",
    "mobilenet_v3_small": "MobileNetV3-S",
    "shufflenet_v2_x0_5": "ShuffleNet-x0.5",
}
ARCH = {
    "mobilemamba_lite": "SSM",
    "mambapsa_cls": "SSM",
    "efficientnet_b0": "CNN",
    "lite_freq_net_v2": "CNN+FFT",
    "mobilenet_v3_small": "CNN",
    "shufflenet_v2_x0_5": "CNN",
}
ARCH_COLOR = {"SSM": C["ssm"], "CNN": C["cnn"], "CNN+FFT": C["freq"]}

GEN_KEYS = [
    "ufd_dalle",
    "ufd_glide_100_10",
    "ufd_glide_100_27",
    "ufd_glide_50_27",
    "ufd_guided",
    "ufd_ldm_100",
    "ufd_ldm_200",
    "ufd_ldm_200_cfg",
]
GEN_LABELS = [
    "DALL·E",
    "Glide 100/10",
    "Glide 100/27",
    "Glide 50/27",
    "Guided",
    "LDM 100",
    "LDM 200",
    "LDM 200-cfg",
]

JPEG_FILES = {
    "mobilemamba_lite": "mobilemamba_lite_jpeg.json",
    "mambapsa_cls": "mambapsa_cls_jpeg.json",
    "efficientnet_b0": "efficientnet_b0_jpeg.json",
    "lite_freq_net_v2": "lite_freq_net_v2_jpeg.json",
    "mobilenet_v3_small": "mobilenet_v3_small_jpeg.json",
    "shufflenet_v2_x0_5": "shufflenet_v2_x0_5_jpeg.json",
}

HEAT_CMAP = LinearSegmentedColormap.from_list(
    "paper_heat",
    ["#FEF3C7", "#FDE68A", "#5EEAD4", "#0F766E", "#134E4A"],
)


def style():
    mpl.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 9.5,
            "axes.titlesize": 11,
            "axes.labelsize": 9.5,
            "axes.edgecolor": C["rule"],
            "axes.labelcolor": C["ink"],
            "axes.titlecolor": C["ink"],
            "xtick.color": C["muted"],
            "ytick.color": C["muted"],
            "text.color": C["ink"],
            "figure.facecolor": "white",
            "axes.facecolor": "white",
            "savefig.facecolor": "white",
            "savefig.dpi": 300,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )


def save(fig, name: str):
    for d in OUT_DIRS:
        d.mkdir(parents=True, exist_ok=True)
        path = d / name
        fig.savefig(path, dpi=300, bbox_inches="tight", pad_inches=0.08)
        print("wrote", path)


def rounded(ax, xy, w, h, text, fc, ec, fontsize=8.2, weight="regular", lw=1.2):
    x, y = xy
    patch = FancyBboxPatch(
        (x, y),
        w,
        h,
        boxstyle="round,pad=0.012,rounding_size=0.04",
        linewidth=lw,
        edgecolor=ec,
        facecolor=fc,
        mutation_aspect=0.3,
    )
    ax.add_patch(patch)
    ax.text(
        x + w / 2,
        y + h / 2,
        text,
        ha="center",
        va="center",
        fontsize=fontsize,
        color=C["ink"],
        fontweight=weight,
        wrap=True,
    )
    return patch


def arrow(ax, p1, p2, color=None):
    color = color or C["muted"]
    ax.add_patch(
        FancyArrowPatch(
            p1,
            p2,
            arrowstyle="-|>",
            mutation_scale=10,
            lw=1.15,
            color=color,
            shrinkA=1,
            shrinkB=1,
        )
    )


def load_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


# ---------------------------------------------------------------------------
# Fig: protocol overview
# ---------------------------------------------------------------------------
def fig_protocol():
    fig, ax = plt.subplots(figsize=(11.6, 3.9))
    ax.set_xlim(0, 112)
    ax.set_ylim(0, 36)
    ax.axis("off")
    ax.set_title(
        "Locked evaluation protocol (Panel A from-scratch; Panel B inference-only)",
        loc="left",
        pad=10,
        fontsize=12,
        fontweight="semibold",
    )

    stages = [
        (2, 11, 20, 18, "Frozen data\nDF train / ID\nUFD / OOD\nno test selection", "#F5F5F5", "#333333"),
        (28, 11, 20, 18, "Shared training\n15 ep · bs64\nAdamW 1e-4\nseed 42 · no PT", "#F5F5F5", "#333333"),
        (54, 20, 20, 13, "Panel A\nCNNs · LiteFreq\nLiteSSM-A/B", "#FFFFFF", "#333333"),
        (54, 3, 20, 13, "Panel B\nUnivFD · NPR\n(pretrained refs)", "#FFFFFF", "#333333"),
        (80, 11, 28, 18, "Report\nID · domain\nUFD Macro\nJPEG Q70\nlatency / thr.", "#F5F5F5", "#333333"),
    ]
    for args in stages:
        ax.add_patch(Rectangle((args[0], args[1]), args[2], args[3], fc=args[5], ec=args[6], lw=0.9))
        ax.text(
            args[0] + args[2] / 2,
            args[1] + args[3] / 2,
            args[4],
            ha="center",
            va="center",
            fontsize=9.2,
            color=C["ink"],
            linespacing=1.28,
        )

    arrow(ax, (22.2, 20), (27.6, 20))
    arrow(ax, (48.2, 20), (53.6, 26.5))
    arrow(ax, (48.2, 20), (53.6, 9.5))
    arrow(ax, (74.2, 26.5), (79.6, 20))
    arrow(ax, (74.2, 9.5), (79.6, 20))

    ax.text(
        2,
        1.6,
        "Primary OOD claim uses UFD Macro (equal-weight per generator). OOD Pooled is supplementary. FLUX = appendix only.",
        fontsize=8.0,
        color=C["muted"],
    )
    save(fig, "fig_protocol_overview.png")
    plt.close(fig)


# ---------------------------------------------------------------------------
# Fig: architecture (shared stem + fork)
# ---------------------------------------------------------------------------
def fig_architecture():
    fig = plt.figure(figsize=(11.8, 6.4))
    ax = fig.add_axes([0.025, 0.045, 0.95, 0.90])
    ax.set_xlim(0, 100)
    ax.set_ylim(0, 64)
    ax.axis("off")
    a_edge, a_fill = "#4472C4", "#EAF0FA"
    b_edge, b_fill = "#70AD47", "#EDF5E8"
    neutral_edge, neutral_fill = "#333333", "#F5F5F5"
    ax.text(2, 61.5, "Study-specific pure-PyTorch SSM classifiers", fontsize=13, fontweight="semibold")
    ax.text(
        2,
        58.5,
        "Shared convolutional encoder and token interface; the residual sequence block is the controlled architectural difference.",
        fontsize=8.4,
        color=C["muted"],
    )

    def box(x, y, w, h, title, detail, fc, ec, title_color=C["ink"]):
        patch = Rectangle((x, y), w, h, linewidth=0.9, edgecolor=ec, facecolor=fc)
        ax.add_patch(patch)
        ax.text(x + 1.2, y + h - 1.55, title, fontsize=8.7, fontweight="semibold", color=title_color, va="top")
        if detail:
            ax.text(x + 1.2, y + h - 3.75, detail, fontsize=7.25, color=C["muted"], va="top", linespacing=1.26)
        return patch

    # Shared encoder: image, progressively reduced feature maps, then token interface.
    box(3, 46, 14, 8, "Input image", "224 × 224 × 3", neutral_fill, neutral_edge)
    for i in range(3):
        for j in range(3):
            ax.add_patch(Rectangle((4.2 + j * 2.4, 47.0 + i * 1.7), 1.55, 1.15, fc="#CBD5E1", ec="white", lw=0.35))
    arrow(ax, (17.5, 50), (22.2, 50))
    box(23, 44.4, 30, 11.2, "Shared convolutional stem", "4 × [Conv 3×3 / s2 · BN · GELU]\n224²×3  →  14²×192", neutral_fill, neutral_edge)
    for i, (w, h) in enumerate([(3.8, 3.8), (3.0, 3.0), (2.3, 2.3), (1.7, 1.7)]):
        x = 27.0 + i * 5.4
        y = 46.6 + (3.8 - h) / 2
        ax.add_patch(Rectangle((x, y), w, h, fc="#CBD5E1", ec="#94A3B8", lw=0.55))
    arrow(ax, (53.5, 50), (58.2, 50))
    box(59, 44.4, 18, 11.2, "Token interface", "Flatten + position\nL = 196, C = 192", neutral_fill, neutral_edge)
    for i in range(7):
        ax.add_patch(Rectangle((62.0 + i * 1.45, 47.0), 1.0, 3.0, fc="#D6D3D1", ec="white", lw=0.35))
    ax.text(79.2, 50, "same input\nfor both heads", fontsize=7.2, color=C["muted"], va="center")
    ax.text(40, 42.0, r"$x\in\mathbb{R}^{B\times3\times224\times224}\ \rightarrow\ h\in\mathbb{R}^{B\times196\times192}$",
            fontsize=7.4, color=C["muted"], ha="center")
    arrow(ax, (67.0, 44.0), (25.0, 39.4), C["shared_edge"])
    arrow(ax, (69.0, 44.0), (75.0, 39.4), C["shared_edge"])

    # Branch panels.
    panels = [
        (3, 9, 44, 30, "LiteSSM-A", r"MRFFILite ×4  ($d_{\mathrm{state}}=8$)", a_edge, a_fill),
        (53, 9, 44, 30, "LiteSSM-B", r"BiViM ×4  ($d_{\mathrm{state}}=16$)", b_edge, b_fill),
    ]
    for x, y, w, h, name, subtitle, edge, soft in panels:
        ax.add_patch(Rectangle((x, y), w, h, linewidth=1.0, edgecolor=edge, facecolor="#FFFFFF"))
        ax.add_patch(Rectangle((x, y + h - 5.4), w, 5.4, fc=soft, ec="none"))
        ax.text(x + 1.4, y + h - 1.75, name, fontsize=10.8, fontweight="semibold", color=edge, va="top")
        ax.text(x + w - 1.4, y + h - 1.95, subtitle, fontsize=7.5, color=C["muted"], ha="right", va="top")

    # LiteSSM-A block flow.
    box(6, 25, 7.2, 6.2, "LN", "", C["bg"], a_edge)
    box(16, 26.2, 10.5, 4.7, "Local path", "DWConv k={3,5,7}", a_fill, a_edge, a_edge)
    box(16, 18.2, 10.5, 4.7, "State path", "SelectiveSSM\nstate = 8", a_fill, a_edge, a_edge)
    box(30, 22.1, 8.7, 5.8, "Fuse", "concat → Linear", C["bg"], a_edge)
    ax.text(41.2, 24.8, "⊕", fontsize=17, color=a_edge, ha="center", va="center")
    ax.text(25, 11.2, "residual block repeated 4 times · expand = 1 · sequential scan", fontsize=7.4, color=C["muted"], ha="center")
    arrow(ax, (13.5, 28.1), (15.5, 28.5), "#666666")
    arrow(ax, (13.5, 27.1), (15.5, 20.6), "#666666")
    arrow(ax, (26.8, 28.5), (29.4, 25.4), "#666666")
    arrow(ax, (26.8, 20.6), (29.4, 24.6), "#666666")
    arrow(ax, (38.9, 25), (40.0, 25), "#666666")
    ax.add_patch(FancyArrowPatch((8.0, 25.0), (40.0, 22.8), arrowstyle="-", lw=0.7, color="#666666"))

    # LiteSSM-B block flow.
    box(56, 25, 7.2, 6.2, "LN", "", C["bg"], b_edge)
    box(66, 26.2, 10.5, 4.7, "Forward scan", "SelectiveSSM\nstate = 16", b_fill, b_edge, b_edge)
    box(66, 18.2, 10.5, 4.7, "Backward scan", "reverse → SSM\n→ reverse", b_fill, b_edge, b_edge)
    box(80, 22.1, 8.7, 5.8, "Fuse", "concat → Linear", C["bg"], b_edge)
    ax.text(91.2, 24.8, "⊕", fontsize=17, color=b_edge, ha="center", va="center")
    ax.text(75, 11.2, "residual block repeated 4 times · bidirectional selective scan", fontsize=7.4, color=C["muted"], ha="center")
    arrow(ax, (63.5, 28.1), (65.5, 28.5), "#666666")
    arrow(ax, (63.5, 27.1), (65.5, 20.6), "#666666")
    arrow(ax, (76.8, 28.5), (79.4, 25.4), "#666666")
    arrow(ax, (76.8, 20.6), (79.4, 24.6), "#666666")
    arrow(ax, (88.9, 25), (90.0, 25), "#666666")
    ax.add_patch(FancyArrowPatch((58.0, 25.0), (90.0, 22.8), arrowstyle="-", lw=0.7, color="#666666"))

    # Branch outputs and shared head convention.
    box(8, 2.5, 34, 6.0, "Classifier head", r"LayerNorm → $\frac{1}{L}\sum_i h_i$ → Linear(192 → 2)", C["bg"], a_edge)
    box(58, 2.5, 34, 6.0, "Classifier head", r"LayerNorm → $\frac{1}{L}\sum_i h_i$ → Linear(192 → 2)", C["bg"], b_edge)
    arrow(ax, (25, 9), (25, 8.7), "#666666")
    arrow(ax, (75, 9), (75, 8.7), "#666666")
    save(fig, "fig_ssm_architecture.png")
    plt.close(fig)


# ---------------------------------------------------------------------------
# Fig: Pareto
# ---------------------------------------------------------------------------
def fig_pareto(frozen: dict, ext: dict | None, lat: dict | None):
    """Prefer figures from render_analysis_figures.fig_pareto_inset.
    Skip overwrite here so re-running this script does not regress label layout."""
    target = OUT_DIRS[0] / "fig2_pareto_ufd_macro.png"
    if target.exists():
        print(f"skip fig_pareto (keep existing {target.name}; regenerate via render_analysis_figures.py)")
        return
    raise FileNotFoundError("Missing Pareto figure; run scripts/render_analysis_figures.py first")



# ---------------------------------------------------------------------------
# Fig: domain gap
# ---------------------------------------------------------------------------
def fig_domain(frozen: dict):
    names = [LABELS[k] for k in ORDER]
    celeb = [frozen["models"][n]["celebahq_auc"] for n in names]
    bed = [frozen["models"][n]["bedroom_auc"] for n in names]
    x = np.arange(len(names))
    w = 0.38

    fig, ax = plt.subplots(figsize=(9.4, 4.8))
    b1 = ax.bar(
        x - w / 2,
        celeb,
        w,
        label="CelebA-HQ",
        color="#0284C7",
        edgecolor="white",
        linewidth=0.6,
        hatch="///",
    )
    b2 = ax.bar(
        x + w / 2,
        bed,
        w,
        label="Bedroom",
        color="#EA580C",
        edgecolor="white",
        linewidth=0.6,
        hatch="...",
    )
    ax.axhline(0.5, color=C["rule"], ls="--", lw=1, zorder=0)
    ax.set_ylim(0.55, 1.02)
    ax.set_xticks(x)
    ax.set_xticklabels(names, rotation=18, ha="right")
    ax.set_ylabel("Within-ID AUC", fontsize=10.5)
    ax.set_title("Content-domain gap on the ID test split (y-axis: 0.55–1.02)", loc="left", fontweight="semibold")
    ax.legend(frameon=True, fancybox=False, edgecolor=C["rule"], loc="lower left")
    ax.grid(True, axis="y", alpha=0.22, color=C["rule"])
    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)
    for spine in ("left", "bottom"):
        ax.spines[spine].set_color(C["rule"])

    for rect, v in zip(b2, bed):
        ax.text(
            rect.get_x() + rect.get_width() / 2,
            v + 0.012,
            f"{v:.3f}",
            ha="center",
            va="bottom",
            fontsize=7.4,
            color=C["muted"],
        )

    fig.tight_layout()
    save(fig, "fig_domain_gap.png")
    plt.close(fig)


# ---------------------------------------------------------------------------
# Fig: heatmap
# ---------------------------------------------------------------------------
def fig_heatmap(pkg: dict):
    mat = np.zeros((len(ORDER), len(GEN_KEYS)))
    for i, n in enumerate(ORDER):
        per = pkg["models"][n]["ufd"]["per_generator"]
        for j, g in enumerate(GEN_KEYS):
            mat[i, j] = per[g]["auc"]

    fig, ax = plt.subplots(figsize=(10.6, 4.9))
    im = ax.imshow(mat, vmin=0.45, vmax=0.95, cmap=HEAT_CMAP, aspect="auto")
    cbar = fig.colorbar(im, ax=ax, fraction=0.035, pad=0.02)
    cbar.set_label("AUC", color=C["muted"], fontsize=10)
    cbar.ax.tick_params(labelsize=9)
    cbar.outline.set_edgecolor(C["rule"])

    ax.set_yticks(range(len(ORDER)))
    ax.set_yticklabels([LABELS[n] for n in ORDER], fontsize=9)
    ax.set_xticks(range(len(GEN_KEYS)))
    ax.set_xticklabels(GEN_LABELS, rotation=28, ha="right", fontsize=8.5)

    ax.add_patch(Rectangle((-0.5, -0.5), 1, len(ORDER), fill=False, ec=C["bad"], lw=1.8, zorder=5))

    for i in range(mat.shape[0]):
        for j in range(mat.shape[1]):
            val = mat[i, j]
            ax.text(
                j,
                i,
                f"{val:.2f}",
                ha="center",
                va="center",
                fontsize=8.0,
                color="white" if val >= 0.74 else C["ink"],
                fontweight="semibold" if j == 0 else "regular",
            )

    ax.set_title("UFD per-generator AUC (Panel A; FLUX excluded)", loc="left", fontweight="semibold")
    for spine in ax.spines.values():
        spine.set_visible(False)
    fig.tight_layout()
    save(fig, "fig3_generator_heatmap.png")
    plt.close(fig)


# ---------------------------------------------------------------------------
# Fig: JPEG multi-Q (appendix)
# ---------------------------------------------------------------------------
def fig_jpeg():
    xs = ["Clean", "Q95", "Q85", "Q70"]
    styles = {
        "mobilemamba_lite": ("-", "o"),
        "mambapsa_cls": ("--", "s"),
        "efficientnet_b0": ("-", "^"),
        "lite_freq_net_v2": ("-.", "D"),
        "mobilenet_v3_small": (":", "v"),
        "shufflenet_v2_x0_5": ("--", "P"),
    }
    fig, ax = plt.subplots(figsize=(9.0, 5.0))
    for key in ORDER:
        path = ROOT / "jpeg_results" / JPEG_FILES[key]
        j = load_json(path)
        ys = [
            float(j["clean_test"]["auc"]),
            float(j["jpeg"]["q95"]["test"]["auc"]),
            float(j["jpeg"]["q85"]["test"]["auc"]),
            float(j["jpeg"]["q70"]["test"]["auc"]),
        ]
        arch = ARCH[key]
        ls, mk = styles[key]
        lw = 2.3 if key == "mobilemamba_lite" else 1.5
        ax.plot(
            xs,
            ys,
            marker=mk,
            ms=6.5,
            lw=lw,
            ls=ls,
            color=ARCH_COLOR[arch],
            alpha=0.95,
            label=LABELS[key],
        )

    ax.set_ylim(0.86, 0.96)
    ax.set_ylabel("ID test AUC", fontsize=10.5)
    ax.set_title("JPEG recompression sweep (locked checkpoints)", loc="left", fontweight="semibold")
    ax.grid(True, axis="both", alpha=0.28, color=C["rule"])
    ax.legend(
        ncol=1,
        fontsize=8.4,
        frameon=True,
        fancybox=False,
        edgecolor=C["rule"],
        loc="center left",
        bbox_to_anchor=(1.02, 0.5),
    )
    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)
    for spine in ("left", "bottom"):
        ax.spines[spine].set_color(C["rule"])
    ax.text(
        0.02,
        0.05,
        "Absolute ΔAUC ≤ 0.003 vs clean",
        transform=ax.transAxes,
        ha="left",
        fontsize=7.8,
        color=C["muted"],
    )
    fig.tight_layout()
    save(fig, "fig_jpeg_quality_sweep.png")
    plt.close(fig)


# ---------------------------------------------------------------------------
# Fig: seed sensitivity (appendix)
# ---------------------------------------------------------------------------
def fig_seed(summary: dict):
    fig, axes = plt.subplots(1, 2, figsize=(9.8, 4.1), sharey=False)
    metrics = [("id_auc", "ID AUC"), ("ufd_macro_auc", "UFD Macro AUC")]
    models = ["LiteSSM-A", "EfficientNet-B0"]
    colors = [C["ssm"], C["cnn"]]
    seeds = [42, 43, 44]

    for ax, (field, title) in zip(axes, metrics):
        for i, (model, color) in enumerate(zip(models, colors)):
            runs = summary[model]["runs"]
            xs = np.array(seeds, dtype=float) + (i - 0.5) * 0.22
            ys = [r[field] for r in runs]
            mean = float(np.mean(ys))
            std = float(np.std(ys, ddof=0))
            ax.axhline(mean, color=color, ls=":", lw=1.0, alpha=0.55, zorder=1)
            ax.errorbar(
                [np.mean(xs)],
                [mean],
                yerr=[std],
                fmt="none",
                ecolor=color,
                elinewidth=1.0,
                capsize=3.5,
                capthick=1.0,
                zorder=2,
                alpha=0.85,
            )
            ax.scatter(xs, ys, s=48, color=color, zorder=4, edgecolors="white", linewidths=0.6, label=model if ax is axes[0] else None)
            ax.scatter([np.mean(xs)], [mean], s=42, marker="s", color=color, zorder=5, edgecolors="white", linewidths=0.7)
        ax.set_xticks(seeds)
        ax.set_xlabel("Training seed")
        ax.set_title(title, loc="left", fontsize=10, fontweight="semibold")
        ax.grid(True, axis="y", alpha=0.22, color=C["rule"])
        for spine in ("top", "right"):
            ax.spines[spine].set_visible(False)
        for spine in ("left", "bottom"):
            ax.spines[spine].set_color(C["rule"])

    axes[0].set_ylabel("AUC")
    axes[0].legend(frameon=True, fancybox=False, edgecolor=C["rule"], fontsize=8.2)
    fig.suptitle("Training-seed sensitivity (locked recipe; n=3)", fontsize=11, fontweight="semibold", y=1.02)
    fig.tight_layout()
    save(fig, "fig_seed_sensitivity.png")
    plt.close(fig)


# ---------------------------------------------------------------------------
# Graphical abstract (wide composite)
# ---------------------------------------------------------------------------
def fig_graphical_abstract(frozen: dict):
    fig = plt.figure(figsize=(12.0, 4.4))
    gs = fig.add_gridspec(1, 3, width_ratios=[1.15, 1.15, 1.0], wspace=0.28)

    # panel 1: protocol chips
    ax0 = fig.add_subplot(gs[0])
    ax0.set_xlim(0, 10)
    ax0.set_ylim(0, 10)
    ax0.axis("off")
    ax0.set_title("Protocol-locked compact AIGC detection", fontsize=10, fontweight="semibold", loc="left")
    items = [
        (0.4, 7.2, 9.2, 2.0, "Train from scratch on DF\n(bedroom + CelebA-HQ)", C["shared"], C["shared_edge"]),
        (0.4, 4.4, 9.2, 2.0, "Compare CNNs / LiteFreq / LiteSSM-A·B\nunder one recipe", C["ssm_soft"], C["ssm"]),
        (0.4, 1.6, 9.2, 2.0, "Report UFD Macro + domain gap\n+ JPEG Q70 + latency", C["freq_soft"], C["freq"]),
    ]
    for it in items:
        rounded(ax0, (it[0], it[1]), it[2], it[3], it[4], it[5], it[6], fontsize=8.0)

    # panel 2: mini pareto
    ax1 = fig.add_subplot(gs[1])
    for key in ORDER:
        name = LABELS[key]
        m = frozen["models"][name]
        ax1.scatter(
            m["batch1_p50_ms"],
            m["ufd_macro_auc"],
            s=max(60, m["params_M"] * 70),
            c=ARCH_COLOR[ARCH[key]],
            edgecolors=C["ink"] if name == "LiteSSM-A" else "white",
            linewidths=1.4 if name == "LiteSSM-A" else 0.7,
            zorder=3,
        )
    ax1.scatter([11.86, 2.89], [0.948, 0.976], s=70, facecolors="none", edgecolors=C["ref"], marker="D", linewidths=1.4)
    ax1.set_xscale("log")
    ax1.set_xlim(2, 320)
    ax1.set_ylim(0.60, 1.02)
    ax1.set_xlabel("Batch-1 latency (ms)", fontsize=8)
    ax1.set_ylabel("UFD Macro", fontsize=8)
    ax1.set_title("Operating points (★ LiteSSM-A)", fontsize=10, fontweight="semibold", loc="left")
    ax1.annotate("LiteSSM-A", (144.4, 0.718), xytext=(20, 12), textcoords="offset points", fontsize=7.5, color=C["ssm"], fontweight="semibold")
    ax1.grid(True, alpha=0.2, color=C["rule"])
    for spine in ("top", "right"):
        ax1.spines[spine].set_visible(False)

    # panel 3: takeaway
    ax2 = fig.add_subplot(gs[2])
    ax2.set_xlim(0, 10)
    ax2.set_ylim(0, 10)
    ax2.axis("off")
    ax2.set_title("Main takeaway", fontsize=10, fontweight="semibold", loc="left")
    rounded(
        ax2,
        (0.3, 5.5),
        9.4,
        3.6,
        "LiteSSM-A: highest Panel-A\nUFD Macro (0.718)\nat 1.74M / 144.4 ms\n≠ lowest-latency model",
        C["ssm_soft"],
        C["ssm"],
        8.4,
        "semibold",
        lw=1.6,
    )
    rounded(
        ax2,
        (0.3, 1.2),
        9.4,
        3.4,
        "Boundaries remain:\nDALL·E ≈ chance (Panel A)\nbedroom ≪ CelebA-HQ\nJPEG Q70 Δ≤0.003",
        C["accent_soft"],
        C["accent"],
        8.2,
    )

    save(fig, "fig_graphical_abstract.png")
    plt.close(fig)


def main():
    style()
    frozen = load_json(ROOT / "freeze" / "frozen_numbers.json")
    pkg = load_json(ROOT / "freeze" / "freeze_package.json")
    ext = load_json(ROOT / "external_refs" / "summary.json") if (ROOT / "external_refs" / "summary.json").exists() else None
    seed = load_json(ROOT / "docs" / "seed_sweep_summary.json")

    fig_protocol()
    fig_architecture()
    fig_pareto(frozen, ext, None)
    fig_domain(frozen)
    fig_heatmap(pkg)
    fig_jpeg()
    fig_seed(seed)
    fig_graphical_abstract(frozen)
    print("ALL FIGURES DONE")


if __name__ == "__main__":
    main()
