#!/usr/bin/env python3
"""Regenerate Figure 2 (Pareto: UFD Macro vs batch-1 latency) and Figure 3 (UFD heatmap, no FLUX)."""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parent
LAT = ROOT / "latency_batch1" / "summary.json"
PKG = ROOT / "freeze" / "freeze_package.json"
EXT = ROOT / "external_refs" / "summary.json"
OUT_DIRS = [
    ROOT / "freeze" / "figures",
    ROOT / "latex" / "figures",
]

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
SHORT_GEN = {
    "ufd_dalle": "DALL·E",
    "ufd_glide_100_10": "G10",
    "ufd_glide_100_27": "G27",
    "ufd_glide_50_27": "G50",
    "ufd_guided": "Guided",
    "ufd_ldm_100": "L100",
    "ufd_ldm_200": "L200",
    "ufd_ldm_200_cfg": "Lcfg",
}
GENS = list(SHORT_GEN.keys())
COLORS = {
    "SSM": "#1f4e79",
    "CNN": "#2e7d4f",
    "CNN+FFT": "#b45309",
    "REF": "#6b21a8",
}


def arch_of(name: str, pkg: dict) -> str:
    return pkg["models"][name].get("arch", "CNN")


def save_all(fig, name: str):
    for d in OUT_DIRS:
        d.mkdir(parents=True, exist_ok=True)
        path = d / name
        fig.savefig(path, dpi=220, bbox_inches="tight")
        print("wrote", path)


def fig2_pareto(lat: dict, pkg: dict, ext: dict | None):
    fig, ax = plt.subplots(figsize=(7.6, 5.2))

    # Panel A
    for n in ORDER:
        m = pkg["models"][n]
        x = float(lat[n]["batch1_latency"]["p50_ms"])
        y = float(m["ufd"]["macro_auc"])
        s = max(70.0, float(m["efficiency"]["params_M"]) * 90)
        arch = arch_of(n, pkg)
        if n == "lite_freq_net_v2":
            arch = "CNN+FFT"
        ax.scatter(
            [x],
            [y],
            s=s,
            c=COLORS.get(arch, "#333"),
            alpha=0.88,
            edgecolors="white",
            linewidths=0.8,
            zorder=3,
            label=None,
        )
        # offset labels to reduce overlap among CNNs near 4–7 ms
        offsets = {
            "mobilenet_v3_small": (6, -12),
            "shufflenet_v2_x0_5": (6, 8),
            "lite_freq_net_v2": (6, -2),
            "efficientnet_b0": (6, 10),
            "mobilemamba_lite": (8, 6),
            "mambapsa_cls": (8, -10),
        }
        ox, oy = offsets.get(n, (6, 4))
        ax.annotate(
            LABELS[n],
            (x, y),
            textcoords="offset points",
            xytext=(ox, oy),
            fontsize=8.5,
            color="#1a1a1a",
        )

    # Optional Panel B reference markers (hollow)
    if ext:
        refs = [
            ("univfd", "UnivFD (ref)", ext["univfd"]),
            ("npr", "NPR (ref)", ext["npr"]),
        ]
        for key, lab, rep in refs:
            x = float(rep["latency_batch1"]["p50_ms"])
            y = float(rep["splits"]["ufd_eval"]["ufd_macro_auc"])
            ax.scatter(
                [x],
                [y],
                s=120,
                facecolors="none",
                edgecolors=COLORS["REF"],
                linewidths=1.6,
                marker="D",
                zorder=4,
            )
            ax.annotate(
                lab,
                (x, y),
                textcoords="offset points",
                xytext=(8, 4),
                fontsize=8,
                color=COLORS["REF"],
            )

    ax.set_xscale("log")
    ax.set_xlabel("Batch-1 latency (ms/image, FP32, model-only, p50)")
    ax.set_ylabel("UFD Macro AUC")
    ax.set_title("Accuracy–efficiency operating points (Panel A; refs hollow)")
    ax.set_ylim(0.60, 1.02)
    ax.grid(True, which="both", alpha=0.28)
    # legend proxies
    from matplotlib.lines import Line2D

    handles = [
        Line2D([0], [0], marker="o", color="w", markerfacecolor=COLORS["SSM"], markersize=9, label="SSM (Panel A)"),
        Line2D([0], [0], marker="o", color="w", markerfacecolor=COLORS["CNN"], markersize=9, label="CNN (Panel A)"),
        Line2D([0], [0], marker="o", color="w", markerfacecolor=COLORS["CNN+FFT"], markersize=9, label="CNN+FFT (Panel A)"),
        Line2D([0], [0], marker="D", color=COLORS["REF"], markerfacecolor="none", markersize=8, label="External ref (Panel B)"),
    ]
    ax.legend(handles=handles, loc="lower right", fontsize=8, framealpha=0.92)
    ax.text(
        0.02,
        0.02,
        "Marker size ∝ Params(M)\nx-axis log-scaled",
        transform=ax.transAxes,
        fontsize=7.5,
        va="bottom",
        color="#444",
    )
    fig.tight_layout()
    save_all(fig, "fig2_pareto_ufd_macro.png")
    plt.close(fig)


def fig3_heatmap(pkg: dict):
    mat = np.zeros((len(ORDER), len(GENS)))
    for i, n in enumerate(ORDER):
        per = pkg["models"][n]["ufd"]["per_generator"]
        for j, g in enumerate(GENS):
            mat[i, j] = per[g]["auc"]

    fig, ax = plt.subplots(figsize=(9.8, 4.6))
    im = ax.imshow(mat, vmin=0.45, vmax=1.0, cmap="viridis", aspect="auto")
    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label("AUC")
    ax.set_yticks(range(len(ORDER)))
    ax.set_yticklabels([LABELS[n] for n in ORDER], fontsize=8.5)
    ax.set_xticks(range(len(GENS)))
    ax.set_xticklabels([SHORT_GEN[g] for g in GENS], rotation=30, ha="right", fontsize=8.5)
    for i in range(mat.shape[0]):
        for j in range(mat.shape[1]):
            ax.text(
                j,
                i,
                f"{mat[i, j]:.2f}",
                ha="center",
                va="center",
                color="w" if mat[i, j] < 0.72 else "k",
                fontsize=7.5,
            )
    ax.set_title("UFD per-generator AUC (Panel A; FLUX excluded)")
    fig.tight_layout()
    save_all(fig, "fig3_generator_heatmap.png")
    plt.close(fig)


def main():
    lat = json.loads(LAT.read_text(encoding="utf-8"))
    pkg = json.loads(PKG.read_text(encoding="utf-8"))
    ext = None
    if EXT.exists():
        ext = json.loads(EXT.read_text(encoding="utf-8"))
    fig2_pareto(lat, pkg, ext)
    fig3_heatmap(pkg)
    print("DONE")


if __name__ == "__main__":
    main()
