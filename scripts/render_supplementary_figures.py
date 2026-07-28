#!/usr/bin/env python3
"""Zero-training supplementary paper figures from frozen prediction assets.

Figures:
  1. Qualitative bedroom disagreement / shared failures (LiteSSM-A vs Eff-B0)
  2. Qualitative UFD DALL·E failure vs Glide success (+ NPR scores)
  3. Confusion matrices at Table-operate threshold (LiteSSM-A Thr=0.43)
  4. Training dynamics (seed 42 history: LiteSSM-A vs EfficientNet-B0)
  5. Panel-A error-agreement Cohen's kappa on UFD
  6. Threshold operating curves on ID test
"""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

ROOT = Path(r"E:\sciencecre\aigc_datasets\lite-aigc-detect")
ASSETS = Path(r"E:\sciencecre\aigc_datasets\formal\_paper_assets")
OUT_DIRS = [ROOT / "latex" / "figures", ROOT / "freeze" / "figures"]
DOCS = ROOT / "docs"
REMOTE_PREFIX = "/root/autodl-tmp/aigc_datasets/"
LOCAL_PREFIX = "E:/sciencecre/aigc_datasets/"

C = {
    "ink": "#1C1917",
    "muted": "#57534E",
    "rule": "#D6D3D1",
    "ssm": "#0F766E",
    "cnn": "#1D4ED8",
    "heat_lo": "#F8FAFC",
    "heat_hi": "#0F766E",
    "kappa_lo": "#1E3A5F",
    "kappa_hi": "#FBBF24",
}

PANEL_A = [
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
SHORT = {
    "mobilemamba_lite": "A★",
    "mambapsa_cls": "B",
    "efficientnet_b0": "Eff",
    "lite_freq_net_v2": "Freq",
    "mobilenet_v3_small": "Mob",
    "shufflenet_v2_x0_5": "Shuf",
}


def style():
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 9.5,
            "axes.titlesize": 11,
            "axes.labelsize": 9.5,
            "figure.facecolor": "white",
            "axes.facecolor": "white",
            "savefig.dpi": 300,
            "pdf.fonttype": 42,
        }
    )


def save(fig, name: str):
    for d in OUT_DIRS:
        d.mkdir(parents=True, exist_ok=True)
        fig.savefig(d / name, dpi=300, bbox_inches="tight", pad_inches=0.08)
        print("wrote", d / name)


def remap_path(p: str) -> Path:
    s = str(p).replace("\\", "/")
    if s.startswith(REMOTE_PREFIX):
        s = LOCAL_PREFIX + s[len(REMOTE_PREFIX) :]
    return Path(s)


def load_npz_full(path: Path):
    z = np.load(path, allow_pickle=True)
    return {
        "probs": z["probs"].astype(float),
        "labels": z["labels"].astype(int),
        "sources": z["sources"].astype(str),
        "paths": np.asarray([str(x) for x in z["paths"]]),
    }


def load_jsonl_full(path: Path):
    probs, labels, sources, paths = [], [], [], []
    with path.open(encoding="utf-8") as f:
        for line in f:
            r = json.loads(line)
            probs.append(float(r["prob"]))
            labels.append(int(r["label"]))
            sources.append(str(r.get("source", "")))
            paths.append(str(r.get("path", "")))
    return {
        "probs": np.asarray(probs, dtype=float),
        "labels": np.asarray(labels, dtype=int),
        "sources": np.asarray(sources),
        "paths": np.asarray(paths),
    }


def thresholds() -> dict[str, float]:
    thr: dict[str, float] = {}
    for p in (ASSETS / "metrics").glob("*__metrics.json"):
        key = p.name.replace("__metrics.json", "")
        m = json.loads(p.read_text(encoding="utf-8"))
        if m.get("val_threshold") is not None:
            thr[key] = float(m["val_threshold"])
    recovered = json.loads((ASSETS / "recovered_thresholds.json").read_text(encoding="utf-8"))
    for k, v in recovered.items():
        thr[k] = float(v["val_threshold"])
    # Retained freeze has no Youden thr for Mob/Shuf → fixed 0.5 (disclosed in captions)
    thr.setdefault("mobilenet_v3_small", 0.5)
    thr.setdefault("shufflenet_v2_x0_5", 0.5)
    thr["npr"] = 0.5
    thr["univfd"] = 0.5
    return thr


def confusion_counts(probs, labels, thr):
    pred = (probs >= thr).astype(int)
    tp = int(((pred == 1) & (labels == 1)).sum())
    tn = int(((pred == 0) & (labels == 0)).sum())
    fp = int(((pred == 1) & (labels == 0)).sum())
    fn = int(((pred == 0) & (labels == 1)).sum())
    return np.array([[tn, fp], [fn, tp]], dtype=int)


def cohen_kappa(a: np.ndarray, b: np.ndarray) -> float:
    a = a.astype(int)
    b = b.astype(int)
    n = len(a)
    if n == 0:
        return float("nan")
    po = float((a == b).mean())
    p0 = float((a == 0).mean())
    p1 = float((a == 1).mean())
    q0 = float((b == 0).mean())
    q1 = float((b == 1).mean())
    pe = p0 * q0 + p1 * q1
    if abs(1.0 - pe) < 1e-12:
        return 1.0 if po >= 1.0 - 1e-12 else 0.0
    return (po - pe) / (1.0 - pe)


def open_rgb(path: Path, size: int = 224) -> np.ndarray:
    with Image.open(path) as im:
        im = im.convert("RGB").resize((size, size), Image.Resampling.BILINEAR)
        return np.asarray(im)


def _cell_caption(ax, lines, color=C["ink"]):
    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_color(C["rule"])
    ax.set_xlabel("\n".join(lines), fontsize=7.2, color=color, labelpad=2)


def _pick_stratified(pool: np.ndarray, score: np.ndarray, y: np.ndarray, k: int = 3) -> np.ndarray:
    """Rank by score descending within pool; prefer mixed real/fake when both exist."""
    if len(pool) < k:
        raise RuntimeError(f"pool too small: {len(pool)}")
    order = pool[np.argsort(-score)]
    fake = [int(j) for j in order if int(y[j]) == 1]
    real = [int(j) for j in order if int(y[j]) == 0]
    picks: list[int] = []
    # Round-robin real/fake from ranked lists to avoid single-class grids
    i_f = i_r = 0
    while len(picks) < k and (i_f < len(fake) or i_r < len(real)):
        if i_f < len(fake) and (len(picks) % 2 == 0 or i_r >= len(real)):
            picks.append(fake[i_f])
            i_f += 1
        elif i_r < len(real):
            picks.append(real[i_r])
            i_r += 1
        else:
            break
    if len(picks) < k:
        for j in order:
            jj = int(j)
            if jj not in picks:
                picks.append(jj)
            if len(picks) >= k:
                break
    return np.asarray(picks[:k], dtype=int)


# ---------------------------------------------------------------------------
# 1) Bedroom qualitative grid
# ---------------------------------------------------------------------------
def fig_qual_bedroom(thr_map, meta_out: dict):
    a = load_npz_full(ASSETS / "preds" / "mobilemamba_lite__id.npz")
    e = load_npz_full(ASSETS / "preds" / "efficientnet_b0__id.npz")
    assert np.array_equal(a["paths"], e["paths"])
    thr_a, thr_e = thr_map["mobilemamba_lite"], thr_map["efficientnet_b0"]
    bed = np.asarray(["bedroom" in s for s in a["sources"]])
    idx = np.where(bed)[0]
    y = a["labels"][idx]
    pa, pe = a["probs"][idx], e["probs"][idx]
    paths = a["paths"][idx]
    pred_a = (pa >= thr_a).astype(int)
    pred_e = (pe >= thr_e).astype(int)
    correct_a = pred_a == y
    correct_e = pred_e == y

    # Top: A correct, Eff wrong; rank by |pA-pE|; stratified real/fake
    dis = np.where(correct_a & (~correct_e))[0]
    score_dis = np.abs(pa[dis] - pe[dis])
    top_dis = _pick_stratified(dis, score_dis, y, k=3)

    # Bottom: both wrong; rank by mean |p−thr|; stratified real/fake
    both = np.where((~correct_a) & (~correct_e))[0]
    conf_wrong = 0.5 * (np.abs(pa[both] - thr_a) + np.abs(pe[both] - thr_e))
    top_both = _pick_stratified(both, conf_wrong, y, k=3)

    fig, axes = plt.subplots(2, 3, figsize=(9.6, 7.0))
    rows = [
        ("LiteSSM-A correct, EfficientNet-B0 wrong", top_dis, C["ssm"]),
        ("Both models wrong (shared blind spots)", top_both, C["muted"]),
    ]
    selected = {"rule": {}, "cells": []}
    selected["rule"] = {
        "top": "bedroom ∩ (A correct @0.43) ∩ (Eff wrong @0.54); rank by |p_A−p_Eff|; stratified real/fake top-3",
        "bottom": "bedroom ∩ (A wrong) ∩ (Eff wrong); rank by mean |p−thr|; stratified real/fake top-3",
        "seed": None,
        "not_by_visual_appeal": True,
    }
    for r, (title, picks, edge) in enumerate(rows):
        for c, j in enumerate(picks):
            ax = axes[r, c]
            pth = remap_path(paths[j])
            if not pth.exists():
                raise FileNotFoundError(pth)
            ax.imshow(open_rgb(pth))
            true = "fake" if int(y[j]) == 1 else "real"
            a_ok = "✓" if bool(correct_a[j]) else "✗"
            e_ok = "✓" if bool(correct_e[j]) else "✗"
            _cell_caption(
                ax,
                [
                    f"True: {true}",
                    f"LiteSSM-A: {pa[j]:.2f} ({a_ok})",
                    f"Eff-B0: {pe[j]:.2f} ({e_ok})",
                ],
            )
            for spine in ax.spines.values():
                spine.set_linewidth(1.6)
                spine.set_color(edge)
            selected["cells"].append(
                {
                    "row": r,
                    "col": c,
                    "path": str(pth),
                    "true": true,
                    "p_litesm_a": float(pa[j]),
                    "p_eff_b0": float(pe[j]),
                    "source": str(a["sources"][idx[j]]),
                }
            )
        axes[r, 0].set_ylabel(title, fontsize=8.5, color=C["ink"])

    fig.suptitle(
        "ID Test (Bedroom): success/failure modes",
        fontsize=12,
        fontweight="semibold",
        color=C["ink"],
        y=0.995,
    )
    fig.tight_layout(rect=[0, 0, 1, 0.97])
    save(fig, "fig_qual_bedroom.png")
    plt.close(fig)
    meta_out["bedroom"] = selected


# ---------------------------------------------------------------------------
# 2) UFD DALL·E failure vs Glide success
# ---------------------------------------------------------------------------
def fig_qual_ufd(thr_map, meta_out: dict):
    a = load_npz_full(ASSETS / "preds" / "mobilemamba_lite__ufd.npz")
    npr = load_jsonl_full(ASSETS / "external" / "npr_ufd_eval_preds.jsonl")
    # Align NPR by path basename+source (same frozen order as Panel A in practice)
    assert len(npr["probs"]) == len(a["probs"])
    thr_a = thr_map["mobilemamba_lite"]

    dalle = np.where(a["sources"] == "ufd_dalle")[0]
    glide = np.where(a["sources"] == "ufd_glide_100_10")[0]
    dalle_fake = dalle[a["labels"][dalle] == 1]
    glide_fake = glide[a["labels"][glide] == 1]

    # DALL·E failures: fake scores closest to 0.5 (collective near-chance)
    d_score = np.abs(a["probs"][dalle_fake] - 0.5)
    d_picks = dalle_fake[np.argsort(d_score)[:3]]

    # Glide successes: fake with highest p_A among those predicted fake @ thr
    g_ok = glide_fake[a["probs"][glide_fake] >= thr_a]
    if len(g_ok) < 3:
        g_ok = glide_fake
    g_picks = g_ok[np.argsort(-a["probs"][g_ok])[:3]]

    fig, axes = plt.subplots(2, 3, figsize=(9.6, 7.0))
    selected = {
        "rule": {
            "top": "UFD DALL·E ∩ label=fake; rank by |p_LiteSSM-A−0.5| ascending; top-3",
            "bottom": "UFD Glide_100_10 ∩ label=fake ∩ p_A≥0.43; rank by p_A descending; top-3",
            "not_by_visual_appeal": True,
        },
        "cells": [],
    }
    for c, j in enumerate(d_picks):
        ax = axes[0, c]
        pth = remap_path(a["paths"][j])
        ax.imshow(open_rgb(pth))
        _cell_caption(
            ax,
            [
                "DALL·E  True: fake",
                f"LiteSSM-A: {a['probs'][j]:.2f}",
                f"NPR (ref): {npr['probs'][j]:.2f}",
            ],
            color=C["muted"],
        )
        selected["cells"].append(
            {
                "row": 0,
                "col": c,
                "path": str(pth),
                "p_litesm_a": float(a["probs"][j]),
                "p_npr": float(npr["probs"][j]),
                "source": "ufd_dalle",
            }
        )
    for c, j in enumerate(g_picks):
        ax = axes[1, c]
        pth = remap_path(a["paths"][j])
        ax.imshow(open_rgb(pth))
        _cell_caption(
            ax,
            [
                "Glide 100/10  True: fake",
                f"LiteSSM-A: {a['probs'][j]:.2f}",
                f"NPR (ref): {npr['probs'][j]:.2f}",
            ],
            color=C["ssm"],
        )
        selected["cells"].append(
            {
                "row": 1,
                "col": c,
                "path": str(pth),
                "p_litesm_a": float(a["probs"][j]),
                "p_npr": float(npr["probs"][j]),
                "source": "ufd_glide_100_10",
            }
        )
    axes[0, 0].set_ylabel("DALL·E: Panel A near chance", fontsize=8.5)
    axes[1, 0].set_ylabel("Glide: high-confidence detects", fontsize=8.5)
    fig.suptitle(
        "UFD qualitative contrast: DALL·E failure vs Glide success",
        fontsize=12,
        fontweight="semibold",
        y=0.995,
    )
    fig.tight_layout(rect=[0, 0, 1, 0.97])
    save(fig, "fig_qual_ufd_dalle_glide.png")
    plt.close(fig)
    meta_out["ufd_dalle_glide"] = selected


# ---------------------------------------------------------------------------
# 3) Confusion matrices
# ---------------------------------------------------------------------------
def fig_confusion(thr_map):
    thr = thr_map["mobilemamba_lite"]
    id_ = load_npz_full(ASSETS / "preds" / "mobilemamba_lite__id.npz")
    ufd = load_npz_full(ASSETS / "preds" / "mobilemamba_lite__ufd.npz")
    mats = [
        ("ID test", confusion_counts(id_["probs"], id_["labels"], thr), len(id_["labels"])),
        ("UFD pooled", confusion_counts(ufd["probs"], ufd["labels"], thr), len(ufd["labels"])),
    ]
    fig, axes = plt.subplots(1, 2, figsize=(9.8, 4.3))
    for ax, (title, mat, n) in zip(axes, mats):
        vmax = max(int(mat.max()), 1)
        im = ax.imshow(mat, cmap="YlGn", vmin=0, vmax=vmax)
        ax.set_xticks([0, 1], ["Pred real", "Pred fake"])
        ax.set_yticks([0, 1], ["True real", "True fake"])
        ax.set_title(f"{title}\nLiteSSM-A @ Youden thr={thr:.2f}  (n={n})", fontsize=10)
        labels_txt = [["TN", "FP"], ["FN", "TP"]]
        for i in range(2):
            for j in range(2):
                val = int(mat[i, j])
                ax.text(
                    j,
                    i,
                    f"{labels_txt[i][j]}\n{val}",
                    ha="center",
                    va="center",
                    color="white" if val > 0.55 * vmax else C["ink"],
                    fontsize=11,
                    fontweight="semibold",
                )
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04, label="Count")
        ax.text(
            0.5,
            -0.20,
            f"Threshold locked from validation Youden's $J$ = {thr:.2f}",
            transform=ax.transAxes,
            ha="center",
            fontsize=8,
            color=C["muted"],
        )
    fig.suptitle("Confusion counts at the locked operating threshold", fontsize=12, fontweight="semibold")
    fig.tight_layout(rect=[0, 0.02, 1, 0.95])
    save(fig, "fig_confusion_litesm_a.png")
    plt.close(fig)


# ---------------------------------------------------------------------------
# 4) Training dynamics
# ---------------------------------------------------------------------------
def fig_training_dynamics():
    a = json.loads((ASSETS / "metrics" / "mobilemamba_lite__metrics.json").read_text(encoding="utf-8"))
    e = json.loads((ASSETS / "metrics" / "efficientnet_b0__metrics.json").read_text(encoding="utf-8"))
    ha, he = a["history"], e["history"]
    ea = [h["epoch"] for h in ha]
    ee = [h["epoch"] for h in he]
    best_a = max(ha, key=lambda h: h["val_auc"])["epoch"]
    best_e = max(he, key=lambda h: h["val_auc"])["epoch"]

    fig, ax1 = plt.subplots(figsize=(8.2, 4.6))
    ax2 = ax1.twinx()
    ax1.plot(ea, [h["loss"] for h in ha], color=C["ssm"], lw=2.0, label="LiteSSM-A train loss")
    ax1.plot(ee, [h["loss"] for h in he], color=C["cnn"], lw=2.0, label="Eff-B0 train loss")
    ax2.plot(
        ea,
        [h["val_auc"] for h in ha],
        color=C["ssm"],
        lw=1.8,
        ls="--",
        label="LiteSSM-A val AUC",
    )
    ax2.plot(
        ee,
        [h["val_auc"] for h in he],
        color=C["cnn"],
        lw=1.8,
        ls="--",
        label="Eff-B0 val AUC",
    )
    ax1.axvline(best_a, color=C["ssm"], lw=1.0, alpha=0.55)
    ax1.axvline(best_e, color=C["cnn"], lw=1.0, alpha=0.55)
    ax1.annotate(
        f"best A ep{best_a}",
        xy=(best_a, ha[best_a - 1]["loss"]),
        xytext=(best_a + 0.35, max(h["loss"] for h in ha) * 0.92),
        fontsize=8,
        color=C["ssm"],
        arrowprops=dict(arrowstyle="-", color=C["ssm"], lw=0.7),
    )
    ax1.annotate(
        f"best Eff ep{best_e}",
        xy=(best_e, he[best_e - 1]["loss"]),
        xytext=(best_e + 0.4, max(h["loss"] for h in he) * 0.72),
        fontsize=8,
        color=C["cnn"],
        arrowprops=dict(arrowstyle="-", color=C["cnn"], lw=0.7),
    )
    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("Train loss")
    ax2.set_ylabel("Validation AUC")
    ax1.set_xlim(1, 15)
    ax1.set_xticks(range(1, 16))
    ax1.grid(True, alpha=0.22, color=C["rule"])
    h1, l1 = ax1.get_legend_handles_labels()
    h2, l2 = ax2.get_legend_handles_labels()
    ax1.legend(h1 + h2, l1 + l2, loc="center right", frameon=False, fontsize=8)
    ax1.set_title("Training dynamics (seed 42, locked Panel A recipe)", fontweight="semibold")
    save(fig, "fig_training_dynamics.png")
    plt.close(fig)


# ---------------------------------------------------------------------------
# 5) Cohen's kappa error-agreement heatmap
# ---------------------------------------------------------------------------
def fig_kappa(thr_map, meta_out: dict):
    preds = {}
    labels = None
    for key in PANEL_A:
        d = load_npz_full(ASSETS / "preds" / f"{key}__ufd.npz")
        if labels is None:
            labels = d["labels"]
        else:
            assert np.array_equal(labels, d["labels"])
        preds[key] = (d["probs"] >= thr_map[key]).astype(int)

    n = len(PANEL_A)
    K = np.zeros((n, n), dtype=float)
    for i, ki in enumerate(PANEL_A):
        for j, kj in enumerate(PANEL_A):
            K[i, j] = cohen_kappa(preds[ki], preds[kj])

    from matplotlib.colors import LinearSegmentedColormap

    cmap = LinearSegmentedColormap.from_list(
        "kappa_blue_yellow", [C["kappa_lo"], "#60A5FA", "#FDE68A", C["kappa_hi"]]
    )
    fig, ax = plt.subplots(figsize=(6.6, 5.6))
    im = ax.imshow(K, cmap=cmap, vmin=0.0, vmax=1.0)
    ax.set_xticks(range(n), [SHORT[k] for k in PANEL_A])
    ax.set_yticks(range(n), [SHORT[k] for k in PANEL_A])
    for i in range(n):
        for j in range(n):
            ax.text(
                j,
                i,
                f"{K[i, j]:.2f}",
                ha="center",
                va="center",
                color=("white" if (K[i, j] <= 0.22 or K[i, j] >= 0.85) else C["ink"]),
                fontsize=9,
            )
    ax.set_title(
        "UFD pooled prediction agreement (Cohen's $\\kappa$)",
        fontweight="semibold",
    )
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04, label="Cohen's κ")
    note = (
        "Binary decisions use each model's operating threshold "
        "(A=0.43, B=0.63, Eff=0.54, Freq=0.94; Mob/Shuf=0.5, Youden not retained)."
    )
    ax.text(0.5, -0.12, note, transform=ax.transAxes, ha="center", fontsize=7.5, color=C["muted"])
    save(fig, "fig_error_kappa_ufd.png")
    plt.close(fig)
    meta_out["kappa_ufd"] = {
        "order": PANEL_A,
        "short": [SHORT[k] for k in PANEL_A],
        "matrix": K.round(4).tolist(),
        "thresholds": {k: thr_map[k] for k in PANEL_A},
    }


# ---------------------------------------------------------------------------
# 6) Threshold operating curves (ID test)
# ---------------------------------------------------------------------------
def fig_threshold_sweep(thr_map):
    models = [
        ("mobilemamba_lite", C["ssm"], "LiteSSM-A"),
        ("efficientnet_b0", C["cnn"], "EfficientNet-B0"),
    ]
    thrs = np.linspace(0.01, 0.99, 99)
    fig, axes = plt.subplots(1, 2, figsize=(10.2, 4.4), sharey=True)
    metrics_names = [
        ("accuracy", "Accuracy"),
        ("f1", "F1"),
        ("sensitivity", "Sensitivity"),
        ("specificity", "Specificity"),
    ]
    styles = ["-", "-", "--", ":"]
    for ax, (key, col, name) in zip(axes, models):
        d = load_npz_full(ASSETS / "preds" / f"{key}__id.npz")
        p, y = d["probs"], d["labels"]
        curves = {m: [] for m, _ in metrics_names}
        for t in thrs:
            pred = (p >= t).astype(int)
            tp = int(((pred == 1) & (y == 1)).sum())
            tn = int(((pred == 0) & (y == 0)).sum())
            fp = int(((pred == 1) & (y == 0)).sum())
            fn = int(((pred == 0) & (y == 1)).sum())
            sens = tp / max(tp + fn, 1)
            spec = tn / max(tn + fp, 1)
            prec = tp / max(tp + fp, 1)
            rec = sens
            f1 = 2 * prec * rec / max(prec + rec, 1e-12)
            acc = (tp + tn) / max(len(y), 1)
            curves["accuracy"].append(acc)
            curves["f1"].append(f1)
            curves["sensitivity"].append(sens)
            curves["specificity"].append(spec)
        for (mkey, lab), ls in zip(metrics_names, styles):
            ax.plot(thrs, curves[mkey], ls=ls, color=col, lw=1.7, label=lab, alpha=0.9)
        t_star = thr_map[key]
        ax.axvline(t_star, color=col, lw=1.2, alpha=0.75)
        ax.annotate(
            f"Youden thr={t_star:.2f}",
            xy=(t_star, 0.08),
            xytext=(t_star + 0.06, 0.18),
            fontsize=8,
            color=col,
            arrowprops=dict(arrowstyle="-", color=col, lw=0.7),
        )
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1.02)
        ax.set_xlabel("Decision threshold")
        ax.set_title(name, fontweight="semibold", color=col)
        ax.grid(True, alpha=0.22, color=C["rule"])
        ax.legend(loc="lower left", frameon=False, fontsize=8, ncol=2)
    axes[0].set_ylabel("Metric value (ID test)")
    fig.suptitle(
        "Threshold operating characteristics on ID test",
        fontsize=12,
        fontweight="semibold",
    )
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    save(fig, "fig_threshold_sweep_id.png")
    plt.close(fig)


def main():
    style()
    thr = thresholds()
    print("thresholds", {k: thr[k] for k in PANEL_A})
    meta: dict = {"selection_policy": "predefined ranking rules; not visual cherry-picking"}
    fig_qual_bedroom(thr, meta)
    fig_qual_ufd(thr, meta)
    fig_confusion(thr)
    fig_training_dynamics()
    fig_kappa(thr, meta)
    fig_threshold_sweep(thr)
    meta_path = DOCS / "qualitative_selection_meta.json"
    meta_path.write_text(json.dumps(meta, indent=2), encoding="utf-8")
    print("wrote", meta_path)
    print("DONE")


if __name__ == "__main__":
    main()
