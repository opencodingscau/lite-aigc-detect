#!/usr/bin/env python3
"""Compute operating metrics + regenerate analysis-depth paper figures."""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D
from mpl_toolkits.axes_grid1.inset_locator import inset_axes

ROOT = Path(r"E:\sciencecre\aigc_datasets\lite-aigc-detect")
ASSETS = Path(r"E:\sciencecre\aigc_datasets\formal\_paper_assets")
OUT_DIRS = [ROOT / "latex" / "figures", ROOT / "freeze" / "figures"]
DOCS = ROOT / "docs"

C = {
    "ink": "#1C1917",
    "muted": "#57534E",
    "rule": "#D6D3D1",
    "ssm": "#0F766E",
    "cnn": "#1D4ED8",
    "freq": "#C2410C",
    "ref": "#44403C",
    "zero": "#9F1239",
}

LABELS = {
    "mobilemamba_lite": "LiteSSM-A",
    "mambapsa_cls": "LiteSSM-B",
    "efficientnet_b0": "EfficientNet-B0",
    "lite_freq_net_v2": "LiteFreqNet v2",
    "mobilenet_v3_small": "MobileNetV3-S",
    "shufflenet_v2_x0_5": "ShuffleNet-x0.5",
    "npr": "NPR (ref)",
    "univfd": "UnivFD (ref)",
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


def load_npz(path: Path):
    z = np.load(path, allow_pickle=True)
    return z["probs"].astype(float), z["labels"].astype(int), z["sources"].astype(str)


def load_jsonl(path: Path):
    probs, labels, sources = [], [], []
    with path.open(encoding="utf-8") as f:
        for line in f:
            r = json.loads(line)
            probs.append(float(r["prob"]))
            labels.append(int(r["label"]))
            sources.append(str(r.get("source", "")))
    return np.asarray(probs), np.asarray(labels), np.asarray(sources)


def clf_metrics(probs, labels, thr):
    pred = (probs >= thr).astype(int)
    tp = int(((pred == 1) & (labels == 1)).sum())
    tn = int(((pred == 0) & (labels == 0)).sum())
    fp = int(((pred == 1) & (labels == 0)).sum())
    fn = int(((pred == 0) & (labels == 1)).sum())
    sens = tp / max(tp + fn, 1)
    spec = tn / max(tn + fp, 1)
    bal = 0.5 * (sens + spec)
    prec = tp / max(tp + fp, 1)
    rec = sens
    f1 = 2 * prec * rec / max(prec + rec, 1e-12)
    return {
        "n": int(len(labels)),
        "threshold": float(thr),
        "balanced_acc": round(bal, 3),
        "f1": round(f1, 3),
        "sensitivity": round(sens, 3),
        "specificity": round(spec, 3),
        "accuracy": round((tp + tn) / max(len(labels), 1), 3),
    }


def thresholds():
    thr = {}
    for p in (ASSETS / "metrics").glob("*__metrics.json"):
        key = p.name.replace("__metrics.json", "")
        m = json.loads(p.read_text(encoding="utf-8"))
        if m.get("val_threshold") is not None:
            thr[key] = float(m["val_threshold"])
    recovered = json.loads((ASSETS / "recovered_thresholds.json").read_text(encoding="utf-8"))
    for k, v in recovered.items():
        thr[k] = float(v["val_threshold"])
    # Panel B: fixed 0.5 operating score (not Panel-A Youden)
    thr["npr"] = 0.5
    thr["univfd"] = 0.5
    return thr


def compute_operating_metrics(thr_map):
    rows = {}
    panel_a = ["mobilemamba_lite", "efficientnet_b0", "lite_freq_net_v2"]
    for key in panel_a:
        thr = thr_map[key]
        id_p, id_y, _ = load_npz(ASSETS / "preds" / f"{key}__id.npz")
        u_p, u_y, _ = load_npz(ASSETS / "preds" / f"{key}__ufd.npz")
        rows[LABELS[key]] = {
            "threshold": thr,
            "threshold_source": "validation Youden J",
            "id": clf_metrics(id_p, id_y, thr),
            "ufd_pooled": clf_metrics(u_p, u_y, thr),
        }
    for key in ["npr", "univfd"]:
        thr = thr_map[key]
        id_p, id_y, _ = load_jsonl(ASSETS / "external" / f"{key}_id_test_preds.jsonl")
        u_p, u_y, _ = load_jsonl(ASSETS / "external" / f"{key}_ufd_eval_preds.jsonl")
        rows[LABELS[key]] = {
            "threshold": thr,
            "threshold_source": "fixed score threshold 0.5 (Panel B reference)",
            "id": clf_metrics(id_p, id_y, thr),
            "ufd_pooled": clf_metrics(u_p, u_y, thr),
        }
    out = {
        "note": "Panel A thresholds fitted on validation Youden J and applied unchanged to ID test and UFD. Panel B uses a fixed score threshold of 0.5.",
        "models": rows,
    }
    path = DOCS / "operating_metrics.json"
    path.write_text(json.dumps(out, indent=2), encoding="utf-8")
    print("wrote", path)
    return out


def fig_pareto_inset():
    frozen = json.loads((ROOT / "freeze" / "frozen_numbers.json").read_text(encoding="utf-8"))
    ext = json.loads((ROOT / "external_refs" / "summary.json").read_text(encoding="utf-8"))
    order = [
        "mobilemamba_lite",
        "mambapsa_cls",
        "efficientnet_b0",
        "lite_freq_net_v2",
        "mobilenet_v3_small",
        "shufflenet_v2_x0_5",
    ]
    arch = {
        "mobilemamba_lite": "SSM",
        "mambapsa_cls": "SSM",
        "efficientnet_b0": "CNN",
        "lite_freq_net_v2": "CNN+FFT",
        "mobilenet_v3_small": "CNN",
        "shufflenet_v2_x0_5": "CNN",
    }
    cols = {"SSM": C["ssm"], "CNN": C["cnn"], "CNN+FFT": C["freq"]}

    fig, ax = plt.subplots(figsize=(9.2, 5.9))
    pts = []
    for key in order:
        name = LABELS[key]
        m = frozen["models"][name]
        x, y = float(m["batch1_p50_ms"]), float(m["ufd_macro_auc"])
        s = max(90.0, float(m["params_M"]) * 95)
        ax.scatter([x], [y], s=s, c=cols[arch[key]], alpha=0.88, edgecolors=("black" if name == "LiteSSM-A" else "white"), linewidths=(1.7 if name == "LiteSSM-A" else 0.8), zorder=3)
        pts.append((name, x, y, cols[arch[key]]))

    for key, lab, dxy in [("univfd", "UnivFD (ref)", (8, 8)), ("npr", "NPR (ref)", (8, -12))]:
        rep = ext[key]
        x = float(rep["latency_batch1"]["p50_ms"])
        y = float(rep["splits"]["ufd_eval"]["ufd_macro_auc"])
        ax.scatter([x], [y], s=70, facecolors="none", edgecolors=C["ref"], linewidths=1.5, marker="D", zorder=4)
        ax.annotate(lab, (x, y), textcoords="offset points", xytext=dxy, fontsize=8.2, color=C["ref"])

    offsets = {
        "LiteSSM-A": (16, 10),
        "LiteSSM-B": (16, -16),
        "EfficientNet-B0": (8, 12),
        "LiteFreqNet v2": (8, -14),
        "MobileNetV3-S": (-58, -16),
        "ShuffleNet-x0.5": (8, 10),
    }
    for name, x, y, _ in pts:
        ox, oy = offsets[name]
        ax.annotate(
            name + (" ★" if name == "LiteSSM-A" else ""),
            (x, y),
            textcoords="offset points",
            xytext=(ox, oy),
            fontsize=8.4,
            fontweight=("semibold" if name == "LiteSSM-A" else "regular"),
            color=C["ink"],
            arrowprops=dict(arrowstyle="-", color=C["rule"], lw=0.7) if name.startswith("LiteSSM") or name == "MobileNetV3-S" else None,
        )

    ax.set_xscale("log")
    ax.set_xlim(2.2, 320)
    ax.set_ylim(0.60, 1.02)
    ax.set_xlabel("Batch-1 latency (ms/image, FP32, model-only p50; log scale)")
    ax.set_ylabel("UFD Macro AUC")
    ax.set_title("Accuracy–efficiency operating points", loc="left", fontweight="semibold")
    ax.grid(True, which="major", alpha=0.22, color=C["rule"])
    handles = [
        Line2D([0], [0], marker="o", color="w", markerfacecolor=C["ssm"], markersize=9, label="SSM (Panel A)"),
        Line2D([0], [0], marker="o", color="w", markerfacecolor=C["cnn"], markersize=9, label="CNN (Panel A)"),
        Line2D([0], [0], marker="o", color="w", markerfacecolor=C["freq"], markersize=9, label="CNN+FFT (Panel A)"),
        Line2D([0], [0], marker="D", color=C["ref"], markerfacecolor="none", markersize=7, label="External ref (Panel B)"),
    ]
    ax.legend(handles=handles, loc="lower right", frameon=True, fancybox=False, edgecolor=C["rule"], fontsize=8)
    ax.text(0.02, 0.03, "Marker size ∝ Params (M)\n★ preferred Panel-A operating point", transform=ax.transAxes, fontsize=7.5, color=C["muted"], va="bottom")

    # inset: Panel A only, 0.60-0.80
    axins = inset_axes(ax, width="42%", height="38%", loc="center left", borderpad=1.8)
    for name, x, y, col in pts:
        axins.scatter([x], [y], s=55, c=col, edgecolors=("black" if name == "LiteSSM-A" else "white"), linewidths=0.8)
        if name in ("LiteSSM-A", "LiteSSM-B", "ShuffleNet-x0.5", "EfficientNet-B0"):
            axins.annotate(name.replace("EfficientNet-B0", "Eff-B0").replace("ShuffleNet-x0.5", "Shuffle"), (x, y), textcoords="offset points", xytext=(4, 3), fontsize=6.5, color=C["ink"])
    axins.set_xscale("log")
    axins.set_xlim(3, 300)
    axins.set_ylim(0.60, 0.80)
    axins.set_title("Panel A inset", fontsize=8, pad=3)
    axins.grid(True, alpha=0.2, color=C["rule"])
    axins.tick_params(labelsize=7)

    fig.subplots_adjust(left=0.10, right=0.98, bottom=0.12, top=0.90)
    save(fig, "fig2_pareto_ufd_macro.png")
    plt.close(fig)


def fig_jpeg_forest():
    pkg = json.loads((ROOT / "freeze" / "freeze_package.json").read_text(encoding="utf-8"))
    order = [
        "mobilemamba_lite",
        "mambapsa_cls",
        "efficientnet_b0",
        "lite_freq_net_v2",
        "mobilenet_v3_small",
        "shufflenet_v2_x0_5",
    ]
    ys, centers, los, his, colors = [], [], [], [], []
    for i, key in enumerate(order):
        j = pkg["models"][key]["jpeg"]
        d = float(j["delta_auc"])
        lo, hi = map(float, j["delta_ci95"])
        ys.append(i)
        centers.append(d)
        los.append(d - lo)
        his.append(hi - d)
        colors.append(C["ssm"] if "mamba" in key or "mambapsa" in key else (C["freq"] if "freq" in key else C["cnn"]))

    fig, ax = plt.subplots(figsize=(8.4, 4.6))
    ax.axvline(0.0, color=C["zero"], ls="--", lw=1.2, zorder=1)
    ax.errorbar(centers, ys, xerr=[los, his], fmt="o", color=C["ink"], ecolor=C["muted"], elinewidth=1.2, capsize=3.5, markersize=0, zorder=2)
    for y, c, col in zip(ys, centers, colors):
        ax.scatter([c], [y], s=55, c=col, zorder=3, edgecolors="white", linewidths=0.6)
    ax.set_yticks(ys)
    ax.set_yticklabels([LABELS[k] for k in order])
    ax.set_xlabel(r"$\Delta$AUC $= \mathrm{AUC}_{Q70}-\mathrm{AUC}_{\mathrm{clean}}$ (paired bootstrap 95% CI)")
    ax.set_title("JPEG Q70 effect on ID AUC", loc="left", fontweight="semibold")
    ax.set_xlim(-0.02, 0.02)
    ax.grid(True, axis="x", alpha=0.25, color=C["rule"])
    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)
    ax.text(0.98, 0.05, "All CIs include or hug zero; no frequency-specific robustness win", transform=ax.transAxes, ha="right", fontsize=7.5, color=C["muted"])
    fig.tight_layout()
    save(fig, "fig_jpeg_delta_forest.png")
    plt.close(fig)


def fig_dalle_diagnostic():
    # Panel A models + NPR
    specs = [
        ("mobilemamba_lite", "npz", C["ssm"]),
        ("efficientnet_b0", "npz", C["cnn"]),
        ("npr", "jsonl", C["ref"]),
    ]
    sources = [("ufd_dalle", "DALL·E"), ("ufd_glide_100_10", "Glide 100/10")]
    fig, axes = plt.subplots(len(specs), len(sources), figsize=(9.8, 7.2), sharex=False, sharey=False)
    for r, (key, kind, col) in enumerate(specs):
        if kind == "npz":
            probs, labels, src = load_npz(ASSETS / "preds" / f"{key}__ufd.npz")
        else:
            probs, labels, src = load_jsonl(ASSETS / "external" / f"{key}_ufd_eval_preds.jsonl")
        for c, (scode, stitle) in enumerate(sources):
            ax = axes[r, c]
            m = src == scode
            pr, lab = probs[m], labels[m]
            # histograms of scores for real/fake
            ax.hist(pr[lab == 0], bins=20, range=(0, 1), alpha=0.55, color="#64748B", label="real", density=True)
            ax.hist(pr[lab == 1], bins=20, range=(0, 1), alpha=0.55, color=col, label="fake", density=True)
            ax.set_xlim(0, 1)
            if r == 0:
                ax.set_title(stitle, fontweight="semibold")
            if c == 0:
                ax.set_ylabel(f"{LABELS[key]}\ndensity")
            if r == len(specs) - 1:
                ax.set_xlabel("P(fake)")
            ax.grid(True, axis="y", alpha=0.2, color=C["rule"])
            for spine in ("top", "right"):
                ax.spines[spine].set_visible(False)
            if r == 0 and c == 1:
                ax.legend(fontsize=7.5, frameon=True, fancybox=False, edgecolor=C["rule"])
    fig.suptitle("Score distributions: hard generator (DALL·E) vs easier Glide subset", fontsize=11, fontweight="semibold", y=0.995)
    fig.tight_layout()
    save(fig, "fig_dalle_score_diag.png")
    plt.close(fig)


def sample_sd_seed():
    summary = json.loads((DOCS / "seed_sweep_summary.json").read_text(encoding="utf-8"))
    out = {}
    for model, block in summary.items():
        rows = block["runs"]
        ids = [r["id_auc"] for r in rows]
        ufds = [r["ufd_macro_auc"] for r in rows]
        out[model] = {
            "runs": rows,
            "id_auc_mean": float(np.mean(ids)),
            "id_auc_std_sample": float(np.std(ids, ddof=1)),
            "ufd_macro_mean": float(np.mean(ufds)),
            "ufd_macro_std_sample": float(np.std(ufds, ddof=1)),
        }
    path = DOCS / "seed_sweep_summary_sample_sd.json"
    path.write_text(json.dumps(out, indent=2), encoding="utf-8")
    print("wrote", path, json.dumps(out, indent=2))
    return out


def freq_ablation_table_numbers():
    # from metrics on disk + paper-consistent rounded values
    rows = []
    mapping = [
        ("LiteSSM-A", "mobilemamba_lite", False),
        ("LiteSSM-A + freq", "mobilemamba_lite_freq", True),
        ("LiteSSM-B", "mambapsa_cls", False),
        ("LiteSSM-B + freq", "mambapsa_cls_freq", True),
        ("LiteFreqNet v2", "lite_freq_net_v2", False),
        ("LiteFreqNet (no-freq)", "lite_freq_net_nofreq", True),
    ]
    for display, key, is_abl in mapping:
        m = json.loads((ASSETS / "metrics" / f"{key}__metrics.json").read_text(encoding="utf-8"))
        rows.append(
            {
                "model": display,
                "registry": key,
                "id_auc": round(float(m["test"]["auc"]), 3),
                "ood_pooled_auc": round(float(m["ood"]["auc"]), 3),
                "ufd_macro": None,  # not recomputed for SSM freq / nofreq
            }
        )
    # Fill known main-table UFD Macro only for primary models
    frozen = json.loads((ROOT / "freeze" / "frozen_numbers.json").read_text(encoding="utf-8"))
    rows[0]["ufd_macro"] = frozen["models"]["LiteSSM-A"]["ufd_macro_auc"]
    rows[2]["ufd_macro"] = frozen["models"]["LiteSSM-B"]["ufd_macro_auc"]
    rows[4]["ufd_macro"] = frozen["models"]["LiteFreqNet v2"]["ufd_macro_auc"]
    path = DOCS / "frequency_ablation.json"
    path.write_text(json.dumps({"note": "SSM+freq and LiteFreq no-freq variants report ID and OOD Pooled under the bake-off protocol; UFD Macro was not recomputed for these ablations.", "rows": rows}, indent=2), encoding="utf-8")
    print("wrote", path)
    return rows


def main():
    style()
    thr = thresholds()
    print("thresholds", thr)
    compute_operating_metrics(thr)
    fig_pareto_inset()
    fig_jpeg_forest()
    fig_dalle_diagnostic()
    sample_sd_seed()
    freq_ablation_table_numbers()
    # also keep old jpeg curve? replaced by forest in main text; leave curve in appendix
    print("DONE")


if __name__ == "__main__":
    main()
