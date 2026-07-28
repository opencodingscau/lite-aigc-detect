#!/usr/bin/env python3
"""Panel A probability-mean ensembles from frozen prediction files (zero retraining).

Protocol (locked):
  - Equal-weight probability mean; fixed members; no UFD/test-set weight tuning.
  - Thresholds for majority/double-fault use validation Youden (or disclosed 0.5).
  - Does not alter freeze/frozen_numbers.json or preferred operating point.
"""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
ASSETS = Path(r"E:\sciencecre\aigc_datasets\formal\_paper_assets")
DOCS = ROOT / "docs"
OUT_DIRS = [ROOT / "latex" / "figures", ROOT / "freeze" / "figures"]

KEYS = [
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
ENSEMBLES = {
    "E_all6_mean": KEYS,
    "E_ssm_cnn_mean": ["mobilemamba_lite", "efficientnet_b0"],
    "E_top3_mean": ["mobilemamba_lite", "mambapsa_cls", "efficientnet_b0"],
    "E_ssm_pair_mean": ["mobilemamba_lite", "mambapsa_cls"],
}


def load_npz(path: Path):
    z = np.load(path, allow_pickle=True)
    return z["probs"].astype(float), z["labels"].astype(int), z["sources"].astype(str)


def auc_safe(y: np.ndarray, p: np.ndarray) -> float:
    y = np.asarray(y).astype(int)
    p = np.asarray(p).astype(float)
    if len(np.unique(y)) < 2:
        return float("nan")
    order = np.argsort(-p)
    y_s = y[order]
    n_pos = float((y == 1).sum())
    n_neg = float((y == 0).sum())
    if n_pos == 0 or n_neg == 0:
        return float("nan")
    tps = np.cumsum(y_s == 1)
    fps = np.cumsum(y_s == 0)
    tpr = np.concatenate([[0.0], tps / n_pos, [1.0]])
    fpr = np.concatenate([[0.0], fps / n_neg, [1.0]])
    return float(np.trapz(tpr, fpr))


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
    for k in ("mobilenet_v3_small", "shufflenet_v2_x0_5"):
        thr.setdefault(k, 0.5)
    return thr


def macro_worst(probs: np.ndarray, labels: np.ndarray, sources: np.ndarray):
    by_source = {}
    for s in sorted(set(sources.tolist())):
        m = sources == s
        by_source[s] = auc_safe(labels[m], probs[m])
    vals = [v for v in by_source.values() if np.isfinite(v)]
    macro = float(np.mean(vals)) if vals else float("nan")
    worst = float(np.min(vals)) if vals else float("nan")
    dalle = by_source.get("ufd_dalle", by_source.get("dalle", worst))
    return macro, worst, float(dalle), by_source


def double_fault(a_err: np.ndarray, b_err: np.ndarray):
    both = a_err & b_err
    union = a_err | b_err
    return {
        "double_fault_rate": float(both.mean()),
        "error_jaccard": float(both.sum() / max(union.sum(), 1)),
    }


def main():
    thr_map = thresholds()
    id_packs = {k: load_npz(ASSETS / "preds" / f"{k}__id.npz") for k in KEYS}
    ufd_packs = {k: load_npz(ASSETS / "preds" / f"{k}__ufd.npz") for k in KEYS}

    singles = {}
    for k in KEYS:
        p, y, s = ufd_packs[k]
        pid, yid, _ = id_packs[k]
        macro, worst, dalle, by_source = macro_worst(p, y, s)
        singles[LABELS[k]] = {
            "id_auc": auc_safe(yid, pid),
            "ufd_macro": macro,
            "worst": worst,
            "dalle": dalle,
            "by_source": by_source,
        }

    ensembles_prob_mean = {}
    ensembles_majority = {}
    for name, members in ENSEMBLES.items():
        p_id = np.mean([id_packs[m][0] for m in members], axis=0)
        y_id = id_packs[members[0]][1]
        p_u = np.mean([ufd_packs[m][0] for m in members], axis=0)
        y_u = ufd_packs[members[0]][1]
        s_u = ufd_packs[members[0]][2]
        macro, worst, dalle, _ = macro_worst(p_u, y_u, s_u)
        ensembles_prob_mean[name] = {
            "members": [LABELS[m] for m in members],
            "id_auc": auc_safe(y_id, p_id),
            "ufd_macro": macro,
            "worst": worst,
            "dalle": dalle,
        }
        # majority vote uses per-member locked thresholds (not tuned on test)
        votes_id = np.stack([(id_packs[m][0] >= thr_map[m]).astype(int) for m in members], axis=0)
        votes_u = np.stack([(ufd_packs[m][0] >= thr_map[m]).astype(int) for m in members], axis=0)
        maj_id = (votes_id.mean(axis=0) >= 0.5).astype(int)
        maj_u = (votes_u.mean(axis=0) >= 0.5).astype(int)
        ensembles_majority[name] = {
            "id_acc": float((maj_id == y_id).mean()),
            "ufd_acc": float((maj_u == y_u).mean()),
        }

    a_key = "mobilemamba_lite"
    a_pred = (ufd_packs[a_key][0] >= thr_map[a_key]).astype(int)
    a_err = a_pred != ufd_packs[a_key][1]
    df = {}
    for k in KEYS:
        if k == a_key:
            continue
        pred = (ufd_packs[k][0] >= thr_map[k]).astype(int)
        err = pred != ufd_packs[k][1]
        df[LABELS[k]] = double_fault(a_err, err)

    out = {
        "note": (
            "Computed from frozen formal/_paper_assets preds; equal-weight probability mean; "
            "no UFD/test-set weight tuning; does not alter freeze/frozen_numbers.json"
        ),
        "singles": singles,
        "ensembles_prob_mean": ensembles_prob_mean,
        "ensembles_majority": ensembles_majority,
        "double_fault_vs_litessm_a_ufd": df,
        "thresholds_used": {LABELS[k]: thr_map[k] for k in KEYS},
    }
    DOCS.mkdir(parents=True, exist_ok=True)
    path = DOCS / "ensemble_panel_a.json"
    path.write_text(json.dumps(out, indent=2), encoding="utf-8")
    print("wrote", path)

    # bar figure
    names = [LABELS[k] for k in KEYS] + [
        r"$E_{\mathrm{all6}}$",
        r"$E_{\mathrm{A+Eff}}$",
        r"$E_{\mathrm{top3}}$",
        r"$E_{\mathrm{A+B}}$",
    ]
    macros = [singles[LABELS[k]]["ufd_macro"] for k in KEYS] + [
        ensembles_prob_mean["E_all6_mean"]["ufd_macro"],
        ensembles_prob_mean["E_ssm_cnn_mean"]["ufd_macro"],
        ensembles_prob_mean["E_top3_mean"]["ufd_macro"],
        ensembles_prob_mean["E_ssm_pair_mean"]["ufd_macro"],
    ]
    colors = ["#0F766E"] + ["#57534E"] * 5 + ["#1D4ED8"] * 4
    fig, ax = plt.subplots(figsize=(8.2, 3.6))
    ax.bar(range(len(names)), macros, color=colors, edgecolor="white", linewidth=0.4)
    ax.axhline(singles["LiteSSM-A"]["ufd_macro"], color="#0F766E", ls="--", lw=1.2, label="LiteSSM-A")
    ax.set_xticks(range(len(names)))
    ax.set_xticklabels(names, rotation=28, ha="right")
    ax.set_ylabel("UFD Macro AUC")
    ax.set_ylim(0.60, 0.74)
    ax.set_title("Panel A singles vs probability-mean ensembles (frozen preds)")
    ax.legend(frameon=False, loc="upper right")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    fig.tight_layout()
    for d in OUT_DIRS:
        d.mkdir(parents=True, exist_ok=True)
        fig.savefig(d / "fig_ensemble_ufd_macro.png", dpi=300, bbox_inches="tight", pad_inches=0.08)
        print("wrote", d / "fig_ensemble_ufd_macro.png")
    plt.close(fig)


if __name__ == "__main__":
    main()
