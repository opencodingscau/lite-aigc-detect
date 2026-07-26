#!/usr/bin/env python3
"""Cold-start table rebuild from the freeze package (no GPU required).

Example::

    python scripts/build_tables.py \\
      --freeze-package freeze/freeze_package.json \\
      --latency-summary latency_batch1/summary.json \\
      --external-summary external_refs/summary.json \\
      --output-dir reproduced_tables

Writes:
  table2_overall.csv
  table3_domain.csv
  table4_ufd_sources.csv
  table5_jpeg.csv
  verification_report.json
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import subprocess
from datetime import datetime, timezone
from pathlib import Path

ORDER = [
    "mobilemamba_lite",
    "mambapsa_cls",
    "efficientnet_b0",
    "lite_freq_net_v2",
    "mobilenet_v3_small",
    "shufflenet_v2_x0_5",
]
PAPER_NAME = {
    "mobilemamba_lite": "LiteSSM-A",
    "mambapsa_cls": "LiteSSM-B",
    "efficientnet_b0": "EfficientNet-B0",
    "lite_freq_net_v2": "LiteFreqNet v2",
    "mobilenet_v3_small": "MobileNetV3-S",
    "shufflenet_v2_x0_5": "ShuffleNet-x0.5",
}
GENS = [
    "ufd_dalle",
    "ufd_glide_100_10",
    "ufd_glide_100_27",
    "ufd_glide_50_27",
    "ufd_guided",
    "ufd_ldm_100",
    "ufd_ldm_200",
    "ufd_ldm_200_cfg",
]

# Locked Panel-A point estimates used for verification (3 decimals unless noted)
EXPECTED = {
    "mobilemamba_lite": {"id": 0.946, "ufd_macro": 0.718, "bedroom": 0.812, "celebahq": 0.999, "worst": 0.504},
    "mambapsa_cls": {"id": 0.945, "ufd_macro": 0.700, "bedroom": 0.815, "celebahq": 0.998, "worst": 0.483},
    "efficientnet_b0": {"id": 0.929, "ufd_macro": 0.667, "bedroom": 0.777, "celebahq": 0.996, "worst": 0.519},
    "lite_freq_net_v2": {"id": 0.906, "ufd_macro": 0.645, "bedroom": 0.713, "celebahq": 0.998, "worst": 0.451},
    "mobilenet_v3_small": {"id": 0.890, "ufd_macro": 0.636, "bedroom": 0.688, "celebahq": 0.999, "worst": 0.455},
    "shufflenet_v2_x0_5": {"id": 0.880, "ufd_macro": 0.674, "bedroom": 0.677, "celebahq": 0.992, "worst": 0.467},
}
TOL = 5e-4  # absolute error after rounding to 3 decimals


def sha256_file(path: Path | None) -> str | None:
    if path is None or not path.exists():
        return None
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def git_commit(repo: Path) -> str | None:
    try:
        out = subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=str(repo), stderr=subprocess.DEVNULL, text=True
        )
        return out.strip()
    except Exception:
        return None


def write_csv(path: Path, rows: list[dict], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    # Force LF so hashes/SHA256SUMS stay stable across platforms (.gitattributes eol=lf).
    with open(path, "w", encoding="utf-8", newline="\n") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames, lineterminator="\n")
        w.writeheader()
        for r in rows:
            w.writerow(r)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--freeze-package", type=Path, default=Path("freeze/freeze_package.json"))
    ap.add_argument("--latency-summary", type=Path, default=Path("latency_batch1/summary.json"))
    ap.add_argument("--external-summary", type=Path, default=Path("external_refs/summary.json"))
    ap.add_argument("--output-dir", type=Path, default=Path("reproduced_tables"))
    ap.add_argument("--tol", type=float, default=TOL)
    args = ap.parse_args()

    pkg = json.loads(args.freeze_package.read_text(encoding="utf-8"))
    lat = json.loads(args.latency_summary.read_text(encoding="utf-8")) if args.latency_summary.exists() else {}
    ext = json.loads(args.external_summary.read_text(encoding="utf-8")) if args.external_summary.exists() else {}

    out = args.output_dir
    out.mkdir(parents=True, exist_ok=True)

    # Table 2 Panel A
    t2 = []
    for n in ORDER:
        m = pkg["models"][n]
        row = {
            "model": PAPER_NAME[n],
            "registry_key": n,
            "arch": m.get("arch"),
            "params_M": m["efficiency"].get("params_M"),
            "flops_G": m["efficiency"].get("flops_G"),
            "batch1_p50_ms": None if n not in lat else round(lat[n]["batch1_latency"]["p50_ms"], 2),
            "batch1_p95_ms": None if n not in lat else round(lat[n]["batch1_latency"]["p95_ms"], 2),
            "thr_bs32": None if n not in lat else round(lat[n]["fps_bs32"], 1),
            "id_auc": round(m["id"]["auc"], 3),
            "ufd_macro": round(m["ufd"]["macro_auc"], 3),
            "worst_auc": round(m["ufd"]["worst_auc"], 3),
            "ood_pooled": round(m["ood_pooled"]["auc"], 3),
        }
        t2.append(row)
    # Panel B refs
    for key, label in (("univfd", "UnivFD (Panel B ref)"), ("npr", "NPR (Panel B ref)")):
        if key not in ext:
            continue
        rep = ext[key]
        t2.append(
            {
                "model": label,
                "registry_key": key,
                "arch": "REF",
                "params_M": round(rep["meta"]["params"] / 1e6, 3),
                "flops_G": None,
                "batch1_p50_ms": round(rep["latency_batch1"]["p50_ms"], 2),
                "batch1_p95_ms": round(rep["latency_batch1"]["p95_ms"], 2),
                "thr_bs32": None,
                "id_auc": round(rep["splits"]["id_test"]["id_auc"], 3),
                "ufd_macro": round(rep["splits"]["ufd_eval"]["ufd_macro_auc"], 3),
                "worst_auc": round(rep["splits"]["ufd_eval"]["worst_auc"], 3),
                "ood_pooled": round(rep["splits"]["ufd_eval"]["pooled_auc"], 3),
            }
        )
    write_csv(
        out / "table2_overall.csv",
        t2,
        [
            "model",
            "registry_key",
            "arch",
            "params_M",
            "flops_G",
            "batch1_p50_ms",
            "batch1_p95_ms",
            "thr_bs32",
            "id_auc",
            "ufd_macro",
            "worst_auc",
            "ood_pooled",
        ],
    )

    # Table 3 domain
    t3 = []
    for n in ORDER:
        m = pkg["models"][n]
        t3.append(
            {
                "model": PAPER_NAME[n],
                "id_auc": round(m["id"]["auc"], 3),
                "celebahq": round(m["domain_celebahq"]["auc"], 3),
                "bedroom": round(m["domain_bedroom"]["auc"], 3),
                "gap": round(m["domain_gap"], 3),
            }
        )
    write_csv(out / "table3_domain.csv", t3, ["model", "id_auc", "celebahq", "bedroom", "gap"])

    # Table 4 UFD sources
    t4 = []
    for n in ORDER:
        m = pkg["models"][n]
        per = m["ufd"]["per_generator"]
        row = {"model": PAPER_NAME[n], "macro": round(m["ufd"]["macro_auc"], 3), "min": round(m["ufd"]["worst_auc"], 3)}
        for g in GENS:
            row[g] = round(per[g]["auc"], 3)
        t4.append(row)
    write_csv(out / "table4_ufd_sources.csv", t4, ["model"] + GENS + ["macro", "min"])

    # Table 5 JPEG
    t5 = []
    for n in ORDER:
        m = pkg["models"][n]
        j = m["jpeg"]
        t5.append(
            {
                "model": PAPER_NAME[n],
                "clean": round(j["clean"]["auc"], 3),
                "q70": round(j["q70"]["auc"], 3),
                "delta": round(j["delta_auc"], 4),
                "ci_lo": round(j["delta_ci95"][0], 4),
                "ci_hi": round(j["delta_ci95"][1], 4),
            }
        )
    write_csv(out / "table5_jpeg.csv", t5, ["model", "clean", "q70", "delta", "ci_lo", "ci_hi"])

    # Verification vs EXPECTED
    diffs = []
    max_abs = 0.0
    for n in ORDER:
        m = pkg["models"][n]
        got = {
            "id": round(m["id"]["auc"], 3),
            "ufd_macro": round(m["ufd"]["macro_auc"], 3),
            "bedroom": round(m["domain_bedroom"]["auc"], 3),
            "celebahq": round(m["domain_celebahq"]["auc"], 3),
            "worst": round(m["ufd"]["worst_auc"], 3),
        }
        for k, exp in EXPECTED[n].items():
            err = abs(got[k] - exp)
            max_abs = max(max_abs, err)
            diffs.append(
                {
                    "model": n,
                    "metric": k,
                    "expected": exp,
                    "got": got[k],
                    "abs_err": err,
                    "pass": err <= args.tol,
                }
            )

    report = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "git_commit": git_commit(Path.cwd()),
        "freeze_package": str(args.freeze_package),
        "freeze_package_sha256": sha256_file(args.freeze_package),
        "latency_summary_sha256": sha256_file(args.latency_summary if args.latency_summary.exists() else None),
        "external_summary_sha256": sha256_file(args.external_summary if args.external_summary.exists() else None),
        "tol": args.tol,
        "max_abs_err": max_abs,
        "all_pass": all(d["pass"] for d in diffs),
        "diffs": diffs,
        "notes": [
            "AUC values verified against locked 3-decimal manuscript targets.",
            "batch-1 latency / Panel B rows come from companion JSON summaries when present.",
            "FPS@bs32 in freeze_package.json may differ from latency_batch1/summary.json; prefer the locked batch-1 session for latency claims.",
        ],
    }
    (out / "verification_report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    status = "PASS" if report["all_pass"] else "FAIL"
    print(f"[{status}] wrote {out}  max_abs_err={max_abs:.6g}")
    if not report["all_pass"]:
        for d in diffs:
            if not d["pass"]:
                print(" mismatch", d)
        raise SystemExit(2)


if __name__ == "__main__":
    main()
