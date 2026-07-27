#!/usr/bin/env python3
"""Pull Pilot A metrics + run UFD Macro eval (eval_by_source) on AutoDL."""
from __future__ import annotations

import json
import os
from pathlib import Path

import paramiko

HOST = os.environ.get("AUTODL_HOST", "connect.cqa1.seetacloud.com")
PORT = int(os.environ.get("AUTODL_PORT", "30553"))
PASS = os.environ.get("AUTODL_PASS", "")
REMOTE = "/root/autodl-tmp/v2_exp"
PY = "/root/miniconda3/bin/python"
LOCAL = Path(__file__).resolve().parents[2] / "v2" / "results" / "pilot_a"
MODELS = ["repvit_m0_9", "mambaout_proxy", "efficientnet_v2_s"]
LITESSM_CKPT = "/root/autodl-tmp/outputs/bakeoff/mobilemamba_lite/best.pt"


def connect():
    c = paramiko.SSHClient()
    c.set_missing_host_key_policy(paramiko.AutoAddPolicy())
    c.connect(HOST, port=PORT, username="root", password=PASS, timeout=60)
    return c


def run(c, cmd, timeout=600):
    print(">>>", cmd[:220], flush=True)
    _, o, e = c.exec_command(cmd, timeout=timeout, get_pty=True)
    out = o.read().decode("utf-8", "replace")
    if out:
        print(out[-2500:] if len(out) > 2500 else out)
    return out


def main():
    if not PASS:
        raise SystemExit("Set AUTODL_PASS")
    LOCAL.mkdir(parents=True, exist_ok=True)
    c = connect()
    try:
        sftp = c.open_sftp()
        for m in MODELS:
            rp = f"{REMOTE}/outputs/pilot_a/{m}/metrics.json"
            lp = LOCAL / f"{m}_metrics.json"
            sftp.get(rp, str(lp))
            print("got", lp.name)
        sftp.close()

        ufd = run(
            c,
            "find /root/autodl-tmp -maxdepth 4 -type d -name UniversalFakeDetect 2>/dev/null | head -5",
        )
        ufd_roots = [ln.strip() for ln in ufd.splitlines() if "UniversalFakeDetect" in ln]
        if not ufd_roots:
            raise SystemExit("UniversalFakeDetect not found on remote")
        ufd_root = ufd_roots[0]
        print("UFD root:", ufd_root)

        # ensure manifests linked
        run(
            c,
            "set -e; "
            f"mkdir -p {REMOTE}/manifests {REMOTE}/outputs/pilot_a/ufd_macro {REMOTE}/logs; "
            f"for f in train val test test_ood; do "
            f"ln -sfn /root/autodl-tmp/preflight/manifests/$f.jsonl {REMOTE}/manifests/$f.jsonl; done",
        )

        jobs = [(m, f"{REMOTE}/outputs/pilot_a/{m}/best.pt") for m in MODELS]
        jobs.append(("mobilemamba_lite", LITESSM_CKPT))

        lines = [
            "#!/bin/bash",
            "set -euo pipefail",
            f"cd {REMOTE}/lite_aigc",
            f"PY={PY}",
            f"MAN={REMOTE}/manifests",
            f"UFD={ufd_root}",
            f"OUT={REMOTE}/outputs/pilot_a/ufd_macro",
            "mkdir -p \"$OUT\"",
        ]
        for name, ckpt in jobs:
            out_json = f"$OUT/{name}_ood_by_source.json"
            lines += [
                f'echo "=== UFD $(date -Is) {name} ==="',
                f'if [ -f "{REMOTE}/outputs/pilot_a/ufd_macro/{name}_ood_by_source.json" ]; then echo skip {name}; else',
                f'  "$PY" eval_by_source.py --model {name} --ckpt {ckpt} '
                f'--manifest-root "$MAN" --ufd-root "$UFD" --out "$OUT" '
                f'2>&1 | tee -a {REMOTE}/logs/ufd_{name}.log',
                # eval writes to out/{model}_ood_by_source.json
                "fi",
            ]
        lines.append(f"echo DONE > {REMOTE}/logs/ufd_eval_done.txt")
        sftp = c.open_sftp()
        with sftp.file(f"{REMOTE}/run_ufd_macro.sh", "w") as f:
            f.write("\n".join(lines) + "\n")
        sftp.close()
        run(c, f"chmod +x {REMOTE}/run_ufd_macro.sh")

        busy = run(c, "pgrep -af 'eval_by_source.py' || echo idle")
        if "eval_by_source.py" in busy and "miniconda" in busy:
            print("eval already running")
        else:
            run(
                c,
                f"nohup bash {REMOTE}/run_ufd_macro.sh > {REMOTE}/logs/ufd_queue.log 2>&1 & echo UFD_PID=$!; "
                f"sleep 5; pgrep -af eval_by_source || true; tail -n 40 {REMOTE}/logs/ufd_queue.log || true",
            )
    finally:
        c.close()


if __name__ == "__main__":
    main()
