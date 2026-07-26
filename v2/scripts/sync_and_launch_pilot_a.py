#!/usr/bin/env python3
"""Sync v2 Pilot A code to AutoDL and launch wave-1 trainings.

Requires env: AUTODL_HOST, AUTODL_PORT, AUTODL_PASS (never commit secrets).
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

import paramiko

HOST = os.environ.get("AUTODL_HOST", "connect.cqa1.seetacloud.com")
PORT = int(os.environ.get("AUTODL_PORT", "30553"))
PASS = os.environ.get("AUTODL_PASS", "")
USER = os.environ.get("AUTODL_USER", "root")

LOCAL_REPO = Path(__file__).resolve().parents[2]
REMOTE_ROOT = "/root/autodl-tmp/v2_exp"
REMOTE_CODE = f"{REMOTE_ROOT}/lite_aigc"
REMOTE_MANIFESTS = f"{REMOTE_ROOT}/manifests"
REMOTE_OUT = f"{REMOTE_ROOT}/outputs/pilot_a"
PY = "/root/miniconda3/bin/python"

WAVE1 = ["repvit_m0_9", "mambaout_proxy", "efficientnet_v2_s"]

UPLOAD_FILES = [
    "train.py",
    "models.py",
    "models_v2.py",
    "data.py",
    "metrics.py",
    "eval_by_source.py",
    "mamba_backbones.py",
    "measure_batch1_latency.py",
]


def connect() -> paramiko.SSHClient:
    if not PASS:
        raise SystemExit("Set AUTODL_PASS")
    c = paramiko.SSHClient()
    c.set_missing_host_key_policy(paramiko.AutoAddPolicy())
    c.connect(HOST, port=PORT, username=USER, password=PASS, timeout=60)
    return c


def run(c: paramiko.SSHClient, cmd: str, timeout: int = 600) -> str:
    print(">>>", cmd, flush=True)
    _, o, e = c.exec_command(cmd, timeout=timeout, get_pty=True)
    out = o.read().decode("utf-8", "replace")
    err = e.read().decode("utf-8", "replace")
    if out:
        print(out, end="" if out.endswith("\n") else "\n")
    if err.strip():
        print("[stderr]", err)
    return out


def sftp_put(c: paramiko.SSHClient, local: Path, remote: str) -> None:
    sftp = c.open_sftp()
    try:
        sftp.put(str(local), remote)
        print(f"upload {local.name} -> {remote}")
    finally:
        sftp.close()


def main() -> None:
    action = sys.argv[1] if len(sys.argv) > 1 else "launch"
    c = connect()
    try:
        if action in ("setup", "launch"):
            run(c, f"mkdir -p {REMOTE_CODE} {REMOTE_MANIFESTS} {REMOTE_OUT} {REMOTE_ROOT}/logs")
            run(
                c,
                "set -e; "
                f"for f in train val test test_ood; do "
                f"ln -sfn /root/autodl-tmp/preflight/manifests/$f.jsonl {REMOTE_MANIFESTS}/$f.jsonl; "
                f"done; "
                f"if [ -f /root/autodl-tmp/outputs/ood_by_source/ufd_eval.jsonl ]; then "
                f"ln -sfn /root/autodl-tmp/outputs/ood_by_source/ufd_eval.jsonl {REMOTE_MANIFESTS}/ufd_eval.jsonl; "
                f"fi; ls -la {REMOTE_MANIFESTS}",
            )
            for name in UPLOAD_FILES:
                lp = LOCAL_REPO / "lite_aigc" / name
                if lp.exists():
                    sftp_put(c, lp, f"{REMOTE_CODE}/{name}")
            run(c, f"{PY} -m pip install -q -U 'timm>=0.9' 2>&1 | tail -8")
            run(
                c,
                f"cd {REMOTE_CODE}; {PY} -c \""
                "from models import build_model; "
                "m=build_model('repvit_m0_9'); "
                "print('repvit ok', sum(p.numel() for p in m.parameters())); "
                "m2=build_model('mambaout_proxy'); "
                "print('proxy ok', sum(p.numel() for p in m2.parameters())); "
                "m3=build_model('efficientnet_v2_s'); "
                "print('effv2 ok', sum(p.numel() for p in m3.parameters()))\"",
            )

        if action == "launch":
            # write queue script remotely
            lines = [
                "#!/bin/bash",
                "set -euo pipefail",
                f"cd {REMOTE_CODE}",
                f"PY={PY}",
                f"MAN={REMOTE_MANIFESTS}",
                f"OUT={REMOTE_OUT}",
                f"LOGDIR={REMOTE_ROOT}/logs",
                f"mkdir -p \"$LOGDIR\" \"$OUT\"",
                "for model in " + " ".join(WAVE1) + "; do",
                '  echo "=== $(date -Is) $model ==="',
                '  "$PY" train.py --model "$model" --manifest-root "$MAN" --out "$OUT" '
                "--epochs 15 --batch 64 --lr 1e-4 --size 224 --seed 42 --eval-ood "
                '| tee "$LOGDIR/${model}.log"',
                "done",
                f'echo DONE > "{REMOTE_ROOT}/logs/wave1_done.txt"',
                'echo "wave1 complete $(date -Is)"',
            ]
            # upload via sftp as text file
            local_q = LOCAL_REPO / "v2" / "scripts" / "_remote_wave1_queue.sh"
            local_q.write_text("\n".join(lines) + "\n", encoding="utf-8", newline="\n")
            sftp_put(c, local_q, f"{REMOTE_ROOT}/run_wave1_queue.sh")
            run(c, f"chmod +x {REMOTE_ROOT}/run_wave1_queue.sh")
            # kill stale queue? don't kill other users' jobs — only start if idle
            out = run(c, "pgrep -af 'train.py --model' || true")
            if "train.py --model" in out:
                print("A train.py is already running; queue script uploaded but not started.")
            else:
                run(
                    c,
                    f"nohup bash {REMOTE_ROOT}/run_wave1_queue.sh "
                    f"> {REMOTE_ROOT}/logs/queue.log 2>&1 & echo QUEUE_PID=$!; sleep 2; "
                    f"pgrep -af 'train.py|run_wave1' || true; "
                    f"tail -n 20 {REMOTE_ROOT}/logs/queue.log || true",
                )
            print("WAVE1:", WAVE1)
            print(f"Remote: {REMOTE_ROOT}")
        elif action == "status":
            run(c, "pgrep -af 'train.py|run_wave1' || echo idle")
            run(c, f"tail -n 15 {REMOTE_ROOT}/logs/queue.log 2>/dev/null || true")
            run(c, f"tail -n 5 {REMOTE_ROOT}/logs/*.log 2>/dev/null || true")
            run(c, f"find {REMOTE_OUT} -name metrics.json 2>/dev/null")
        else:
            raise SystemExit("usage: setup|launch|status")
    finally:
        c.close()


if __name__ == "__main__":
    main()
