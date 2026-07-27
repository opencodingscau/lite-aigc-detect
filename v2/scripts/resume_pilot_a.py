#!/usr/bin/env python3
"""Inspect / resume Pilot A on AutoDL after instance reboot."""
from __future__ import annotations

import json
import os
import sys

import paramiko

HOST = os.environ.get("AUTODL_HOST", "connect.cqa1.seetacloud.com")
PORT = int(os.environ.get("AUTODL_PORT", "30553"))
PASS = os.environ.get("AUTODL_PASS", "")
USER = os.environ.get("AUTODL_USER", "root")
REMOTE_ROOT = "/root/autodl-tmp/v2_exp"
WAVE1 = ["repvit_m0_9", "mambaout_proxy", "efficientnet_v2_s"]
PY = "/root/miniconda3/bin/python"


def connect():
    if not PASS:
        raise SystemExit("Set AUTODL_PASS")
    c = paramiko.SSHClient()
    c.set_missing_host_key_policy(paramiko.AutoAddPolicy())
    c.connect(HOST, port=PORT, username=USER, password=PASS, timeout=60)
    return c


def run(c, cmd, timeout=300):
    print(">>>", cmd, flush=True)
    _, o, e = c.exec_command(cmd, timeout=timeout, get_pty=True)
    out = o.read().decode("utf-8", "replace")
    err = e.read().decode("utf-8", "replace")
    if out:
        print(out, end="" if out.endswith("\n") else "\n")
    if err.strip():
        print("[stderr]", err)
    return out


def metrics_summary(c):
    out = run(
        c,
        f"find {REMOTE_ROOT}/outputs/pilot_a -name metrics.json 2>/dev/null",
    )
    paths = [ln.strip() for ln in out.splitlines() if ln.strip().endswith("metrics.json")]
    for p in paths:
        raw = run(c, f"cat {p}")
        try:
            m = json.loads(raw)
            print(
                f"DONE {m.get('model')}: test_auc={m.get('test',{}).get('auc')} "
                f"ood_auc={m.get('ood',{}).get('auc')} params_M={m.get('params_M')}"
            )
        except Exception as e:  # noqa: BLE001
            print("parse fail", p, e)
    return {p.split("/")[-2] for p in paths}


def main():
    action = sys.argv[1] if len(sys.argv) > 1 else "resume"
    c = connect()
    try:
        run(c, "uname -a; nvidia-smi -L; df -h /root/autodl-tmp | tail -1")
        run(c, f"ls -la {REMOTE_ROOT} || echo NO_V2_EXP")
        run(c, f"ls -la {REMOTE_ROOT}/outputs/pilot_a 2>/dev/null || echo no_outputs")
        run(c, f"ls -la {REMOTE_ROOT}/logs 2>/dev/null || echo no_logs")
        run(c, "pgrep -af 'train.py|run_wave1' || echo idle")
        run(c, f"tail -n 30 {REMOTE_ROOT}/logs/queue.log 2>/dev/null || true")
        for m in WAVE1:
            run(c, f"echo ==== {m} ====; tail -n 12 {REMOTE_ROOT}/logs/{m}.log 2>/dev/null || echo missing")
        done = metrics_summary(c)
        print("completed models:", sorted(done))
        pending = [m for m in WAVE1 if m not in done]
        print("pending:", pending)

        if action == "status":
            return

        # ensure code/manifests still there
        run(
            c,
            "set -e; "
            f"mkdir -p {REMOTE_ROOT}/lite_aigc {REMOTE_ROOT}/manifests {REMOTE_ROOT}/outputs/pilot_a {REMOTE_ROOT}/logs; "
            f"for f in train val test test_ood; do "
            f"ln -sfn /root/autodl-tmp/preflight/manifests/$f.jsonl {REMOTE_ROOT}/manifests/$f.jsonl; done; "
            f"if [ -f /root/autodl-tmp/outputs/ood_by_source/ufd_eval.jsonl ]; then "
            f"ln -sfn /root/autodl-tmp/outputs/ood_by_source/ufd_eval.jsonl {REMOTE_ROOT}/manifests/ufd_eval.jsonl; fi; "
            f"test -f {REMOTE_ROOT}/lite_aigc/train.py && test -f {REMOTE_ROOT}/lite_aigc/models_v2.py && echo CODE_OK",
        )

        busy = run(c, "pgrep -f '/train.py --model' || true")
        if "train.py --model" in busy and "pgrep" not in busy.split("train.py --model")[0][-40:]:
            # crude: if real python train running
            if "/root/miniconda3/bin/python train.py" in busy or "train.py --model" in busy:
                print("Training already running; not starting another queue.")
                return

        if not pending:
            print("Wave-1 already complete.")
            return

        # rewrite resume queue for pending only
        lines = [
            "#!/bin/bash",
            "set -euo pipefail",
            f"cd {REMOTE_ROOT}/lite_aigc",
            f"PY={PY}",
            f"MAN={REMOTE_ROOT}/manifests",
            f"OUT={REMOTE_ROOT}/outputs/pilot_a",
            f"LOGDIR={REMOTE_ROOT}/logs",
            "mkdir -p \"$LOGDIR\" \"$OUT\"",
            "for model in " + " ".join(pending) + "; do",
            '  if [ -f "$OUT/$model/metrics.json" ]; then echo "skip done $model"; continue; fi',
            '  echo "=== $(date -Is) $model ==="',
            '  "$PY" train.py --model "$model" --manifest-root "$MAN" --out "$OUT" '
            "--epochs 15 --batch 64 --lr 1e-4 --size 224 --seed 42 --eval-ood "
            '| tee -a "$LOGDIR/${model}.log"',
            "done",
            f'echo DONE > "{REMOTE_ROOT}/logs/wave1_done.txt"',
            'echo "wave1 resume complete $(date -Is)"',
        ]
        sftp = c.open_sftp()
        with sftp.file(f"{REMOTE_ROOT}/run_wave1_resume.sh", "w") as f:
            f.write("\n".join(lines) + "\n")
        sftp.close()
        run(c, f"chmod +x {REMOTE_ROOT}/run_wave1_resume.sh")
        run(
            c,
            f"nohup bash {REMOTE_ROOT}/run_wave1_resume.sh > {REMOTE_ROOT}/logs/queue_resume.log 2>&1 & echo RESUME_PID=$!; "
            f"sleep 4; pgrep -af 'train.py --model' || true; tail -n 20 {REMOTE_ROOT}/logs/queue_resume.log || true",
        )
    finally:
        c.close()


if __name__ == "__main__":
    main()
