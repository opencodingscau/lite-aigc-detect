#!/usr/bin/env python3
"""Upload Pilot B scripts and launch queue on AutoDL."""
from __future__ import annotations

import os
from pathlib import Path

import paramiko

HOST = os.environ.get("AUTODL_HOST", "connect.cqa1.seetacloud.com")
PORT = int(os.environ.get("AUTODL_PORT", "30553"))
PASS = os.environ.get("AUTODL_PASS", "")
LOCAL = Path(__file__).resolve().parents[2]
REMOTE = "/root/autodl-tmp/v2_exp"


def main():
    if not PASS:
        raise SystemExit("Set AUTODL_PASS")
    c = paramiko.SSHClient()
    c.set_missing_host_key_policy(paramiko.AutoAddPolicy())
    c.connect(HOST, port=PORT, username="root", password=PASS, timeout=60)

    def run(cmd, timeout=600):
        print(">>>", cmd[:200], flush=True)
        _, o, e = c.exec_command(cmd, timeout=timeout, get_pty=True)
        out = o.read().decode("utf-8", "replace")
        print(out[-2000:] if len(out) > 2000 else out)
        return out

    run(f"mkdir -p {REMOTE}/scripts {REMOTE}/lite_aigc {REMOTE}/logs/pilot_b {REMOTE}/outputs/pilot_b")
    sftp = c.open_sftp()
    uploads = [
        ("v2/scripts/export_teacher_softlabels.py", f"{REMOTE}/scripts/export_teacher_softlabels.py"),
        ("v2/scripts/train_gated_distill.py", f"{REMOTE}/scripts/train_gated_distill.py"),
        ("v2/scripts/run_pilot_b_queue.sh", f"{REMOTE}/scripts/run_pilot_b_queue.sh"),
        ("lite_aigc/eval_external_refs.py", f"{REMOTE}/lite_aigc/eval_external_refs.py"),
        ("lite_aigc/models.py", f"{REMOTE}/lite_aigc/models.py"),
        ("lite_aigc/models_v2.py", f"{REMOTE}/lite_aigc/models_v2.py"),
        ("lite_aigc/mamba_backbones.py", f"{REMOTE}/lite_aigc/mamba_backbones.py"),
        ("lite_aigc/data.py", f"{REMOTE}/lite_aigc/data.py"),
        ("lite_aigc/train.py", f"{REMOTE}/lite_aigc/train.py"),
        ("lite_aigc/metrics.py", f"{REMOTE}/lite_aigc/metrics.py"),
    ]
    for rel, rem in uploads:
        lp = LOCAL / rel
        if lp.exists():
            sftp.put(str(lp), rem)
            print("upload", rel)
    sftp.close()

    # ensure pool + manifests
    run(
        "set -e; "
        f"test -f {REMOTE}/manifests/pilot_b/distill_pool.jsonl; "
        f"for f in train val test test_ood; do "
        f"ln -sfn /root/autodl-tmp/preflight/manifests/$f.jsonl {REMOTE}/manifests/$f.jsonl; done; "
        f"chmod +x {REMOTE}/scripts/run_pilot_b_queue.sh"
    )

    busy = run("pgrep -af 'train_gated_distill|export_teacher|run_pilot_b' || echo idle")
    if "train_gated_distill" in busy and "miniconda" in busy:
        print("Pilot B already running")
    else:
        run(
            f"nohup bash {REMOTE}/scripts/run_pilot_b_queue.sh > {REMOTE}/logs/pilot_b/queue.log 2>&1 & echo PID=$!; "
            f"sleep 6; pgrep -af 'export_teacher|train_gated|run_pilot_b' || true; "
            f"tail -n 40 {REMOTE}/logs/pilot_b/queue.log || true"
        )
    c.close()


if __name__ == "__main__":
    main()
