#!/bin/bash
set -euo pipefail
cd /root/autodl-tmp/v2_exp/lite_aigc
PY=/root/miniconda3/bin/python
MAN=/root/autodl-tmp/v2_exp/manifests
OUT=/root/autodl-tmp/v2_exp/outputs/pilot_a
LOGDIR=/root/autodl-tmp/v2_exp/logs
mkdir -p "$LOGDIR" "$OUT"
for model in repvit_m0_9 mambaout_proxy efficientnet_v2_s; do
  echo "=== $(date -Is) $model ==="
  "$PY" train.py --model "$model" --manifest-root "$MAN" --out "$OUT" --epochs 15 --batch 64 --lr 1e-4 --size 224 --seed 42 --eval-ood | tee "$LOGDIR/${model}.log"
done
echo DONE > "/root/autodl-tmp/v2_exp/logs/wave1_done.txt"
echo "wave1 complete $(date -Is)"
