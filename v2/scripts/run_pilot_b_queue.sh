#!/bin/bash
# Pilot B on AutoDL: softlabels -> gated distill matrix
set -euo pipefail
ROOT=/root/autodl-tmp/v2_exp
PY=/root/miniconda3/bin/python
CODE=$ROOT/lite_aigc
SCR=$ROOT/scripts
LOG=$ROOT/logs/pilot_b
OUT=$ROOT/outputs/pilot_b
mkdir -p "$LOG" "$OUT" "$SCR"
export PYTHONPATH="$CODE:${PYTHONPATH:-}"

echo "=== $(date -Is) export teacher softlabels ==="
$PY "$SCR/export_teacher_softlabels.py" \
  --pool "$ROOT/manifests/pilot_b/distill_pool.jsonl" \
  --ext-root /root/autodl-tmp/external \
  --out-dir "$OUT/teacher_soft" \
  2>&1 | tee "$LOG/export_soft.log"

MERGED=$OUT/teacher_soft/teachers_merged.jsonl
test -f "$MERGED"

LITESSM_CKPT=/root/autodl-tmp/outputs/bakeoff/mobilemamba_lite/best.pt

run_one() {
  local student="$1" recipe="$2" run_name="$3" init="${4:-}"
  if [ -f "$OUT/$run_name/metrics.json" ]; then
    echo "skip done $run_name"
    return 0
  fi
  echo "=== $(date -Is) distill $run_name ==="
  EXTRA=()
  if [ -n "$init" ]; then EXTRA=(--init-ckpt "$init"); fi
  $PY "$SCR/train_gated_distill.py" \
    --student "$student" \
    --recipe "$recipe" \
    --run-name "$run_name" \
    --merged "$MERGED" \
    --manifest-root "$ROOT/manifests" \
    --out "$OUT" \
    --epochs 15 --batch 64 --lr 1e-4 --seed 42 \
    "${EXTRA[@]}" \
    2>&1 | tee "$LOG/${run_name}.log"
}

for recipe in npr_only univfd_only gated_dual; do
  run_one repvit_m0_9 "$recipe" "repvit_m0_9__${recipe}"
  run_one efficientnet_v2_s "$recipe" "efficientnet_v2_s__${recipe}"
  run_one mobilemamba_lite "$recipe" "mobilemamba_lite__${recipe}_warm" "$LITESSM_CKPT"
done

# from-scratch LiteSSM-A gated dual
run_one mobilemamba_lite gated_dual "mobilemamba_lite__gated_dual_scratch"

echo DONE > "$LOG/pilot_b_done.txt"
echo "pilot_b complete $(date -Is)"
