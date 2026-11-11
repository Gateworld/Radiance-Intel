#!/bin/bash
# SPDX-License-Identifier: Apache-2.0
# Part of the Intel port of radiance (milestone 3, run 116, note 146 item 3). Inside the step's container (card1,
# STEP_ENGINE=1 for /d): format B's ops on the one-layer convert's real weights (layer 0's experts in w4g64i8h)
# against libref -- moe_gemm_q (each case's shape and error) and its int8 quantiser had_quant_act_i8g.
#   bash /out/kb-fb.sh [MAXTOK=4]
M=/d/flashnext-w4g64i8h-l0.rad
H=/opt/ri/radiance_home
for op in moe_gemm_q had_quant_act_i8g; do
  echo "=== $op --max-tok ${1:-4} ($(date -u +%H:%M:%S))"
  timeout 600 rad-kbench -m $M --home $H --kernels libxe:libref --op $op --max-ctx 4096 --max-tok "${1:-4}" \
    --report /out/r-fb-$op.md > /out/kb-fb-$op.log 2>&1
  echo "rc=$? $(grep -m1 -o -E '[0-9]+ case\(s\) checked, [0-9]+ passed, [0-9]+ failed' /out/r-fb-$op.md)"
  grep -E "w4g64i8h|i8g" /out/r-fb-$op.md | head -20
done
