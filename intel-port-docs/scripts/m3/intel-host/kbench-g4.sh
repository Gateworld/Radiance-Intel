#!/bin/bash
# SPDX-License-Identifier: Apache-2.0
# Part of the Intel port of radiance (milestone 3, run 110, owner note 129 item 3). Inside the step's container (one
# card, STEP_ENGINE=1 for /d): the trunk's W8A8 product (gemm_nt_q) against libref on the real container's weights,
# small (owner note 3), with the GEMV as it is (NX) and with RADIANCE_XE_GEMV_RT=1 (the 4-row launches' transpose-
# reduce). RT is meant to give the GEMV's bits, so both reports should be the same file (md5). The step folder's libxe.so.
#   bash /out/kbench-g4.sh [MAXTOK=8]
M=/d/qwen3.8-next-flash-fp8-iq4r-moe.rad
H=/opt/ri/radiance_home
MT=${1:-8}
[ -f /out/libxe.so ] && cp /out/libxe.so $H/kernels/libxe.so && echo "libxe.so from /out"
for v in 0 1; do
  echo "=== RADIANCE_XE_GEMV_NX=1 RADIANCE_XE_GEMV_RT=$v --max-tok $MT ($(date -u +%H:%M:%S))"
  RADIANCE_XE_GEMV_NX=1 RADIANCE_XE_GEMV_RT=$v timeout 300 rad-kbench -m $M --home $H \
    --kernels libxe:libref --op gemm_nt_q --max-ctx 4096 --max-tok $MT --report /out/report-rt$v.md > /out/kb-rt$v.log 2>&1
  echo "rc=$? $(grep -m1 -o -E '[0-9]+ case\(s\) checked, [0-9]+ passed, [0-9]+ failed.*' /out/report-rt$v.md)"
done
md5sum /out/report-rt*.md
