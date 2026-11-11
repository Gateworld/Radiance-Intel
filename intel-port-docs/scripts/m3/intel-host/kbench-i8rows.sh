#!/bin/bash
# SPDX-License-Identifier: Apache-2.0
# Part of the Intel port of radiance (milestone 3, run 99, owner note 105 item 1). Inside the step's container (one card,
# STEP_ENGINE=1 for /d): the trunk's W8A8 product (gemm_nt_q) against libref on the real container's weights, with the
# GEMV and with the matrix-unit rows form (RADIANCE_XE_GEMM_DPAS_ROWS=1, and run 101's accumulate variants 2 / 3). The
# rows form is meant to give the GEMV's bits, so a variant's report should be the GEMV's file (md5).
#   bash /out/kbench-i8rows.sh [MAXTOK=8] [MODES=0,1]   (commas: ssh flattens quoted spaces)
M=/d/qwen3.8-next-flash-fp8-iq4r-moe.rad
H=/opt/ri/radiance_home
MT=${1:-8}   # run 101: 15 / 64 too (prefill chunks of 8 rows and the tails past the 64-row blocks)
MODES=${2:-0,1}
for r in ${MODES//,/ }; do
  echo "=== RADIANCE_XE_GEMM_DPAS_ROWS=$r --max-tok $MT"
  RADIANCE_XE_GEMM_DPAS_ROWS=$r rad-kbench -m $M --home $H --kernels libxe:libref --op gemm_nt_q --max-ctx 4096 \
    --max-tok $MT --report /out/report-rows$r.md
  echo "rc=$?"
done
md5sum /out/report-rows*.md
