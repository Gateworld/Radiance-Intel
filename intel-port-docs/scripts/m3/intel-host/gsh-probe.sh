#!/bin/bash
# SPDX-License-Identifier: Apache-2.0
# Part of the Intel port of radiance (milestone 3, run 285, group A of owner note 232's audit). Inside the step's
# container (card1 alone): xe_router_sh_test from the step folder against the image's libxe.so -- the fused router op
# (router_gemm_topk_scatter_quant) + the standalone int8 GEMV against router_gemm_topk_scatter_quant_sh (the shared
# expert's gate_up in the router launch) at M 1 / 4 / 8 / 12, every output byte for byte on the card (RESULT lines,
# PASS / FAIL). RADIANCE_XE_GEMV_NX=1 as the decode configs run (the form the paired kernel copies). No engine, no
# model: deterministic data, seconds.
#   bash /out/gsh-probe.sh
P=/out/xe_router_sh_test
chmod +x $P
echo "=== router_sh_test ($(date -u +%H:%M:%S))"
RADIANCE_XE_GEMV_MULTI2=1 RADIANCE_XE_GEMV_NX=1 RI_LIBXE=${RI_LIBXE:-/opt/ri/radiance_home/kernels/libxe.so} timeout 300 $P 2>&1 | grep -E "^router_sh|^RESULT|^PASS|^FAIL|libxe:|rror|refu"
echo "rc=${PIPESTATUS[0]} ($(date -u +%H:%M:%S))"
# group B's row check in the same step (the coordinator, run 285): gemv_multi_test's x3 + bf16 case against the x3 and
# bf16 rows, under RADIANCE_XE_GEMV_MULTI2=b (the pairs 4 / 5 cases run too, as gm-probe.sh runs them)
M=/out/xe_gemv_multi_test
if [ -f $M ]; then
  chmod +x $M
  echo "=== gemv_multi_test with the x3 + bf16 row ($(date -u +%H:%M:%S))"
  RADIANCE_XE_GEMV_MULTI=1 RADIANCE_XE_GEMV_MULTI2=b RADIANCE_XE_GEMV_NX=1 RI_LIBXE=${RI_LIBXE:-/opt/ri/radiance_home/kernels/libxe.so} timeout 300 $M 2>&1 | grep -E "^gemv_multi|^RESULT|^PASS|^FAIL|libxe:|rror|refu"
  echo "rc=${PIPESTATUS[0]} ($(date -u +%H:%M:%S))"
fi
