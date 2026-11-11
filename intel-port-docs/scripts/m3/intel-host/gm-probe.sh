#!/bin/bash
# SPDX-License-Identifier: Apache-2.0
# Part of the Intel port of radiance (milestone 3, run 274, the decode step's pairs 4 and 5). Inside the step's container
# (card1 alone): xe_gemv_multi_test from the step folder against the image's libxe.so -- three gemm_nt_q rows vs
# gemm_nt_q_x3, and gemm_nt_q + gemm_nt vs gemm_nt_q_bf16_pair, over one activation at M 1 / 4, every output byte for
# byte on the card (RESULT lines, PASS / FAIL). RADIANCE_XE_GEMV_NX=1 as the decode configs run (the form the paired
# kernels copy). No engine, no model: deterministic data, seconds.
#   bash /out/gm-probe.sh
P=/out/xe_gemv_multi_test
chmod +x $P
echo "=== gemv_multi_test ($(date -u +%H:%M:%S))"
RADIANCE_XE_GEMV_MULTI=1 RADIANCE_XE_GEMV_NX=1 RI_LIBXE=${RI_LIBXE:-/opt/ri/radiance_home/kernels/libxe.so} timeout 300 $P 2>&1 | grep -E "^gemv_multi|^RESULT|^PASS|^FAIL|libxe:|rror|refu"
echo "rc=${PIPESTATUS[0]} ($(date -u +%H:%M:%S))"
