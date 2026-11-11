#!/bin/bash
# SPDX-License-Identifier: Apache-2.0
# Part of the Intel port of radiance (milestone 3, run 267, owner lever #3). Inside the step's container (card1 alone):
# xe_moe_bf16_dpas_test from the step folder -- the protected experts' bf16 GEMM on the matrix units against the WMMA
# form's arithmetic (the host reference): three shapes, the ulp report (RESULT lines), canaries, the launcher's guards.
# Fails above 2 bf16 ulps, on a wrong row / column or a broken canary.
#   bash /out/moe-bf16-probe.sh
P=/out/xe_moe_bf16_dpas_test
chmod +x $P
echo "=== moe_bf16_dpas_test ($(date -u +%H:%M:%S))"
# run 268 (MB4): the old op r4d_rad_moe_gemm_bf16 from the image's libxe.so, scored beside the matrix form
RI_LIBXE=${RI_LIBXE:-/opt/ri/radiance_home/kernels/libxe.so} timeout 300 $P 2>&1 | grep -E "^moe_bf16|^RESULT|^DIFF|^PASS|^FAIL|^SKIP|rror|refu"
echo "rc=${PIPESTATUS[0]} ($(date -u +%H:%M:%S))"
