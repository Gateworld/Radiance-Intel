#!/bin/bash
# SPDX-License-Identifier: Apache-2.0
# Part of the Intel port of radiance (milestone 3, run 58, stage B). gemm_nt bf16's test on the card: the matrix-unit
# form (M >= 64) and the GEMV, each against the double reference with canaries, then the prefill-shaped timings
# (2,048 rows) both ways. The binary comes from /out (amd-host's build of the same sources as the image's libxe.so),
# copied to /tmp because the mount may be noexec.
#   m3-step.sh B11t 600 <image> bash /out/b11t.sh
set -u
cp /out/xe_gemv_bf16_test /tmp/t && chmod +x /tmp/t || { echo "b11t: no test binary in /out"; exit 2; }
rc=0
echo "=== the matrix-unit form on (default)"
timeout 300 /tmp/t 5 || rc=1
echo "=== RADIANCE_XE_GEMM_BF16_DPAS=0 (the GEMV for every M)"
env RADIANCE_XE_GEMM_BF16_DPAS=0 timeout 300 /tmp/t 5 || rc=1
echo "b11t: done rc=$rc"
exit $rc
