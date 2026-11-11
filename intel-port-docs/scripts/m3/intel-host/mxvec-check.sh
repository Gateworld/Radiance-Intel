#!/bin/bash
# SPDX-License-Identifier: Apache-2.0
# Part of the Intel port of radiance (milestone 3, run 110, owner note 129 item 5). Inside the step's container (one
# card): the expert matrix form's element-wise check (xe_moe_planes_bench CHECK=1 / 2, BENCH_TP2) with STAGE on and
# RADIANCE_XE_MOE_MX_VEC 0 / 1; VEC stages the same values, so both must print the same worst d/bound, positions and
# bias. The step folder's libxe.so.
#   bash /out/mxvec-check.sh
L=/opt/ri/radiance_home/kernels/libxe.so
[ -f /out/libxe.so ] && L=/out/libxe.so && echo "libxe.so from /out"
for vc in 0 1; do
  for v in 1 2; do
    echo "=== CHECK=$v RADIANCE_XE_MOE_MX_STAGE=1 RADIANCE_XE_MOE_MX_VEC=$vc ($(date -u +%H:%M:%S))"
    env RADIANCE_XE_MOE_MX=1 RADIANCE_XE_MOE_MX_STAGE=1 RADIANCE_XE_MOE_MX_VEC=$vc CHECK=$v BENCH_TP2=1 timeout 120 \
      /opt/ri/tests/xe_moe_planes_bench "$L" 2>&1 | grep -E "RESULT|bad|worst|check:" | head -8
  done
done
