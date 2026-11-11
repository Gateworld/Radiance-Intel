#!/bin/bash
# SPDX-License-Identifier: Apache-2.0
# Part of the Intel port of radiance (milestone 3, run 108). Inside the step's container (one card): the expert
# matrix form's element-wise check (xe_moe_planes_bench CHECK=1 / 2: 196 routed rows, an exact host reference of
# ~1 s of one core) with the compact grid off and on, the libxe.so from the step folder if present.
#   bash /out/mx-check.sh
L=/opt/ri/radiance_home/kernels/libxe.so
[ -f /out/libxe.so ] && L=/out/libxe.so
for st in ${STAGES:-0}; do
CXS=${CXS:-0,1}
for cx in ${CXS//,/ }; do
  for v in 1 2; do
    # cx d = RADIANCE_XE_MOE_MX_COMPACT unset (the library's default, run 109)
    if [ "$cx" = d ]; then ce=(); else ce=(RADIANCE_XE_MOE_MX_COMPACT=$cx); fi
    echo "=== CHECK=$v ${ce[*]:-(compact unset)} RADIANCE_XE_MOE_MX_STAGE=$st ($(date -u +%H:%M:%S))"
    env RADIANCE_XE_MOE_MX=1 "${ce[@]}" RADIANCE_XE_MOE_MX_STAGE=$st CHECK=$v BENCH_TP2=1 timeout 120 \
      /opt/ri/tests/xe_moe_planes_bench "$L" 2>&1 | grep -E "RESULT|bad|worst|check:" | head -8
  done
done
done
