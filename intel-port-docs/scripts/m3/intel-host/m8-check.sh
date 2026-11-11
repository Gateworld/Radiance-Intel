#!/bin/bash
# SPDX-License-Identifier: Apache-2.0
# Part of the Intel port of radiance (milestone 3, run 110, owner note 129 item 2). Inside the step's container (one
# card): xe_moe_planes_bench's element-wise check (CHECK=1 / 2: 196 routed rows, an exact host reference of ~1 s of one
# core) on the several-token v2m path (RADIANCE_XE_MOE_MX=0) with RADIANCE_XE_MOE8M off and on; the step folder's
# libxe.so if present. moe8 is bit for bit, so both must print the same worst d/bound, positions and bias.
#   bash /out/m8-check.sh
L=/opt/ri/radiance_home/kernels/libxe.so
[ -f /out/libxe.so ] && L=/out/libxe.so
for m in 0 1; do
  for v in 1 2; do
    echo "=== CHECK=$v RADIANCE_XE_MOE_MX=0 RADIANCE_XE_MOE8M=$m ($(date -u +%H:%M:%S))"
    env RADIANCE_XE_MOE_MX=0 RADIANCE_XE_MOE8M=$m CHECK=$v BENCH_TP2=1 timeout 120 \
      /opt/ri/tests/xe_moe_planes_bench "$L" 2>&1 | grep -E "RESULT|check:" | head -6
  done
done
