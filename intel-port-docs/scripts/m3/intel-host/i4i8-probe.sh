#!/bin/bash
# SPDX-License-Identifier: Apache-2.0
# Part of the Intel port of radiance (milestone 3, run 110, owner note 131). Inside the step's container (one card):
# xe_i4i8_probe from the step folder -- the i4 x i8 expert GEMV prototype against today's libxe expert GEMV at one
# rank's shapes, 1 row and the verify mix, in one process. Pass 1: the launcher's defaults; pass 2: today's best at
# the verify (RADIANCE_XE_MOE8M=1, RADIANCE_XE_MOE8M_R=8). Pass 1 also dumps launch 0's outputs and exact group sums
# to /out/i4i8-dump for the host reference, which runs on amd-host (owner note 3), not here.
#   bash /out/i4i8-probe.sh [LAUNCHES=48] [RUNS=5]
L=/opt/ri/radiance_home/kernels/libxe.so
[ -f /out/libxe.so ] && L=/out/libxe.so && echo "libxe.so from /out"
P=/out/xe_i4i8_probe
chmod +x $P
mkdir -p /out/i4i8-dump
echo "=== pass 1: defaults, DUMP ($(date -u +%H:%M:%S))"
env DUMP=/out/i4i8-dump timeout 300 $P "$L" ${1:-48} ${2:-5} 2>&1 | grep -E "^RESULT|^device|^gate_up|^down|rror|refused"
echo "rc=${PIPESTATUS[0]}"
echo "=== pass 2: RADIANCE_XE_MOE8M=1 RADIANCE_XE_MOE8M_R=8 ($(date -u +%H:%M:%S))"
env RADIANCE_XE_MOE8M=1 RADIANCE_XE_MOE8M_R=8 timeout 300 $P "$L" ${1:-48} ${2:-5} 2>&1 | grep -E "^RESULT|rror|refused"
echo "rc=${PIPESTATUS[0]} ($(date -u +%H:%M:%S))"
ls -la /out/i4i8-dump | head -20
