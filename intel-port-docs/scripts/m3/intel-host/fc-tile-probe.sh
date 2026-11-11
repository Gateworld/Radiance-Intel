#!/bin/bash
# SPDX-License-Identifier: Apache-2.0
# Part of the Intel port of radiance (milestone 3, run 190, owner notes 188 / 189: the expert-layout probe, stage 1).
# Inside the step's container (one card): xe_fc_tile_probe from the step folder -- format C's decode GEMM through the
# image's own libxe against form T1 over a repack of the same entries, the bytes compared, then the per-launch times
# (host wall over LAUNCHES back-to-back launches, the median of RUNS) at gate_up's and down's per-card shapes, at the
# verify mix and at one token. The go bar (note 189): >= 25 % off the launch, bit-identical.
#   bash /out/fc-tile-probe.sh [LAUNCHES=48] [RUNS=5]
L=/opt/ri/radiance_home/kernels/libxe.so
[ -f /out/libxe.so ] && L=/out/libxe.so && echo "libxe.so from /out"
P=/out/xe_fc_tile_probe
chmod +x $P
echo "=== fc_tile_probe ($(date -u +%H:%M:%S)): $L"
# RI_PAIR=1 (run 193, owner note 191 item 2): the concurrency probe in the same process (PAIR lines)
# E1 / E1b (run 206, owner note 202 A.4): the ESIMD form of down, RI_E1_LW threads a work-group (3rd argument, 16)
# E3 (run 279, owner note 218 stage 2): the whole-kernel ESIMD form of gate_up (E3 / RESULT-E3 lines)
env RI_PAIR=1 RI_E1_LW=${3:-16} timeout 300 $P "$L" ${1:-48} ${2:-5} 2>&1 | grep -E "^fc_tile_probe|^PASS|^FAIL|^RESULT|^PAIR|^E1|^E3|^T2|first difference|rror|refused"
echo "rc=${PIPESTATUS[0]} ($(date -u +%H:%M:%S))"
