#!/bin/bash
# SPDX-License-Identifier: Apache-2.0
# Part of the Intel port of radiance (milestone 3, run 271, owner lever #2's pricing probe; the coordinator's item 2 of
# logs/run271-light-poll-design.md). Inside the step's container (BOTH cards, no engine, no model): ar_overlap_probe from
# the step folder against the image's own libxe.so -- the engine's 16 MiB all-reduce (proto 6, the bandwidth form) on
# queue A of each card beside the trunk's int8 matrix-unit GEMM at 4,096 rows on queue B, alone and together. Every
# all-reduce wait is bounded by the library's spin (RADIANCE_XE_AR_SPIN) and the whole run sits under a host timeout.
#   bash /out/ar-overlap-probe.sh [N_A=20] [N_G=20] [M=4096] [N=2048] [K=2048]
P=/out/ar_overlap_probe
L=/opt/ri/radiance_home/kernels/libxe.so
chmod +x $P
echo "=== ar_overlap_probe ($(date -u +%H:%M:%S)) proto ${RADIANCE_XE_AR_PROTO:-default} spin ${RADIANCE_XE_AR_SPIN:-default}"
timeout 180 $P $L ${1:-20} ${2:-20} ${3:-4096} ${4:-2048} ${5:-2048} 2>&1 | grep -E "^ar_overlap|^RESULT|^VERDICT|^SKIP|rror|stall|gave up"
echo "rc=${PIPESTATUS[0]} ($(date -u +%H:%M:%S))"
