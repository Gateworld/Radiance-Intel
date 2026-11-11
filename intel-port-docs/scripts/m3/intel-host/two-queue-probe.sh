#!/bin/bash
# SPDX-License-Identifier: Apache-2.0
# Part of the Intel port of radiance (milestone 3, run 267, owner lever #2's step 0: "card test only"). Inside the step's
# container (card1 alone): two_queue_probe from the step folder -- do two in-order queues run kernels concurrently on
# one B70, and how much does a memory-bound kernel slow while a spinning kernel shares the EUs. Every spin is bounded
# (the cap) and the whole run sits under a host timeout.
#   bash /out/two-queue-probe.sh [PASSES=40] [GROUPS=64] [CAP=4000000]
P=/out/two_queue_probe
chmod +x $P
echo "=== two_queue_probe ($(date -u +%H:%M:%S))"
timeout 120 $P ${1:-40} ${2:-64} ${3:-4000000} 2>&1 | grep -E "^two_queue|^QUEUES|^RESULT|^VERDICT|^SKIP|rror"
echo "rc=${PIPESTATUS[0]} ($(date -u +%H:%M:%S))"
