#!/bin/bash
# SPDX-License-Identifier: Apache-2.0
# Part of the Intel port of radiance (milestone 3, after run 89's pause, owner note 87). Inside the step's container
# (both cards): HB2's case c alone in a fresh step -- host USM 16.6 GiB as 9 allocations of 2 GiB, copies and a kernel
# read on card0 then card1 -- the "pool in pieces" fix, without the import mode that faulted before it (m3 fault 23).
#   bash /out/hostbind-matrix3.sh
export RI_HB_STUCK_S=${RI_HB_STUCK_S:-20}
S=/out/hostbind-summary.txt
: > $S
echo "=== c-usm-16.6g: xe_hostbind_probe 16.593 01 2048 1 usm" >> /out/hostbind.log
timeout 120 xe_hostbind_probe 16.593 01 2048 1 usm >> /out/hostbind.log 2>&1
echo "c-usm-16.6g rc=$? $(grep -E 'STUCK|every|failed' /out/hostbind.log | tail -1)" >> $S
cat $S
