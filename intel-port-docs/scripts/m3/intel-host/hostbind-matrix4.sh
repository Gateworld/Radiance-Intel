#!/bin/bash
# SPDX-License-Identifier: Apache-2.0
# Part of the Intel port of radiance (milestone 3, run 92, owner note 90 item 2). Inside the step's container (both
# cards' nodes): HB2's case c on card1 alone -- host USM 16.6 GiB as 9 allocations of 2 GiB in a context over both
# cards, copies and a kernel read on card1 only (the importing card, the fix's case) -- while card0 may still hold M62's
# stuck copy job. Every piece is 2 GiB (owner note 87).
#   bash /out/hostbind-matrix4.sh
export RI_HB_STUCK_S=${RI_HB_STUCK_S:-20}
S=/out/hostbind-summary.txt
: > $S
echo "=== c1-usm-16.6g: xe_hostbind_probe 16.593 1 2048 1 usm" >> /out/hostbind.log
timeout 120 xe_hostbind_probe 16.593 1 2048 1 usm >> /out/hostbind.log 2>&1
echo "c1-usm-16.6g rc=$? $(grep -E 'STUCK|every|failed|WRONG' /out/hostbind.log | tail -1)" >> $S
cat $S
