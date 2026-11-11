#!/bin/bash
# SPDX-License-Identifier: Apache-2.0
# Part of the Intel port of radiance (milestone 3, run 93, owner note 91 item 3). Inside the step's container (both
# cards): shared USM bound to card 1 (`shared1`) -- does card 1 create the buffer itself (nothing in card 0's fdinfo),
# do its copies and kernels read it, and does it stay in system memory (card 1's VRAM unchanged)? 2 GiB pieces.
#   bash /out/hostbind-matrix7.sh
export RI_HB_STUCK_S=${RI_HB_STUCK_S:-20}
S=/out/hostbind-summary.txt
: > $S
run() {   # name args...
  local name=$1; shift
  echo "=== $name: xe_hostbind_probe $*" >> /out/hostbind.log
  timeout 120 xe_hostbind_probe "$@" >> /out/hostbind.log 2>&1
  local rc=$?
  echo "$name rc=$rc $(grep -E 'STUCK|every|failed|WRONG' /out/hostbind.log | tail -1)" >> $S
}
run a-shared1-4g-1     4      1 2048 1 shared1
run b-shared1-16.6g-1  16.593 1 2048 1 shared1
cat $S
