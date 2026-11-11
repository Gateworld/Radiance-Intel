#!/bin/bash
# SPDX-License-Identifier: Apache-2.0
# Part of the Intel port of radiance (milestone 3, run 93, owner note 91 item 3). Inside the step's container (both
# cards): host memory from a context of card 1 alone (`solo1`), copied from and read by kernels on card 1 through the
# shared context's queue. Card 0 never maps it. Every piece is 2 GiB (owner note 87) until card 1's ownership is shown.
#   bash /out/hostbind-matrix6.sh
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
run a-solo1-4g-1     4      1 2048 1 solo1
run b-solo1-16.6g-1  16.593 1 2048 1 solo1
cat $S
