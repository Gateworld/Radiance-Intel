#!/bin/bash
# SPDX-License-Identifier: Apache-2.0
# Part of the Intel port of radiance (milestone 3, run 93, owner note 91 item 3). Inside the step's container (both
# cards): can card 1 own host memory instead of importing card 0's? `own1` allocates through a second context with card
# 1 first and uses it from the shared context's queues. Every piece is 2 GiB (owner note 87). The fdinfo lines at each
# case's end show imports as "system" on the importing card.
#   bash /out/hostbind-matrix5.sh
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
run a-usm-4g-1     4      1  2048 1 usm
run b-own1-4g-1    4      1  2048 1 own1
run c-own1-16.6g-1 16.593 1  2048 1 own1
run d-own1-16.6g-0 16.593 0  2048 1 own1
cat $S
