#!/bin/bash
# SPDX-License-Identifier: Apache-2.0
# Part of the Intel port of radiance (milestone 3, run 89, owner note 87). Inside the step's container (both cards): the
# two candidate fixes for the two-card bind bug, with every host allocation or registered piece at 2 GiB (HB1: 4 GiB
# binds on card1, 8 GiB crashed the kernel). usm = separate host USM allocations (the pool in pieces); import = one
# contiguous mmap registered in pieces (zexDriverImportExternalPointer). Each case copies from every piece on card0
# then card1 and runs a kernel reading every piece on each. One line a case in /out/hostbind-summary.txt.
#   bash /out/hostbind-matrix2.sh
export RI_HB_STUCK_S=${RI_HB_STUCK_S:-20}
S=/out/hostbind-summary.txt
: > $S
run() {   # name args...
  local name=$1; shift
  echo "=== $name: xe_hostbind_probe $*" >> /out/hostbind.log
  timeout 120 xe_hostbind_probe "$@" >> /out/hostbind.log 2>&1
  local rc=$?
  echo "$name rc=$rc $(grep -E 'STUCK|every|ImportExternal|failed' /out/hostbind.log | tail -1)" >> $S
}
run a-usm-2g      2      01 2048 1 usm
run b-import-2g   2      01 2048 1 import
run c-usm-16.6g   16.593 01 2048 1 usm
run d-import-16.6g 16.593 01 2048 1 import
cat $S
