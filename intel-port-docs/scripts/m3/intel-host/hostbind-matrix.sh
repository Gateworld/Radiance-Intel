#!/bin/bash
# SPDX-License-Identifier: Apache-2.0
# Part of the Intel port of radiance (milestone 3, run 88, owner note 84). Inside the step's container (both cards): the
# two-card first step's stuck bind without the model, case by case (xe_hostbind_probe, PATH /opt/ri/tests). The
# engine's case is 16,991 MiB touched by card0 then card1 (R15). Each case is its own process: a stuck bind ends it
# after RI_HB_STUCK_S seconds (exit 3), `timeout` is the backstop. One line a case in /out/hostbind-summary.txt.
#   bash /out/hostbind-matrix.sh
export RI_HB_STUCK_S=${RI_HB_STUCK_S:-20}
S=/out/hostbind-summary.txt
: > $S
run() {   # name args...
  local name=$1; shift
  echo "=== $name: xe_hostbind_probe $*" >> /out/hostbind.log
  timeout 90 xe_hostbind_probe "$@" >> /out/hostbind.log 2>&1
  local rc=$?
  echo "$name rc=$rc $(grep -E 'STUCK|every touch' /out/hostbind.log | tail -1)" >> $S
}
run a-1.5g-01      1.5   01
run b-4g-01        4     01
run c-8g-01        8     01
run d-16.6g-01     16.593 01
run e-16.6g-1      16.593 1
run f-16.6g-01-1g  16.593 01 1024
run g-16.6g-01-nof 16.593 01 0 0
cat $S
