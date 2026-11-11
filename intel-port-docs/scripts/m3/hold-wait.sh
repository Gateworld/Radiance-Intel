#!/bin/bash
# SPDX-License-Identifier: Apache-2.0
# Part of the Intel port of radiance (milestone 3, run 110, owner note 132 item 2). Before a card step: while
# ~/code/radiance-intel/HOLD-INTEL-HOST exists, wait (a check a minute, at most MAX seconds, default 2700 = 45 min), and
# say how long it held. Exit 0 when free; exit 1 when it still holds after MAX (then: WAITING-FOR-OWNER, amd-host-only work).
#   bash m3/hold-wait.sh [MAX]
F=~/code/radiance-intel/HOLD-INTEL-HOST
max=${1:-2700}; t=0
while [ -e "$F" ]; do
  [ "$t" -ge "$max" ] && { echo "hold-wait: HOLD-INTEL-HOST still there after $t s"; exit 1; }
  sleep 60; t=$((t + 60))
done
echo "hold-wait: free after $t s"
