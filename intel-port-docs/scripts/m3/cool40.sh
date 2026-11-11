#!/bin/bash
# SPDX-License-Identifier: Apache-2.0
# Part of the Intel port of radiance (milestone 3, run 95, owner note 89: cool to <= 40 C before each load). Polls
# the guard's cool check every 30 s until intel-host reads 40 C or less, at most MAX seconds (default 1800).
#   bash m3/cool40.sh [MAX]
G=~/code/radiance-intel/intel-host-engine-guard.sh
max=${1:-1800}; t=0
while :; do
  line=$($G --cool 1800)
  if echo "$line" | grep -q -E 'cool at ([0-3][0-9]|40) C'; then echo "$line"; exit 0; fi
  [ "$t" -ge "$max" ] && { echo "cool40: not at 40 C after $t s: $line"; exit 1; }
  sleep 30; t=$((t + 30))
done
