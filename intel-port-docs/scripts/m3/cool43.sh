#!/bin/bash
# SPDX-License-Identifier: Apache-2.0
# Part of the Intel port of radiance (milestone 3, run 108, owner note 121: cool to <= 43 C before each load until the
# new heatsink). Polls the guard's cool check every 30 s until intel-host reads 43 C or less, at most MAX seconds (default
# 1800). The guard's own --cool returns at 44 C.
#   bash m3/cool43.sh [MAX]
G=~/code/radiance-intel/intel-host-engine-guard.sh
max=${1:-1800}; t=0
while :; do
  line=$($G --cool 1800)
  if echo "$line" | grep -q -E 'cool at ([0-3][0-9]|4[0-3]) C'; then echo "$line"; exit 0; fi
  [ "$t" -ge "$max" ] && { echo "cool43: not at 43 C after $t s: $line"; exit 1; }
  sleep 30; t=$((t + 30))
done
