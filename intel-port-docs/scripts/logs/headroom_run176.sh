#!/bin/bash
# SPDX-License-Identifier: Apache-2.0
# Part of the Intel port (run 176, owner note 179 item 2): one ten-minute rotation at a given headroom (bestrx1024 /
# bestrx768), staged from H119's card files, then its misses a step (stats_delta) and each card's lowest driver-free
# VRAM from the trace lines.
#   bash logs/headroom_run176.sh STEP IMAGE_ID CFG
set -u
STEP=$1; IMG=$2; CFG=$3
W=$HOME/code/radiance-intel
D=/mnt/fast-storage/models/test/radiance-intel
R=$W/results-m3/card01
cd "$W" || exit 1
ssh intel-host mkdir -p "$D/m3/$STEP" || exit 1
ssh intel-host "tar -C $D/m3/H119 -cf - card flashnext-card1.args | tar -C $D/m3/$STEP -xf -" || exit 1
ssh intel-host tee "$D/m3/$STEP/m2-heat.sh" < "$W/m3/intel-host/m2-heat.sh" > /dev/null || exit 1
timeout 2400 bash logs/hstep.sh "$STEP" "$IMG" "$CFG" 10 > "logs/hstep-$STEP.txt" 2>&1
echo "hstep $STEP rc=$?"
grep -E "guard|klog2|requests|after the first" "logs/hstep-$STEP.txt" | cut -c1-200
ssh intel-host cat "$D/m3/$STEP/stats-0.txt" > "$R/$STEP/stats-0.txt"
ssh intel-host cat "$D/m3/$STEP/stats-final.txt" > "$R/$STEP/stats-final.txt"
python3 -I m3/stats_delta.py "$R/$STEP/stats-0.txt" "$R/$STEP/stats-final.txt" 48 | grep -E "per step|tiers"
for dv in 0 1; do
  grep -o "l0 vram trace dev $dv after play [0-9]*: process VRAM [0-9]* MiB ([^)]*); driver free [0-9]* MiB" "$R/$STEP/serve.log" \
    | grep -o "driver free [0-9]*" | awk -v d=$dv '{ if (min == "" || $3 < min) min = $3 } END { print "dev " d ": lowest driver free " min " MiB" }'
done
