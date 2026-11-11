#!/bin/bash
# SPDX-License-Identifier: Apache-2.0
# Part of the Intel port (run 178, owner note 180 A): one ten-minute rotation (hstep.sh), then its misses a step,
# the per-position draft acceptance from /metrics and the acceptance by prompt family.
#   bash logs/hdraft_run178.sh STEP IMAGE_ID CFG
set -u
STEP=$1; IMG=$2; CFG=$3
W=$HOME/code/radiance-intel
D=/mnt/fast-storage/models/test/radiance-intel
R=$W/results-m3/card01
cd "$W" || exit 1
timeout 2400 bash logs/hstep.sh "$STEP" "$IMG" "$CFG" 10 > "logs/hstep-$STEP.txt" 2>&1
echo "hstep $STEP rc=$?"
tail -8 "logs/hstep-$STEP.txt" | cut -c1-200
grep -E "guard|klog2" "logs/hstep-$STEP.txt" | cut -c1-200
for f in stats-0 stats-final metrics-0 metrics-final; do
  ssh intel-host cat "$D/m3/$STEP/$f.txt" > "$R/$STEP/$f.txt"
done
python3 -I m3/stats_delta.py "$R/$STEP/stats-0.txt" "$R/$STEP/stats-final.txt" 48 | grep "per step"
grep -E "draft_position_acceptance|draft_acceptance_ratio" "$R/$STEP/metrics-final.txt" | cut -c1-140
python3 -I m3/draft_by_family.py "$R/$STEP/serve.log" | tail -12
