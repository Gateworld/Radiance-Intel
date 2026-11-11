#!/bin/bash
# SPDX-License-Identifier: Apache-2.0
# Part of the Intel port: run 206's container-listing step (the format C level-table count): stage m3/STEP on intel-host with
# m3/intel-host/radinfo.sh, the guard, m3-step.sh on card1 with the model folder at /d (STEP_ENGINE=1; no GPU work), the
# kernel log 20 s after, the listing fetched to results-m3/card01/STEP/.
#   bash logs/rstep.sh STEP IMAGE_ID
set -u
STEP=$1; IMG=$2
W=$HOME/code/radiance-intel
D=/mnt/fast-storage/models/test/radiance-intel
R=$W/results-m3/card01
ssh intel-host mkdir -p "$D/m3/$STEP" || exit 1
ssh intel-host tee "$D/m3/$STEP/radinfo.sh" < "$W/m3/intel-host/radinfo.sh" > /dev/null || exit 1
timeout 300 "$W/intel-host-engine-guard.sh" --precheck 8 || exit 1
deadline=$(( $(date +%s) + 1800 ))
"$W/intel-host-engine-guard.sh" --hot 80 "ri-m3-$STEP" 600 > "$W/logs/guard-$STEP.txt" 2>&1 &
guard=$!
timeout 700 ssh intel-host env STEP_CARD=1 STEP_ENGINE=1 STEP_DEADLINE=$deadline \
    bash "$D/m3-step.sh" "$STEP" 500 "$IMG" bash /out/radinfo.sh > "$W/logs/step-$STEP.txt" 2>&1
grep 'rc=' "$W/logs/step-$STEP.txt" | tail -1
wait $guard
tail -1 "$W/logs/guard-$STEP.txt"
sleep 20
ssh intel-host bash "$D/ri-klog2.sh" "$STEP"
mkdir -p "$R/$STEP"
ssh intel-host cat "$D/out-$STEP.txt" > "$R/$STEP-out.txt"
scp -q "intel-host:$D/m3/$STEP/recipe.txt" "intel-host:$D/m3/$STEP/radinfo-v.txt" "$R/$STEP/"
cat "$R/$STEP-out.txt" | cut -c1-200
