#!/bin/bash
# SPDX-License-Identifier: Apache-2.0
# Part of the Intel port: the expert-layout + two-list probe step (owner notes 189 / 191 item 2) under the step rules:
# stage m3/STEP on intel-host with build-m1l0/tests/xe_fc_tile_probe + m3/intel-host/fc-tile-probe.sh, cool to <= 43 C,
# precheck, the guard at --hot 80, m3-step.sh on card1 alone (no engine), the kernel log 20 s after, the output fetched.
#   bash logs/pstep.sh STEP IMAGE_ID [LAUNCHES=48] [RUNS=5]
set -u
STEP=$1; IMG=$2; LN=${3:-48}; RU=${4:-5}
W=$HOME/code/radiance-intel
D=/mnt/fast-storage/models/test/radiance-intel
R=$W/results-m3/card01
ssh intel-host mkdir -p "$D/m3/$STEP" || exit 1
scp -q "$W/build-m1l0/tests/xe_fc_tile_probe" "intel-host:$D/m3/$STEP/xe_fc_tile_probe" || exit 1
ssh intel-host tee "$D/m3/$STEP/fc-tile-probe.sh" < "$W/m3/intel-host/fc-tile-probe.sh" > /dev/null || exit 1
timeout 1900 bash "$W/m3/cool43.sh" 1800 || exit 1
timeout 300 "$W/intel-host-engine-guard.sh" --precheck 8 || exit 1
deadline=$(( $(date +%s) + 1800 ))
"$W/intel-host-engine-guard.sh" --hot 80 "ri-m3-$STEP" 1000 > "$W/logs/guard-$STEP.txt" 2>&1 &
guard=$!
timeout 900 ssh intel-host env STEP_CARD=1 STEP_DEADLINE=$deadline \
    bash "$D/m3-step.sh" "$STEP" 600 "$IMG" bash /out/fc-tile-probe.sh "$LN" "$RU" > "$W/logs/step-$STEP.txt" 2>&1
grep 'rc=' "$W/logs/step-$STEP.txt" | tail -1
wait $guard
tail -1 "$W/logs/guard-$STEP.txt"
sleep 20
ssh intel-host bash "$D/ri-klog2.sh" "$STEP"
mkdir -p "$R/$STEP"
ssh intel-host cat "$D/out-$STEP.txt" > "$R/$STEP-out.txt"
grep -E "^fc_tile_probe|^PASS|^FAIL|^RESULT|^PAIR|^E1|^T2|first difference|rror|refused|rc=""$R/$STEP-out.txt" | cut -c1-200
