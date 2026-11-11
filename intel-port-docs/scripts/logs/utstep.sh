#!/bin/bash
# SPDX-License-Identifier: Apache-2.0
# Part of the Intel port: note 169 item 3 (run 164) -- a plain command in the counters test image (ri-m3ut:N) on both
# cards under the step rules, e.g. unitrace's metric list, to see whether the metrics libraries load and what
# groups the B70 reports. No model: STEP_ENGINE=0.
#   bash logs/utstep.sh STEP IMAGE_ID COMMAND...     e.g. bash logs/utstep.sh UT1 0783a034b003 /opt/unitrace/bin/unitrace --metric-list
set -u
STEP=$1; IMG=$2; shift 2
W=$HOME/code/radiance-intel
D=/mnt/fast-storage/models/test/radiance-intel
ssh intel-host mkdir -p "$D/m3/$STEP" || exit 1
timeout 1900 bash "$W/m3/cool43.sh" 1800 || exit 1
timeout 300 "$W/intel-host-engine-guard.sh" --precheck 1 || exit 1
deadline=$(( $(date +%s) + 600 ))
"$W/intel-host-engine-guard.sh" --hot 80 "ri-m3-$STEP" 120 > "$W/logs/guard-$STEP.txt" 2>&1 &
guard=$!
timeout 500 ssh intel-host env STEP_CARD=both STEP_CPUS=2 STEP_NOTE7_CPUS2=1 STEP_DEADLINE=$deadline \
    bash "$D/m3-step.sh" "$STEP" 300 "$IMG" "$@" > "$W/logs/step-$STEP.txt" 2>&1
grep 'rc=' "$W/logs/step-$STEP.txt" | tail -1
wait $guard
tail -1 "$W/logs/guard-$STEP.txt"
sleep 10
ssh intel-host bash "$D/ri-klog2.sh" "$STEP"
ssh intel-host cat "$D/out-$STEP.txt" > "$W/results-m3/card01/$STEP-out.txt"
head -60 "$W/results-m3/card01/$STEP-out.txt" | cut -c1-150
