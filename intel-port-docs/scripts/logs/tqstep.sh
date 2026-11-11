#!/bin/bash
# SPDX-License-Identifier: Apache-2.0
# Part of the Intel port (run 267, owner lever #2's step 0): the two-queue concurrency probe on card1 under the step
# rules (logs/pstep.sh's form): stage m3/STEP on intel-host with the 2640 tree's tests/two_queue_probe and
# m3/intel-host/two-queue-probe.sh, cool to <= 43 C, precheck, the guard at --hot 80, m3-step.sh on card1 alone (no engine,
# no model), the kernel log 20 s after, the output fetched. Bounded spin (the cap) + a 120 s host timeout inside, a
# 300 s step cap outside; a fault -> stop, the peer1 probe before any further card step.
#   bash logs/tqstep.sh STEP IMAGE_ID [PASSES=40] [GROUPS=64] [CAP=4000000] [TREE=/srv/bulk/ri/build-m1l0neo2640]
set -u
STEP=$1; IMG=$2; PS=${3:-40}; GR=${4:-64}; CAP=${5:-4000000}; TREE=${6:-/srv/bulk/ri/build-m1l0neo2640}
W=$HOME/code/radiance-intel
D=/mnt/fast-storage/models/test/radiance-intel
R=$W/results-m3/card01
ssh intel-host mkdir -p "$D/m3/$STEP" || exit 1
scp -q "$TREE/tests/two_queue_probe" "intel-host:$D/m3/$STEP/two_queue_probe" || exit 1
ssh intel-host tee "$D/m3/$STEP/two-queue-probe.sh" < "$W/m3/intel-host/two-queue-probe.sh" > /dev/null || exit 1
timeout 1900 bash "$W/m3/cool43.sh" 1800 || exit 1
timeout 300 "$W/intel-host-engine-guard.sh" --precheck 8 || exit 1
deadline=$(( $(date +%s) + 900 ))
"$W/intel-host-engine-guard.sh" --hot 80 "ri-m3-$STEP" 1000 > "$W/logs/guard-$STEP.txt" 2>&1 &
guard=$!
timeout 600 ssh intel-host env STEP_CARD=1 STEP_DEADLINE=$deadline \
    bash "$D/m3-step.sh" "$STEP" 300 "$IMG" bash /out/two-queue-probe.sh "$PS" "$GR" "$CAP" > "$W/logs/step-$STEP.txt" 2>&1
grep 'rc=' "$W/logs/step-$STEP.txt" | tail -1
wait $guard
tail -1 "$W/logs/guard-$STEP.txt"
sleep 20
ssh intel-host bash "$D/ri-klog2.sh" "$STEP"
ssh intel-host cat "$D/out-$STEP.txt" > "$R/$STEP-out.txt"
grep -E '^RESULT|^VERDICT|^SKIP|rc=' "$R/$STEP-out.txt" | cut -c1-200
