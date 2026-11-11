#!/bin/bash
# SPDX-License-Identifier: Apache-2.0
# Part of the Intel port (run 271, owner lever #2's pricing probe; the coordinator's item 2 of
# logs/run271-light-poll-design.md): the all-reduce overlap probe on BOTH cards under the step rules (logs/tqstep.sh's
# form): stage m3/STEP on intel-host with the 2640 tree's tests/ar_overlap_probe and m3/intel-host/ar-overlap-probe.sh, cool to
# <= 43 C, precheck, the guard at --hot 80, m3-step.sh with both cards (no engine, no model), the kernel log 20 s after,
# the output fetched. Every all-reduce wait is bounded by the library's spin, the binary sits under a 180 s timeout
# inside and a 300 s step cap outside; a fault -> stop, the peer1 probe before any further card step.
#   bash logs/arostep.sh STEP IMAGE_ID [N_A=20] [N_G=20] [M=4096] [N=2048] [K=2048] [TREE=/srv/bulk/ri/build-m1l0neo2640]
set -u
STEP=$1; IMG=$2; NA=${3:-20}; NG=${4:-20}; MM=${5:-4096}; NN=${6:-2048}; KK=${7:-2048}; TREE=${8:-/srv/bulk/ri/build-m1l0neo2640}
W=$HOME/code/radiance-intel
D=/mnt/fast-storage/models/test/radiance-intel
R=$W/results-m3/card01
ssh intel-host mkdir -p "$D/m3/$STEP" || exit 1
scp -q "$TREE/tests/ar_overlap_probe" "intel-host:$D/m3/$STEP/ar_overlap_probe" || exit 1
ssh intel-host tee "$D/m3/$STEP/ar-overlap-probe.sh" < "$W/m3/intel-host/ar-overlap-probe.sh" > /dev/null || exit 1
timeout 1900 bash "$W/m3/cool43.sh" 1800 || exit 1
timeout 300 "$W/intel-host-engine-guard.sh" --precheck 8 || exit 1
deadline=$(( $(date +%s) + 900 ))
"$W/intel-host-engine-guard.sh" --hot 80 "ri-m3-$STEP" 1000 > "$W/logs/guard-$STEP.txt" 2>&1 &
guard=$!
timeout 600 ssh intel-host env STEP_CARD=both STEP_DEADLINE=$deadline \
    bash "$D/m3-step.sh" "$STEP" 300 "$IMG" bash /out/ar-overlap-probe.sh "$NA" "$NG" "$MM" "$NN" "$KK" > "$W/logs/step-$STEP.txt" 2>&1
grep 'rc=' "$W/logs/step-$STEP.txt" | tail -1
wait $guard
tail -1 "$W/logs/guard-$STEP.txt"
sleep 20
ssh intel-host bash "$D/ri-klog2.sh" "$STEP"
ssh intel-host cat "$D/out-$STEP.txt" > "$R/$STEP-out.txt"
grep -E '^RESULT|^VERDICT|^SKIP|rc=' "$R/$STEP-out.txt" | cut -c1-220
