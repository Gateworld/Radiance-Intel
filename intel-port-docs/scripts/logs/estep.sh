#!/bin/bash
# SPDX-License-Identifier: Apache-2.0
# Part of the Intel port (run 279, owner note 218 stage 2): the whole-kernel ESIMD form of format C's gate_up (E3 in
# xe_fc_tile_probe) against today's kernel on card1, under the step rules (logs/pstep.sh's form with rbstep.sh's
# staging): stage m3/STEP on intel-host with the 2640 tree's tests/xe_fc_tile_probe, its radiance_home/kernels/libxe.so
# (today's kernel, the same build as the image's) and m3/intel-host/fc-tile-probe.sh; cool to <= 43 C; precheck; the guard
# at --hot 80 (GUARD_CAP, default 300 s); m3-step.sh on card1 alone (no engine, no model); the kernel log 20 s after;
# the output fetched to results-m3/card01/STEP-out.txt. The probe runs gate_up (and down) at the verify mix (4 tokens,
# 27 experts) and at one token (10 experts); the gate (owner note 218 stage 2): E3 bit-identical to today AND gate_up
# <= ~70 us a launch at 4 tokens.
#   timeout 1900 bash logs/estep.sh STEP IMAGE_ID [LAUNCHES=48] [RUNS=5] [TREE=/srv/bulk/ri/build-m1l0neo2640]
set -u
STEP=$1; IMG=$2; LN=${3:-48}; RU=${4:-5}; TREE=${5:-/srv/bulk/ri/build-m1l0neo2640}
W=$HOME/code/radiance-intel
D=/mnt/fast-storage/models/test/radiance-intel
R=$W/results-m3/card01
ssh intel-host mkdir -p "$D/m3/$STEP" || exit 1
scp -q "$TREE/tests/xe_fc_tile_probe" "intel-host:$D/m3/$STEP/xe_fc_tile_probe" || exit 1
scp -q "$TREE/radiance_home/kernels/libxe.so" "intel-host:$D/m3/$STEP/libxe.so" || exit 1
ssh intel-host tee "$D/m3/$STEP/fc-tile-probe.sh" < "$W/m3/intel-host/fc-tile-probe.sh" > /dev/null || exit 1
timeout 1900 bash "$W/m3/cool43.sh" 1800 || exit 1
timeout 300 "$W/intel-host-engine-guard.sh" --precheck 8 || exit 1
deadline=$(( $(date +%s) + 1500 ))
"$W/intel-host-engine-guard.sh" --hot 80 "ri-m3-$STEP" "${GUARD_CAP:-300}" > "$W/logs/guard-$STEP.txt" 2>&1 &
guard=$!
timeout 1200 ssh intel-host env STEP_CARD=1 STEP_DEADLINE=$deadline \
    bash "$D/m3-step.sh" "$STEP" 900 "$IMG" bash /out/fc-tile-probe.sh "$LN" "$RU" > "$W/logs/step-$STEP.txt" 2>&1
grep 'rc=' "$W/logs/step-$STEP.txt" | tail -1
wait $guard
tail -1 "$W/logs/guard-$STEP.txt"
sleep 20
ssh intel-host bash "$D/ri-klog2.sh" "$STEP"
mkdir -p "$R"
ssh intel-host cat "$D/out-$STEP.txt" > "$R/$STEP-out.txt"
grep -E "^fc_tile_probe|^PASS|^FAIL|^RESULT|^PAIR|^E1|^E3|^T2|first difference|rror|refused|rc=" "$R/$STEP-out.txt" | cut -c1-200
