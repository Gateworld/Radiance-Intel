#!/bin/bash
# SPDX-License-Identifier: Apache-2.0
# Part of the Intel port (run 280, owner note 219 B.0 / note 222): the LL all-reduce's in-kernel phase stamps on both
# cards, no engine, under the step rules (logs/estep.sh's form with STEP_CARD=both, as ar-bench's steps ran): stage
# m3/STEP on intel-host with the 2640 tree's tests/xe_allreduce_test, tests/xe_clock_probe and radiance_home/kernels/libxe.so
# (the build with XE_AR_STAMPS) and m3/intel-host/ar-stamps.sh; cool to <= 43 C; precheck; the guard at --hot 80 (GUARD_CAP,
# default 300 s); m3-step.sh on both cards; the kernel log 20 s after; the output fetched to results-m3/card01/STEP-out.txt.
#   timeout 1900 bash logs/arstep.sh STEP IMAGE_ID [N=2000] [TREE=/srv/bulk/ri/build-m1l0neo2640]
set -u
STEP=$1; IMG=$2; N=${3:-2000}; TREE=${4:-/srv/bulk/ri/build-m1l0neo2640}
W=$HOME/code/radiance-intel
D=/mnt/fast-storage/models/test/radiance-intel
R=$W/results-m3/card01
ssh intel-host mkdir -p "$D/m3/$STEP" || exit 1
scp -q "$TREE/tests/xe_allreduce_test" "intel-host:$D/m3/$STEP/xe_allreduce_test" || exit 1
scp -q "$TREE/tests/xe_clock_probe" "intel-host:$D/m3/$STEP/xe_clock_probe" || exit 1
scp -q "$TREE/radiance_home/kernels/libxe.so" "intel-host:$D/m3/$STEP/libxe.so" || exit 1
ssh intel-host tee "$D/m3/$STEP/ar-stamps.sh" < "$W/m3/intel-host/ar-stamps.sh" > /dev/null || exit 1
timeout 1900 bash "$W/m3/cool43.sh" 1800 || exit 1
timeout 300 "$W/intel-host-engine-guard.sh" --precheck 8 || exit 1
deadline=$(( $(date +%s) + 1500 ))
"$W/intel-host-engine-guard.sh" --hot 80 "ri-m3-$STEP" "${GUARD_CAP:-300}" > "$W/logs/guard-$STEP.txt" 2>&1 &
guard=$!
timeout 1200 ssh intel-host env STEP_CARD=both STEP_DEADLINE=$deadline \
    bash "$D/m3-step.sh" "$STEP" 900 "$IMG" bash /out/ar-stamps.sh "$N" > "$W/logs/step-$STEP.txt" 2>&1
grep 'rc=' "$W/logs/step-$STEP.txt" | tail -1
wait $guard
tail -1 "$W/logs/guard-$STEP.txt"
sleep 20
ssh intel-host bash "$D/ri-klog2.sh" "$STEP"
mkdir -p "$R"
ssh intel-host cat "$D/out-$STEP.txt" > "$R/$STEP-out.txt"
grep -E "^===|^RESULT|^STAMPS|^  |STALL|rror|refused|rc=" "$R/$STEP-out.txt" | cut -c1-260
