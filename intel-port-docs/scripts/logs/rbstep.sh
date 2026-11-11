#!/bin/bash
# SPDX-License-Identifier: Apache-2.0
# Part of the Intel port (run 279, owner note 216 item 3: the router redesign's bench first): xe_router_bench on card1
# under the step rules (logs/qkstep.sh's form, which is mbstep.sh's): stage m3/STEP on intel-host with the 2640 tree's
# tests/xe_router_bench, its radiance_home/kernels/libxe.so (the bench loads /out/libxe.so, so the variants need no new
# image) and m3/intel-host/router-bench.sh, cool to <= 43 C, precheck, the guard at --hot 80 (GUARD_CAP, default 300 s: the
# bench's twelve runs take a few minutes), m3-step.sh on card1 alone (no engine), the kernel log 20 s after, the output
# fetched to results-m3/card01/STEP-out.txt.
#   timeout 1900 bash logs/rbstep.sh STEP IMAGE_ID [TREE=/srv/bulk/ri/build-m1l0neo2640]
set -u
STEP=$1; IMG=$2; TREE=${3:-/srv/bulk/ri/build-m1l0neo2640}
W=$HOME/code/radiance-intel
D=/mnt/fast-storage/models/test/radiance-intel
R=$W/results-m3/card01
ssh intel-host mkdir -p "$D/m3/$STEP" || exit 1
scp -q "$TREE/tests/xe_router_bench" "intel-host:$D/m3/$STEP/xe_router_bench" || exit 1
scp -q "$TREE/radiance_home/kernels/libxe.so" "intel-host:$D/m3/$STEP/libxe.so" || exit 1
ssh intel-host tee "$D/m3/$STEP/router-bench.sh" < "$W/m3/intel-host/router-bench.sh" > /dev/null || exit 1
timeout 1900 bash "$W/m3/cool43.sh" 1800 || exit 1
timeout 300 "$W/intel-host-engine-guard.sh" --precheck 8 || exit 1
deadline=$(( $(date +%s) + 1500 ))
"$W/intel-host-engine-guard.sh" --hot 80 "ri-m3-$STEP" "${GUARD_CAP:-300}" > "$W/logs/guard-$STEP.txt" 2>&1 &
guard=$!
timeout 1500 ssh intel-host env STEP_CARD=1 STEP_DEADLINE=$deadline \
    bash "$D/m3-step.sh" "$STEP" 1200 "$IMG" bash /out/router-bench.sh > "$W/logs/step-$STEP.txt" 2>&1
grep 'rc=' "$W/logs/step-$STEP.txt" | tail -1
wait $guard
tail -1 "$W/logs/guard-$STEP.txt"
sleep 20
ssh intel-host bash "$D/ri-klog2.sh" "$STEP"
ssh intel-host cat "$D/out-$STEP.txt" > "$R/$STEP-out.txt"
grep -E '^===|^RESULT|rror|refused|rc=' "$R/$STEP-out.txt" | cut -c1-200
