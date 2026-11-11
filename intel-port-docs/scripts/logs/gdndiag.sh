#!/bin/bash
# SPDX-License-Identifier: Apache-2.0
# Part of the Intel port (run 284: the coordinator's diagnostic sweep of the GDN slice-group kernel). logs/gdnstep.sh's
# form with m3/intel-host/gdn-diag.sh: stage m3/STEP on intel-host with the tree's tests/xe_gdn_split_bench and its
# radiance_home/kernels/libxe.so -- the tree must be a BENCH BUILD configured with XE_GDN_SG_DIAG=ON (the timing-only
# variants V1-V6; refused otherwise, the knob would be ignored) -- and gdn-diag.sh; cool to <= 43 C; precheck; the
# guard at --hot 80; m3-step.sh on card1 alone (no engine, no model); the kernel log 20 s after; the output fetched to
# results-m3/card01/STEP-out.txt. No image carries this libxe.so (make-rt-neo2.sh refuses the tree while it is ON).
#   timeout 2400 bash logs/gdndiag.sh STEP IMAGE_ID [LAUNCHES=200] [TS=1,4] [TREE=/srv/bulk/ri/build-m1l0neo2640]
set -u
STEP=$1; IMG=$2; LN=${3:-200}; TS=${4:-1,4}; TREE=${5:-/srv/bulk/ri/build-m1l0neo2640}
W=$HOME/code/radiance-intel
D=/mnt/fast-storage/models/test/radiance-intel
R=$W/results-m3/card01
grep -q "^XE_GDN_SG_DIAG:BOOL=ON" "$TREE/CMakeCache.txt" || { echo "$TREE is not a diag build (XE_GDN_SG_DIAG is not ON)"; exit 1; }
ssh intel-host mkdir -p "$D/m3/$STEP" || exit 1
scp -q "$TREE/tests/xe_gdn_split_bench" "intel-host:$D/m3/$STEP/xe_gdn_split_bench" || exit 1
scp -q "$TREE/radiance_home/kernels/libxe.so" "intel-host:$D/m3/$STEP/libxe.so" || exit 1
ssh intel-host tee "$D/m3/$STEP/gdn-diag.sh" < "$W/m3/intel-host/gdn-diag.sh" > /dev/null || exit 1
timeout 1900 bash "$W/m3/cool43.sh" 1800 || exit 1
timeout 300 "$W/intel-host-engine-guard.sh" --precheck 8 || exit 1
deadline=$(( $(date +%s) + 2000 ))
"$W/intel-host-engine-guard.sh" --hot 80 "ri-m3-$STEP" "${GUARD_CAP:-600}" > "$W/logs/guard-$STEP.txt" 2>&1 &
guard=$!
timeout 1800 ssh intel-host env STEP_CARD=1 STEP_DEADLINE=$deadline \
    bash "$D/m3-step.sh" "$STEP" 1500 "$IMG" bash /out/gdn-diag.sh "$LN" "$TS" > "$W/logs/step-$STEP.txt" 2>&1
grep 'rc=' "$W/logs/step-$STEP.txt" | tail -1
wait $guard
tail -1 "$W/logs/guard-$STEP.txt"
sleep 20
ssh intel-host bash "$D/ri-klog2.sh" "$STEP"
mkdir -p "$R"
ssh intel-host cat "$D/out-$STEP.txt" > "$R/$STEP-out.txt"
grep -E "^===|^RESULT|refused|rror|rc=|ru_sg|recurrent_update|split_norm" "$R/$STEP-out.txt" | cut -c1-170
