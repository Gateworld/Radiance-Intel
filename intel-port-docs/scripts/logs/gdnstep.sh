#!/bin/bash
# SPDX-License-Identifier: Apache-2.0
# Part of the Intel port (run 283, owner note 224 item 2: the GDN decode update redesign's gate). xe_gdn_split_bench on
# card1 under the step rules (logs/estep.sh's form): stage m3/STEP on intel-host with the 2640 tree's tests/xe_gdn_split_bench,
# its radiance_home/kernels/libxe.so (the kernel under test; the bench loads /out/libxe.so, so no new image) and
# m3/intel-host/gdn-split.sh; cool to <= 43 C; precheck; the guard at --hot 80 (GUARD_CAP, default 600 s); m3-step.sh on
# card1 alone (no engine, no model); the kernel log 20 s after; the output fetched to results-m3/card01/STEP-out.txt.
# gdn-split.sh runs both state forms at every SPLIT (the per-call hashes and the time a call) and, with SEQ_REPS > 0,
# the anchor-seq schedule at every SPLIT compared with SPLIT 0's hashes. The gate: every hash equal to SPLIT 0's.
#   timeout 2400 bash logs/gdnstep.sh STEP IMAGE_ID [LAUNCHES=200] [SPLITS=0,2,4,8,16] [SEQ_REPS=20] [TREE=/srv/bulk/ri/build-m1l0neo2640]
# (SPLITS comma-separated: a quoted space list is re-split by the remote shell -- KG14 ran SPLIT 0 alone)
set -u
STEP=$1; IMG=$2; LN=${3:-200}; SPLITS=${4:-0,2,4,8,16}; SEQ=${5:-20}; TREE=${6:-/srv/bulk/ri/build-m1l0neo2640}
W=$HOME/code/radiance-intel
D=/mnt/fast-storage/models/test/radiance-intel
R=$W/results-m3/card01
ssh intel-host mkdir -p "$D/m3/$STEP" || exit 1
scp -q "$TREE/tests/xe_gdn_split_bench" "intel-host:$D/m3/$STEP/xe_gdn_split_bench" || exit 1
scp -q "$TREE/radiance_home/kernels/libxe.so" "intel-host:$D/m3/$STEP/libxe.so" || exit 1
ssh intel-host tee "$D/m3/$STEP/gdn-split.sh" < "$W/m3/intel-host/gdn-split.sh" > /dev/null || exit 1
timeout 1900 bash "$W/m3/cool43.sh" 1800 || exit 1
timeout 300 "$W/intel-host-engine-guard.sh" --precheck 8 || exit 1
deadline=$(( $(date +%s) + 2000 ))
"$W/intel-host-engine-guard.sh" --hot 80 "ri-m3-$STEP" "${GUARD_CAP:-600}" > "$W/logs/guard-$STEP.txt" 2>&1 &
guard=$!
timeout 1800 ssh intel-host env STEP_CARD=1 STEP_DEADLINE=$deadline \
    bash "$D/m3-step.sh" "$STEP" 1500 "$IMG" bash /out/gdn-split.sh "$LN" "$SPLITS" "$SEQ" > "$W/logs/step-$STEP.txt" 2>&1
grep 'rc=' "$W/logs/step-$STEP.txt" | tail -1
wait $guard
tail -1 "$W/logs/guard-$STEP.txt"
sleep 20
ssh intel-host bash "$D/ri-klog2.sh" "$STEP"
mkdir -p "$R"
ssh intel-host cat "$D/out-$STEP.txt" > "$R/$STEP-out.txt"
grep -E "^===|^RESULT|refused|rror|rc=" "$R/$STEP-out.txt" | cut -c1-200
