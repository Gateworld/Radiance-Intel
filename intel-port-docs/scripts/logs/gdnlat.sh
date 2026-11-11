#!/bin/bash
# SPDX-License-Identifier: Apache-2.0
# Part of the Intel port (run 283; briefs/2026-11-11-gdn-split-latency.md, KG17): logs/gdnstep.sh's form with
# m3/intel-host/gdn-lat.sh -- the VS 2 split with RADIANCE_XE_GDN_LAT 0 / 1 / 3 / 7 against the fused kernel: stage the
# tree's tests/xe_gdn_split_bench, its radiance_home/kernels/libxe.so and the script to m3/STEP on intel-host; cool to
# <= 43 C; precheck; the guard at --hot 80; m3-step.sh on card1 alone (no engine, no model); the kernel log 20 s after;
# the output fetched to results-m3/card01/STEP-out.txt. The gate: every hash = SPLIT 0's (the per-call ones and the
# anchor-seq schedule x SEQ_REPS); the timing line: >= 15 % per launch at T 4 -> the engine M-pair.
# STRESS (the coordinator's addition): anchor-seq repetitions at LAT 7 against SPLIT 0, every hash equal (N1's gate).
#   timeout 2400 bash logs/gdnlat.sh STEP IMAGE_ID [LAUNCHES=200] [LATS=0,1,3,7] [SEQ_REPS=20] [STRESS=500] [TREE=/srv/bulk/ri/build-m1l0neo2640]
set -u
STEP=$1; IMG=$2; LN=${3:-200}; LATS=${4:-0,1,3,7}; SEQ=${5:-20}; STRESS=${6:-500}; TREE=${7:-/srv/bulk/ri/build-m1l0neo2640}
W=$HOME/code/radiance-intel
D=/mnt/fast-storage/models/test/radiance-intel
R=$W/results-m3/card01
ssh intel-host mkdir -p "$D/m3/$STEP" || exit 1
scp -q "$TREE/tests/xe_gdn_split_bench" "intel-host:$D/m3/$STEP/xe_gdn_split_bench" || exit 1
scp -q "$TREE/radiance_home/kernels/libxe.so" "intel-host:$D/m3/$STEP/libxe.so" || exit 1
ssh intel-host tee "$D/m3/$STEP/gdn-lat.sh" < "$W/m3/intel-host/gdn-lat.sh" > /dev/null || exit 1
timeout 1900 bash "$W/m3/cool43.sh" 1800 || exit 1
timeout 300 "$W/intel-host-engine-guard.sh" --precheck 8 || exit 1
deadline=$(( $(date +%s) + 2000 ))
"$W/intel-host-engine-guard.sh" --hot 80 "ri-m3-$STEP" "${GUARD_CAP:-900}" > "$W/logs/guard-$STEP.txt" 2>&1 &
guard=$!
timeout 1800 ssh intel-host env STEP_CARD=1 STEP_DEADLINE=$deadline \
    bash "$D/m3-step.sh" "$STEP" 1500 "$IMG" bash /out/gdn-lat.sh "$LN" "$LATS" "$SEQ" "$STRESS" > "$W/logs/step-$STEP.txt" 2>&1
grep 'rc=' "$W/logs/step-$STEP.txt" | tail -1
wait $guard
tail -1 "$W/logs/guard-$STEP.txt"
sleep 20
ssh intel-host bash "$D/ri-klog2.sh" "$STEP"
mkdir -p "$R"
ssh intel-host cat "$D/out-$STEP.txt" > "$R/$STEP-out.txt"
grep -E "^===|^RESULT|refused|rror|rc=|recurrent_update|split_norm" "$R/$STEP-out.txt" | cut -c1-170
