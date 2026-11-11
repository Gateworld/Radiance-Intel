#!/bin/bash
# SPDX-License-Identifier: Apache-2.0
# Part of the Intel port: owner notes 169 item 3 / 171 (run 164) -- an M-step (m3/mstep.sh's form) with the engine
# wrapped by a unitrace prefix (UT_PREFIX, read by m3-tps.sh) in the counters test image ri-m3ut:N, for the decode
# kernels' hardware counters; the per-instance metrics land in m3/STEP/perfquery.metrics.<pid>.csv on intel-host and are
# fetched beside the usual files. The tok/s under the query is not a measurement (the query adds ~50 %).
#   bash logs/mutstep.sh STEP IMAGE_ID CFG REF_STEP REF_CFG "UT_PREFIX"
set -u
STEP=$1; IMG=$2; CFG=$3; REF=$4; REFCFG=$5; UT=$6
W=$HOME/code/radiance-intel
D=/mnt/fast-storage/models/test/radiance-intel
R=$W/results-m3/card01
ssh intel-host mkdir "$D/m3/$STEP" || exit 1
ssh intel-host cp -r "$D/m3/$REF/card" "$D/m3/$REF/flashnext-card1.args" "$D/m3/$STEP/" || exit 1
ssh intel-host tee "$D/m3/$STEP/m3-tps.sh" < "$W/m3/intel-host/m3-tps.sh" > /dev/null || exit 1
timeout 1900 bash "$W/m3/cool43.sh" 1800 || exit 1
timeout 300 "$W/intel-host-engine-guard.sh" --precheck 40 || exit 1
deadline=$(( $(date +%s) + 1800 ))
"$W/intel-host-engine-guard.sh" --hot 80 "ri-m3-$STEP" 1000 > "$W/logs/guard-$STEP.txt" 2>&1 &
guard=$!
# run 206: owner note 175 (libavx the step default) -- every engine step runs under the io_uring seccomp profile, as
# m3/mstep.sh's (U1 without it: libavx's n-gram gather could not make a ring and the engine refused to start)
timeout 1100 ssh intel-host env STEP_ENGINE=1 STEP_CARD=both STEP_CPUS=2 STEP_NOTE7_CPUS2=1 STEP_DEADLINE=$deadline \
    STEP_SECCOMP=$D/seccomp-iouring.json \
    bash "$D/m3-step.sh" "$STEP" 900 "$IMG" env "UT_PREFIX='$UT'" bash /out/m3-tps.sh "$CFG" 2 > "$W/logs/step-$STEP.txt" 2>&1
grep 'rc=' "$W/logs/step-$STEP.txt" | tail -1
wait $guard
tail -1 "$W/logs/guard-$STEP.txt"
sleep 20
ssh intel-host bash "$D/ri-klog2.sh" "$STEP"
mkdir -p "$R/$STEP/$CFG"
ssh intel-host cat "$D/out-$STEP.txt" > "$R/$STEP-out.txt"
ssh intel-host cat "$D/m3/$STEP/$CFG/tps-1.sse" > "$R/$STEP/$CFG/tps-1.sse"
ssh intel-host cat "$D/m3/$STEP/$CFG/tps-2.sse" > "$R/$STEP/$CFG/tps-2.sse"
ssh intel-host cat "$D/m3/$STEP/$CFG/serve.log" > "$R/$STEP/$CFG/serve.log"
ssh intel-host ls -la "$D/m3/$STEP/" | grep perfquery
ssh intel-host "cat $D/m3/$STEP/perfquery.metrics.*.csv" > "$R/$STEP/perfquery.metrics.csv"
# run 206: `unitrace --stall-sampling -o /out/stall` writes stall.<pid> and stall.metrics.<pid>*
scp -q "intel-host:$D/m3/$STEP/stall*" "$R/$STEP/"
grep -E 'up=|run [12]:' "$R/$STEP-out.txt"
python3 -I "$W/m3/tools/sse-text.py" "$R/$STEP/$CFG/tps-1.sse" "$R/$REF/$REFCFG/tps-1.sse" | tail -1
python3 -I "$W/m3/tools/sse-text.py" "$R/$STEP/$CFG/tps-2.sse" "$R/$REF/$REFCFG/tps-2.sse" | tail -1
