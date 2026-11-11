#!/bin/bash
# SPDX-License-Identifier: Apache-2.0
# Part of the Intel port: a hand-run prefill step (note 168 item 0, run 162) under the step rules: stage m3/STEP on
# intel-host from a reference prefill step's card/ and args plus today's m3-prefill.sh, cool to <= 43 C, precheck, the
# guard at --hot 80, m3-step.sh with both cards, the kernel log 20 s after, fetch the output and serve.log.
#   bash logs/bstep.sh STEP IMAGE_ID CFG REF_STEP "NAMES"     e.g. bash logs/bstep.sh B59 a5d67cc3fb28 nomtpcappfmjm64prof B58 "7k 20k"
set -u
STEP=$1; IMG=$2; CFG=$3; REF=$4; NAMES=$5; UT=${6:-}   # UT: a unitrace prefix for the engine (note 169 item 3)
W=$HOME/code/radiance-intel
D=/mnt/fast-storage/models/test/radiance-intel
R=$W/results-m3/card01
ssh intel-host mkdir -p "$D/m3/$STEP" || exit 1
ssh intel-host "tar -C $D/m3/$REF -cf - card flashnext-card1.args | tar -C $D/m3/$STEP -xf -" || exit 1
ssh intel-host tee "$D/m3/$STEP/m3-prefill.sh" < "$W/m3/intel-host/m3-prefill.sh" > /dev/null || exit 1
# never a libxe.so override from an older staging (B59's second try died on B58's)
ssh intel-host mv "$D/m3/$STEP/libxe.so" "$D/m3/$STEP/libxe.so.off" 2>/dev/null || true
timeout 1900 bash "$W/m3/cool43.sh" 1800 || exit 1
timeout 300 "$W/intel-host-engine-guard.sh" --precheck 40 || exit 1
deadline=$(( $(date +%s) + 1800 ))
"$W/intel-host-engine-guard.sh" --hot 80 "ri-m3-$STEP" 1000 > "$W/logs/guard-$STEP.txt" 2>&1 &
guard=$!
# run 169 (owner note 161: 2 or 4 CPUs a step): BSTEP_CPUS=4 in the environment runs the step at 4; BSTEP_SECCOMP=1
# runs it under m3/intel-host/seccomp-iouring.json (staged at $D/seccomp-iouring.json: podman's default profile plus the
# three io_uring syscalls, owner note 45), which libavx's n-gram gather needs
# owner note 175 (00:27, 11-04): libavx is the step default, so every engine step runs under the profile
SECC="STEP_SECCOMP=$D/seccomp-iouring.json"
timeout 1500 ssh intel-host env STEP_ENGINE=1 STEP_CARD=both STEP_CPUS=${BSTEP_CPUS:-2} STEP_NOTE7_CPUS2=1 STEP_DEADLINE=$deadline $SECC \
    bash "$D/m3-step.sh" "$STEP" 1200 "$IMG" env "PREFILL_NAMES='$NAMES'" "UT_PREFIX='$UT'" bash /out/m3-prefill.sh "$CFG" 1 > "$W/logs/step-$STEP.txt" 2>&1
grep 'rc=' "$W/logs/step-$STEP.txt" | tail -1
wait $guard
tail -1 "$W/logs/guard-$STEP.txt"
sleep 20
ssh intel-host bash "$D/ri-klog2.sh" "$STEP"
mkdir -p "$R/$STEP"
ssh intel-host cat "$D/out-$STEP.txt" > "$R/$STEP-out.txt"
ssh intel-host cat "$D/m3/$STEP/serve.log" > "$R/$STEP/serve.log"
grep -E 'prefill|first token|7k|20k|rc=' "$R/$STEP-out.txt" | tail -8 | cut -c1-160
