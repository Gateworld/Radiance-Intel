#!/bin/bash
# SPDX-License-Identifier: Apache-2.0
# Part of the Intel port: logs/bstep.sh with a libxe.so override (run 166, constrained mode: no image build): the
# prefill B-step staged as bstep.sh stages it, then the given libxe.so copied into the step folder, which
# m3-prefill.sh installs over the image's copy before the engine starts ("libxe.so from /out" in serve.log).
#   bash logs/bstepx.sh STEP IMAGE_ID CFG REF_STEP "NAMES" LIBXE_SO [UT_PREFIX]
set -u
STEP=$1; IMG=$2; CFG=$3; REF=$4; NAMES=$5; SO=$6; UT=${7:-}
W=$HOME/code/radiance-intel
D=/mnt/fast-storage/models/test/radiance-intel
R=$W/results-m3/card01
ssh intel-host mkdir -p "$D/m3/$STEP" || exit 1
ssh intel-host "tar -C $D/m3/$REF -cf - card flashnext-card1.args | tar -C $D/m3/$STEP -xf -" || exit 1
ssh intel-host tee "$D/m3/$STEP/m3-prefill.sh" < "$W/m3/intel-host/m3-prefill.sh" > /dev/null || exit 1
ssh intel-host tee "$D/m3/$STEP/libxe.so" < "$SO" > /dev/null || exit 1
# run 253 (supervisor-approved): a split libxe (m3/split-link.sh) needs its libxe_b.so beside it in /out
[ -f "$(dirname "$SO")/libxe_b.so" ] && { ssh intel-host tee "$D/m3/$STEP/libxe_b.so" < "$(dirname "$SO")/libxe_b.so" > /dev/null || exit 1; }
ssh intel-host ls -la "$D/m3/$STEP/libxe.so" | cut -c1-100
timeout 1900 bash "$W/m3/cool43.sh" 1800 || exit 1
timeout 300 "$W/intel-host-engine-guard.sh" --precheck 40 || exit 1
deadline=$(( $(date +%s) + 1800 ))
"$W/intel-host-engine-guard.sh" --hot 80 "ri-m3-$STEP" 1000 > "$W/logs/guard-$STEP.txt" 2>&1 &
guard=$!
# run 220: under note 45's io_uring seccomp profile, as bstep.sh (owner note 175: libavx is the step default; without
# the profile its n-gram reader cannot make a ring and the engine stops at declare -- B161)
SECC="STEP_SECCOMP=$D/seccomp-iouring.json"
timeout 1500 ssh intel-host env STEP_ENGINE=1 STEP_CARD=both STEP_CPUS=2 STEP_NOTE7_CPUS2=1 STEP_DEADLINE=$deadline $SECC \
    bash "$D/m3-step.sh" "$STEP" 1200 "$IMG" env "PREFILL_NAMES='$NAMES'" "UT_PREFIX='$UT'" bash /out/m3-prefill.sh "$CFG" 1 > "$W/logs/step-$STEP.txt" 2>&1
grep 'rc=' "$W/logs/step-$STEP.txt" | tail -1
wait $guard
tail -1 "$W/logs/guard-$STEP.txt"
sleep 20
ssh intel-host bash "$D/ri-klog2.sh" "$STEP"
mkdir -p "$R/$STEP"
ssh intel-host cat "$D/out-$STEP.txt" > "$R/$STEP-out.txt"
ssh intel-host cat "$D/m3/$STEP/serve.log" > "$R/$STEP/serve.log"
grep -c "libxe.so from /out" "$R/$STEP/serve.log"
grep -E 'prefill|first token|7k|20k|rc=' "$R/$STEP-out.txt" | tail -8 | cut -c1-160
