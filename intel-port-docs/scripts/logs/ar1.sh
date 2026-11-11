#!/bin/bash
# SPDX-License-Identifier: Apache-2.0
# Part of the Intel port: run 170 (owner note 172 d) -- libxe's two-rank all-reduce test on both cards inside the
# runtime image (its /opt/ri/tests/xe_allreduce_test over the image's libxe.so), under the step rules: stage m3/STEP
# on intel-host, cool to <= 43 C, precheck, the guard at --hot 80, m3-step.sh with both cards, the kernel log 20 s after,
# the output fetched. PROTO selects RADIANCE_XE_AR_PROTO (6: the bandwidth form for messages >= 256 KiB, the LL
# form 5 below); the test's large peer cases (bf16 262,144 in place, f32 65,536 into y) are the ones the old flag
# form failed (AB1 / AB2: "ranks differ"), so ITERS of them passing is the form's ordering proof on the card.
#   bash logs/ar1.sh STEP IMAGE_ID [ITERS=200] [PROTO=6] [TEAR=0]     TEAR=N: XE_AR_TEST_TEAR=N, the forced tear
set -u
STEP=$1; IMG=$2; IT=${3:-200}; PROTO=${4:-6}; TEAR=${5:-0}
W=$HOME/code/radiance-intel
D=/mnt/fast-storage/models/test/radiance-intel
ssh intel-host mkdir -p "$D/m3/$STEP" || exit 1
printf '#!/bin/bash\nexport RADIANCE_XE_AR_PROTO=%s XE_AR_TEST_TEAR=%s\necho "proto $RADIANCE_XE_AR_PROTO tear $XE_AR_TEST_TEAR iters %s"\ntimeout 1500 /opt/ri/tests/xe_allreduce_test /opt/ri/radiance_home/kernels/libxe.so %s 2>&1 | grep -E "device|peer access|case |RESULT|FAIL|stalled|differ|canar|re-reads|rc" | tail -60\n' "$PROTO" "$TEAR" "$IT" "$IT" > "$W/logs/ar1-inner.sh"
ssh intel-host tee "$D/m3/$STEP/ar.sh" < "$W/logs/ar1-inner.sh" > /dev/null || exit 1
timeout 1900 bash "$W/m3/cool43.sh" 1800 || exit 1
timeout 300 "$W/intel-host-engine-guard.sh" --precheck 2 || exit 1
deadline=$(( $(date +%s) + 1800 ))
"$W/intel-host-engine-guard.sh" --hot 80 "ri-m3-$STEP" 1000 > "$W/logs/guard-$STEP.txt" 2>&1 &
guard=$!
timeout 1900 ssh intel-host env STEP_CARD=both STEP_CPUS=2 STEP_NOTE7_CPUS2=1 STEP_DEADLINE=$deadline \
    bash "$D/m3-step.sh" "$STEP" 1600 "$IMG" bash /out/ar.sh > "$W/logs/step-$STEP.txt" 2>&1
grep 'rc=' "$W/logs/step-$STEP.txt" | tail -1
wait $guard
tail -1 "$W/logs/guard-$STEP.txt"
sleep 20
ssh intel-host bash "$D/ri-klog2.sh" "$STEP"
ssh intel-host cat "$D/out-$STEP.txt" > "$W/results-m3/card01/$STEP-out.txt"
grep -E "proto|device|peer|RESULT|FAIL|stalled|differ|canar|rc=" "$W/results-m3/card01/$STEP-out.txt" | tail -30 | cut -c1-160
