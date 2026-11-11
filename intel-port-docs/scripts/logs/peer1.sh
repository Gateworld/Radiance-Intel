#!/bin/bash
# SPDX-License-Identifier: Apache-2.0
# Part of the Intel port: note 163 item 2 (run 161) -- the peer-path probe on both cards inside the runtime image,
# under the step rules: stage m3/STEP on intel-host (the binary + peer.sh), cool to <= 43 C, precheck, the guard at
# --hot 80, m3-step.sh with both cards, the kernel log 20 s after, fetch the output.
#   bash logs/peer1.sh STEP IMAGE_ID [iters]
set -u
STEP=$1; IMG=$2; IT=${3:-2000}
W=$HOME/code/radiance-intel
D=/mnt/fast-storage/models/test/radiance-intel
ssh intel-host mkdir -p "$D/m3/$STEP" || exit 1
ssh intel-host tee "$D/m3/$STEP/m2b_peer" < "$W/logs/peer/m2b_peer" > /dev/null || exit 1
ssh intel-host chmod +x "$D/m3/$STEP/m2b_peer" || exit 1
printf '#!/bin/bash\nldd /out/m2b_peer | grep -E "not found|sycl|ze_loader"\n/out/m2b_peer gpu %s\n' "$IT" > "$W/logs/peer/peer.sh"
ssh intel-host tee "$D/m3/$STEP/peer.sh" < "$W/logs/peer/peer.sh" > /dev/null || exit 1
timeout 1900 bash "$W/m3/cool43.sh" 1800 || exit 1
timeout 300 "$W/intel-host-engine-guard.sh" --precheck 2 || exit 1
deadline=$(( $(date +%s) + 900 ))
"$W/intel-host-engine-guard.sh" --hot 80 "ri-m3-$STEP" 300 > "$W/logs/guard-$STEP.txt" 2>&1 &
guard=$!
timeout 700 ssh intel-host env STEP_CARD=both STEP_CPUS=2 STEP_NOTE7_CPUS2=1 STEP_DEADLINE=$deadline \
    bash "$D/m3-step.sh" "$STEP" 500 "$IMG" bash /out/peer.sh > "$W/logs/step-$STEP.txt" 2>&1
grep 'rc=' "$W/logs/step-$STEP.txt" | tail -1
wait $guard
tail -1 "$W/logs/guard-$STEP.txt"
sleep 20
ssh intel-host bash "$D/ri-klog2.sh" "$STEP"
ssh intel-host cat "$D/out-$STEP.txt" > "$W/results-m3/card01/$STEP-out.txt"
cat "$W/results-m3/card01/$STEP-out.txt"
