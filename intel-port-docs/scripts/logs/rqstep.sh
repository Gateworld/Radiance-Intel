#!/bin/bash
# SPDX-License-Identifier: Apache-2.0
# Part of the Intel port (run 273, the decode step's pairs 1 and 2): the two paired launches' row checks on card1 under
# the step rules (logs/qkstep.sh's form, which is mbstep.sh's): stage m3/STEP on intel-host with the 2640 tree's
# tests/xe_router_quant_test and tests/xe_gate_quant_test and m3/intel-host/rq-probe.sh, cool to <= 43 C, precheck, the
# guard at --hot 80 (GUARD_CAP, default 120 s: no engine, the container is gone in seconds), m3-step.sh on card1
# alone, the kernel log 20 s after, the output fetched to results-m3/card01/STEP-out.txt. The image's libxe.so must
# carry the router_topk_scatter_quant and gated_quant_pair rows (image 300 or later).
#   timeout 1900 bash logs/rqstep.sh STEP IMAGE_ID [TREE=/srv/bulk/ri/build-m1l0neo2640]
set -u
STEP=$1; IMG=$2; TREE=${3:-/srv/bulk/ri/build-m1l0neo2640}
W=$HOME/code/radiance-intel
D=/mnt/fast-storage/models/test/radiance-intel
R=$W/results-m3/card01
ssh intel-host mkdir -p "$D/m3/$STEP" || exit 1
scp -q "$TREE/tests/xe_router_quant_test" "intel-host:$D/m3/$STEP/xe_router_quant_test" || exit 1
scp -q "$TREE/tests/xe_gate_quant_test" "intel-host:$D/m3/$STEP/xe_gate_quant_test" || exit 1
ssh intel-host tee "$D/m3/$STEP/rq-probe.sh" < "$W/m3/intel-host/rq-probe.sh" > /dev/null || exit 1
timeout 1900 bash "$W/m3/cool43.sh" 1800 || exit 1
timeout 300 "$W/intel-host-engine-guard.sh" --precheck 8 || exit 1
deadline=$(( $(date +%s) + 900 ))
"$W/intel-host-engine-guard.sh" --hot 80 "ri-m3-$STEP" "${GUARD_CAP:-120}" > "$W/logs/guard-$STEP.txt" 2>&1 &
guard=$!
timeout 600 ssh intel-host env STEP_CARD=1 STEP_DEADLINE=$deadline \
    bash "$D/m3-step.sh" "$STEP" 300 "$IMG" bash /out/rq-probe.sh > "$W/logs/step-$STEP.txt" 2>&1
grep 'rc=' "$W/logs/step-$STEP.txt" | tail -1
wait $guard
tail -1 "$W/logs/guard-$STEP.txt"
sleep 20
ssh intel-host bash "$D/ri-klog2.sh" "$STEP"
ssh intel-host cat "$D/out-$STEP.txt" > "$R/$STEP-out.txt"
grep -E '^RESULT|^PASS|^FAIL|^SKIP|rc=' "$R/$STEP-out.txt" | cut -c1-220
