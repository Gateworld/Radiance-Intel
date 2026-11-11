#!/bin/bash
# SPDX-License-Identifier: Apache-2.0
# Part of the Intel port (run 279, owner note 221 rule 2): the smallest reproducer of the hc prefill module's IGC 2.42.6
# hang -- libxe's hc row test on card1 at 64 and 4,096 rows with the prefill matrix-unit module, the IGC 2.28.4 build of
# the module against the IGC 2.42.6 build, bytes compared (m3/intel-host/hcpf-probe.sh) -- under the step rules (logs/estep.sh's
# form): stage m3/STEP on intel-host with the 2640 tree's tests/xe_hc_persist_test (RI_HC_ROWS) and its libxe.so (IGC 2.42.6),
# the two module builds as libxe_hcpf_228.so (build-m1l0's, IGC 2.28.4 -- the images' module today) and libxe_hcpf_2426.so
# (the 2640 tree's own), chosen by RADIANCE_XE_HC_PF_MOD_PATH; cool to <= 43 C; precheck; the guard at --hot 80
# (GUARD_CAP, default 300 s); m3-step.sh on card1 alone (no engine, no model), a host timeout a run inside; the kernel
# log 20 s after; the output fetched to results-m3/card01/STEP-out.txt. A hang or fault = STOP (owner note 221 rule 3):
# no retry this run, the peer probe and the reset procedure, the owner told before any second attempt.
#   timeout 1900 bash logs/hcstep.sh STEP IMAGE_ID [RUN_TIMEOUT=120] [TREE=/srv/bulk/ri/build-m1l0neo2640]
set -u
STEP=$1; IMG=$2; TO=${3:-120}; TREE=${4:-/srv/bulk/ri/build-m1l0neo2640}
W=$HOME/code/radiance-intel
D=/mnt/fast-storage/models/test/radiance-intel
R=$W/results-m3/card01
ssh intel-host mkdir -p "$D/m3/$STEP" || exit 1
scp -q "$TREE/tests/xe_hc_persist_test" "intel-host:$D/m3/$STEP/xe_hc_persist_test" || exit 1
scp -q "$TREE/radiance_home/kernels/libxe.so" "intel-host:$D/m3/$STEP/libxe.so" || exit 1
scp -q "$W/build-m1l0/radiance_home/kernels/xe/libxe_hcpf.so" "intel-host:$D/m3/$STEP/libxe_hcpf_228.so" || exit 1
scp -q "$TREE/radiance_home/kernels/xe/libxe_hcpf.so" "intel-host:$D/m3/$STEP/libxe_hcpf_2426.so" || exit 1
ssh intel-host tee "$D/m3/$STEP/hcpf-probe.sh" < "$W/m3/intel-host/hcpf-probe.sh" > /dev/null || exit 1
timeout 1900 bash "$W/m3/cool43.sh" 1800 || exit 1
timeout 300 "$W/intel-host-engine-guard.sh" --precheck 8 || exit 1
deadline=$(( $(date +%s) + 1200 ))
"$W/intel-host-engine-guard.sh" --hot 80 "ri-m3-$STEP" "${GUARD_CAP:-300}" > "$W/logs/guard-$STEP.txt" 2>&1 &
guard=$!
timeout 1000 ssh intel-host env STEP_CARD=1 STEP_DEADLINE=$deadline \
    bash "$D/m3-step.sh" "$STEP" 900 "$IMG" bash /out/hcpf-probe.sh "$TO" > "$W/logs/step-$STEP.txt" 2>&1
grep 'rc=' "$W/logs/step-$STEP.txt" | tail -1
wait $guard
tail -1 "$W/logs/guard-$STEP.txt"
sleep 20
ssh intel-host bash "$D/ri-klog2.sh" "$STEP"
mkdir -p "$R"
ssh intel-host cat "$D/out-$STEP.txt" > "$R/$STEP-out.txt"
grep -E "^===|^---|^RESULT|^STOP|rc=|matrix units|PF module|differ|rror|refused|hcpf" "$R/$STEP-out.txt" | cut -c1-200
