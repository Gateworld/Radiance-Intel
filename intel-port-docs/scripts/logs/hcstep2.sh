#!/bin/bash
# SPDX-License-Identifier: Apache-2.0
# Part of the Intel port (run 280, owner note 223 step 2: HC2). The hc prefill module's three builds against the per-row
# reference on card1 under the step rules (logs/hcstep.sh's form): stage m3/STEP on intel-host with the 2640 tree's
# tests/xe_hc_persist_test and its libxe.so (IGC 2.42.6), TODAY's module (m3/libxe_hcpf_228_unpatched.so, saved before
# the rebuild: the images' libxe_hcpf.so, sha256 9bba184b...c536) as libxe_hcpf_228u.so, the PATCHED module built by IGC
# 2.28.4 (build-m1l0's) as libxe_hcpf_228p.so and the PATCHED module built by IGC 2.42.6 (the 2640 tree's) as
# libxe_hcpf_2426p.so, plus m3/intel-host/hcpf-probe2.sh; cool to <= 43 C; precheck; the guard at --hot 80 (GUARD_CAP, default
# 300 s); m3-step.sh on card1 alone (no engine, no model), a host timeout a run inside; the kernel log 20 s after; the
# output fetched to results-m3/card01/STEP-out.txt. A hang or fault = STOP (owner note 223 step 2 / note 221 rule 3): no
# retry this run, the peer probe (logs/peer1.sh) next, the owner told; at most two faulting attempts in all.
#   timeout 1900 bash logs/hcstep2.sh STEP IMAGE_ID [RUN_TIMEOUT=120] [TREE=/srv/bulk/ri/build-m1l0neo2640] [MOD3]
# run 283 (owner note 223's fallback, HC3): MOD3 names the THIRD module's file in place of the tree's own -- the hc module
# built by IGC v2.43.9 in the 243 tree (/srv/bulk/ri/build-m1l0neo243/radiance_home/kernels/xe/libxe_hcpf.so) -- staged
# under the probe's third slot (libxe_hcpf_2426p.so: the probe's names are fixed; the step's header says what it holds).
set -u
STEP=$1; IMG=$2; TO=${3:-120}; TREE=${4:-/srv/bulk/ri/build-m1l0neo2640}; MOD3=${5:-$TREE/radiance_home/kernels/xe/libxe_hcpf.so}
echo "third module (the probe's 2426p slot): $MOD3"; sha256sum "$MOD3" || exit 1
W=$HOME/code/radiance-intel
D=/mnt/fast-storage/models/test/radiance-intel
R=$W/results-m3/card01
ssh intel-host mkdir -p "$D/m3/$STEP" || exit 1
scp -q "$TREE/tests/xe_hc_persist_test" "intel-host:$D/m3/$STEP/xe_hc_persist_test" || exit 1
scp -q "$TREE/radiance_home/kernels/libxe.so" "intel-host:$D/m3/$STEP/libxe.so" || exit 1
scp -q "$W/m3/libxe_hcpf_228_unpatched.so" "intel-host:$D/m3/$STEP/libxe_hcpf_228u.so" || exit 1
scp -q "$W/build-m1l0/radiance_home/kernels/xe/libxe_hcpf.so" "intel-host:$D/m3/$STEP/libxe_hcpf_228p.so" || exit 1
scp -q "$MOD3" "intel-host:$D/m3/$STEP/libxe_hcpf_2426p.so" || exit 1
ssh intel-host tee "$D/m3/$STEP/hcpf-probe2.sh" < "$W/m3/intel-host/hcpf-probe2.sh" > /dev/null || exit 1
timeout 1900 bash "$W/m3/cool43.sh" 1800 || exit 1
timeout 300 "$W/intel-host-engine-guard.sh" --precheck 8 || exit 1
deadline=$(( $(date +%s) + 1200 ))
"$W/intel-host-engine-guard.sh" --hot 80 "ri-m3-$STEP" "${GUARD_CAP:-300}" > "$W/logs/guard-$STEP.txt" 2>&1 &
guard=$!
timeout 1000 ssh intel-host env STEP_CARD=1 STEP_DEADLINE=$deadline \
    bash "$D/m3-step.sh" "$STEP" 900 "$IMG" bash /out/hcpf-probe2.sh "$TO" > "$W/logs/step-$STEP.txt" 2>&1
grep 'rc=' "$W/logs/step-$STEP.txt" | tail -1
wait $guard
tail -1 "$W/logs/guard-$STEP.txt"
sleep 20
ssh intel-host bash "$D/ri-klog2.sh" "$STEP"
mkdir -p "$R"
ssh intel-host cat "$D/out-$STEP.txt" > "$R/$STEP-out.txt"
grep -E "^===|^---|^RESULT|^STOP|rc=|matrix units|PF module|differ|rror|refused|hcpf" "$R/$STEP-out.txt" | cut -c1-200
