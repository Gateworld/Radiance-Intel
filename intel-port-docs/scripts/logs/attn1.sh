#!/bin/bash
# SPDX-License-Identifier: Apache-2.0
# Part of the Intel port: note 168 item 1 (run 165) -- the block-sparse attention probe (probe2/attn_tile_probe.cpp,
# built by m1/build.sh attnprobe) on a card inside the runtime image, under the step rules: stage m3/STEP on intel-host
# (the binary + attn.sh), cool to <= 43 C, precheck, the guard at --hot 80, m3-step.sh with both cards (the probe
# takes device 0), the kernel log 20 s after, fetch the output.
#   bash logs/attn1.sh STEP IMAGE_ID [L=5604] [iters=12]
set -u
STEP=$1; IMG=$2; L=${3:-5604}; IT=${4:-12}
W=$HOME/code/radiance-intel
D=/mnt/fast-storage/models/test/radiance-intel
ssh intel-host mkdir -p "$D/m3/$STEP" || exit 1
ssh intel-host tee "$D/m3/$STEP/attn_tile_probe" < "$W/logs/attnprobe/attn_tile_probe" > /dev/null || exit 1
ssh intel-host chmod +x "$D/m3/$STEP/attn_tile_probe" || exit 1
printf '#!/bin/bash\nldd /out/attn_tile_probe | grep -E "not found|sycl|ze_loader"\ntimeout 600 /out/attn_tile_probe gpu %s %s 1\n' "$L" "$IT" > "$W/logs/attnprobe/attn.sh"
ssh intel-host tee "$D/m3/$STEP/attn.sh" < "$W/logs/attnprobe/attn.sh" > /dev/null || exit 1
timeout 1900 bash "$W/m3/cool43.sh" 1800 || exit 1
timeout 300 "$W/intel-host-engine-guard.sh" --precheck 2 || exit 1
deadline=$(( $(date +%s) + 900 ))
"$W/intel-host-engine-guard.sh" --hot 80 "ri-m3-$STEP" 300 > "$W/logs/guard-$STEP.txt" 2>&1 &
guard=$!
timeout 800 ssh intel-host env STEP_CARD=both STEP_CPUS=2 STEP_NOTE7_CPUS2=1 STEP_DEADLINE=$deadline \
    bash "$D/m3-step.sh" "$STEP" 700 "$IMG" bash /out/attn.sh > "$W/logs/step-$STEP.txt" 2>&1
grep 'rc=' "$W/logs/step-$STEP.txt" | tail -1
wait $guard
tail -1 "$W/logs/guard-$STEP.txt"
sleep 20
ssh intel-host bash "$D/ri-klog2.sh" "$STEP"
ssh intel-host cat "$D/out-$STEP.txt" > "$W/results-m3/card01/$STEP-out.txt"
grep -E "device|warm-up|kernel:|verify|PASS|FAIL|failed|rc=" "$W/results-m3/card01/$STEP-out.txt" | cut -c1-160
