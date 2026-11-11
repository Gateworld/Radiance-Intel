#!/bin/bash
# SPDX-License-Identifier: Apache-2.0
# Part of the Intel port: note 168 item 2 (run 164) -- the matrix-unit form of format C's expert GEMM as the prefill
# default (image 188) checked on the cards: B89 the 7K and 20K first tokens (against B60 = the run form, B64 = the
# f16 matrix form), KDP1 the KL replay of the exact path's prompts (`kld-gl.sh dpas`, note 149's bar), then M250 the
# single-prompt decode half (MTP on; decode stays on the run form, the text must equal M229's).
#   bash logs/dpas_steps3.sh IMAGE_ID
set -u
IMG=$1
W=$HOME/code/radiance-intel
D=/mnt/fast-storage/models/test/radiance-intel
R=$W/results-m3/card01
timeout 2700 bash "$W/logs/bstep.sh" B89 "$IMG" nomtpcappfmjm64tr B58 "7k 20k" > "$W/logs/bstep-B89.txt" 2>&1
grep -h -E "prefill-7k|prefill-20k|klog2|rc=" "$W/logs/bstep-B89.txt" | cut -c1-120
# the KL replay: KDP1 staged by hand (TSK2's args + gl-replay.jsonl, today's kld-gl.sh)
timeout 1900 bash "$W/m3/cool43.sh" 1800 || exit 1
timeout 300 "$W/intel-host-engine-guard.sh" --precheck 40 || exit 1
deadline=$(( $(date +%s) + 1800 ))
"$W/intel-host-engine-guard.sh" --hot 80 "ri-m3-KDP1" 1000 > "$W/logs/guard-KDP1.txt" 2>&1 &
guard=$!
timeout 1500 ssh intel-host env STEP_ENGINE=1 STEP_CARD=both STEP_CPUS=2 STEP_NOTE7_CPUS2=1 STEP_DEADLINE=$deadline \
    bash "$D/m3-step.sh" KDP1 1200 "$IMG" bash /out/kld-gl.sh dpas > "$W/logs/step-KDP1.txt" 2>&1
grep 'rc=' "$W/logs/step-KDP1.txt" | tail -1
wait $guard
tail -1 "$W/logs/guard-KDP1.txt"
sleep 20
ssh intel-host bash "$D/ri-klog2.sh" KDP1
mkdir -p "$R/KDP1"
ssh intel-host cat "$D/out-KDP1.txt" > "$R/KDP1-out.txt"
ssh intel-host cat "$D/m3/KDP1/kld-dpas.json" > "$R/KDP1/kld-dpas.json"
tail -14 "$R/KDP1-out.txt" | cut -c1-160
timeout 3600 bash "$W/m3/mstep.sh" M250 "$IMG" svcmtpx-gnt0-hc8-rp-gnx M229 svcmtpx-gnt0-hc8-rp-gnx > "$W/logs/mstep-M250.txt" 2>&1
grep -hE "run 2:|klog2|identical|differ|peak Tctl" "$W/logs/mstep-M250.txt" | cut -c1-150
