#!/bin/bash
# SPDX-License-Identifier: Apache-2.0
# Part of the Intel port: note 168 item 1 (run 165) -- sycl-tla's flash-attention example (head 128) and the head-256
# variant (probe2/fmha256.cpp), built by m1/build.sh fmha, run on a card inside the runtime image under the step
# rules at this model's shape (one rank: 12 q heads over 1 kv head, head 256, causal, 4,096 rows = a prefill chunk):
# the toolchain check and the real speed of the register-resident FMHA. Stage m3/STEP on intel-host (the binaries +
# fmha.sh), cool to <= 43 C, precheck, the guard at --hot 80, m3-step.sh with both cards, the kernel log, fetch.
#   bash logs/fmha1.sh STEP IMAGE_ID [rows=4096] [iters=20]
set -u
STEP=$1; IMG=$2; L=${3:-4096}; IT=${4:-20}
W=$HOME/code/radiance-intel
D=/mnt/fast-storage/models/test/radiance-intel
ssh intel-host mkdir -p "$D/m3/$STEP" || exit 1
ssh intel-host tee "$D/m3/$STEP/fmha128" < "$W/logs/fmha/fmha128" > /dev/null || exit 1
ssh intel-host tee "$D/m3/$STEP/fmha256" < "$W/logs/fmha/fmha256" > /dev/null || exit 1
ssh intel-host chmod +x "$D/m3/$STEP/fmha128" "$D/m3/$STEP/fmha256" || exit 1
A="--batch=1 --num_heads_q=12 --num_heads_kv=1 --seq_len_qo=$L --seq_len_kv=$L --is_causal --iterations=$IT --warmup=2 --verify=1"
printf '#!/bin/bash\nldd /out/fmha256 | grep -E "not found|sycl|ze_loader"\necho "== head 128"\ntimeout 600 /out/fmha128 %s --head_size_vo=128\necho "== head 256"\ntimeout 600 /out/fmha256 %s --head_size_vo=256\necho "== head 256, non-causal"\ntimeout 600 /out/fmha256 --batch=1 --num_heads_q=12 --num_heads_kv=1 --seq_len_qo=%s --seq_len_kv=%s --iterations=%s --warmup=2 --verify=1 --head_size_vo=256\n' "$A" "$A" "$L" "$L" "$IT" > "$W/logs/fmha/fmha.sh"
ssh intel-host tee "$D/m3/$STEP/fmha.sh" < "$W/logs/fmha/fmha.sh" > /dev/null || exit 1
timeout 1900 bash "$W/m3/cool43.sh" 1800 || exit 1
timeout 300 "$W/intel-host-engine-guard.sh" --precheck 2 || exit 1
deadline=$(( $(date +%s) + 900 ))
"$W/intel-host-engine-guard.sh" --hot 80 "ri-m3-$STEP" 300 > "$W/logs/guard-$STEP.txt" 2>&1 &
guard=$!
timeout 800 ssh intel-host env STEP_CARD=both STEP_CPUS=2 STEP_NOTE7_CPUS2=1 STEP_DEADLINE=$deadline \
    bash "$D/m3-step.sh" "$STEP" 700 "$IMG" bash /out/fmha.sh > "$W/logs/step-$STEP.txt" 2>&1
grep 'rc=' "$W/logs/step-$STEP.txt" | tail -1
wait $guard
tail -1 "$W/logs/guard-$STEP.txt"
sleep 20
ssh intel-host bash "$D/ri-klog2.sh" "$STEP"
ssh intel-host cat "$D/out-$STEP.txt" > "$W/results-m3/card01/$STEP-out.txt"
grep -E "==|Disposition|Performance|Batch:|rror|rc=" "$W/results-m3/card01/$STEP-out.txt" | cut -c1-200
