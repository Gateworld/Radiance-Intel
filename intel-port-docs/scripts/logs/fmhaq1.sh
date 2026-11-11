#!/bin/bash
# SPDX-License-Identifier: Apache-2.0
# Part of the Intel port: note 168 item 1 (run 165) -- sycl-tla's FMHA at head 256 with the per-query page selection
# (probe2/fmha_qsa, built by m1/build.sh fmhaqsa) on a card inside the runtime image under the step rules, at the
# 7K's two chunks of one rank (12 q heads over 1 kv head): chunk 1 = 4,096 rows over 4,096 keys, chunk 2 = 1,508
# rows over 5,604 keys; each verified on sampled rows against a host reference over the selected keys, then the
# dense control (FMHA_QSA=0) for the mask's cost.
#   bash logs/fmhaq1.sh STEP IMAGE_ID [iters=20]
set -u
STEP=$1; IMG=$2; IT=${3:-20}
W=$HOME/code/radiance-intel
D=/mnt/fast-storage/models/test/radiance-intel
ssh intel-host mkdir -p "$D/m3/$STEP" || exit 1
ssh intel-host tee "$D/m3/$STEP/fmha_qsa" < "$W/logs/fmha/fmha_qsa" > /dev/null || exit 1
ssh intel-host chmod +x "$D/m3/$STEP/fmha_qsa" || exit 1
C="--batch=1 --num_heads_q=12 --num_heads_kv=1 --is_causal --warmup=2 --verify=1 --head_size_vo=256 --iterations=$IT"
printf '#!/bin/bash\necho "== chunk 1, selected"\ntimeout 600 /out/fmha_qsa %s --seq_len_qo=4096 --seq_len_kv=4096\necho "== chunk 2, selected"\ntimeout 600 /out/fmha_qsa %s --seq_len_qo=1508 --seq_len_kv=5604\necho "== chunk 1, dense control"\nFMHA_QSA=0 timeout 600 /out/fmha_qsa %s --seq_len_qo=4096 --seq_len_kv=4096\necho "== chunk 2, dense control"\nFMHA_QSA=0 timeout 600 /out/fmha_qsa %s --seq_len_qo=1508 --seq_len_kv=5604\n' "$C" "$C" "$C" "$C" > "$W/logs/fmha/fmhaq.sh"
ssh intel-host tee "$D/m3/$STEP/fmhaq.sh" < "$W/logs/fmha/fmhaq.sh" > /dev/null || exit 1
timeout 1900 bash "$W/m3/cool43.sh" 1800 || exit 1
timeout 300 "$W/intel-host-engine-guard.sh" --precheck 2 || exit 1
deadline=$(( $(date +%s) + 900 ))
"$W/intel-host-engine-guard.sh" --hot 80 "ri-m3-$STEP" 300 > "$W/logs/guard-$STEP.txt" 2>&1 &
guard=$!
timeout 800 ssh intel-host env STEP_CARD=both STEP_CPUS=2 STEP_NOTE7_CPUS2=1 STEP_DEADLINE=$deadline \
    bash "$D/m3-step.sh" "$STEP" 700 "$IMG" bash /out/fmhaq.sh > "$W/logs/step-$STEP.txt" 2>&1
grep 'rc=' "$W/logs/step-$STEP.txt" | tail -1
wait $guard
tail -1 "$W/logs/guard-$STEP.txt"
sleep 20
ssh intel-host bash "$D/ri-klog2.sh" "$STEP"
ssh intel-host cat "$D/out-$STEP.txt" > "$W/results-m3/card01/$STEP-out.txt"
grep -E "==|verify_sparse|Disposition|Performance|rror|rc=" "$W/results-m3/card01/$STEP-out.txt" | cut -c1-200
