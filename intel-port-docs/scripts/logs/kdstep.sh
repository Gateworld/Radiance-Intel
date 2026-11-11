#!/bin/bash
# SPDX-License-Identifier: Apache-2.0
# Part of the Intel port: a KL replay step (m3/intel-host/kld-gl.sh MODE) under the step rules, staged from a reference
# KL step's flashnext-card1.args + gl-replay.jsonl plus today's kld-gl.sh; cool to <= 43 C, precheck, the guard at
# --hot 80, m3-step.sh on both cards, the kernel log 20 s after, the json and the output fetched.
#   bash logs/kdstep.sh STEP IMAGE_ID MODE REF_STEP [CORPUS.jsonl]      e.g. bash logs/kdstep.sh KDP2 a4f772a29e65 dpas KDP1
# run 170: CORPUS, a local JSONL, replaces the reference step's gl-replay.jsonl (the agreement corpus's prompts are
# all under 2,051 tokens, below the sparse attention's bound, so a prefill-attention form needs long documents)
set -u
STEP=$1; IMG=$2; MODE=$3; REF=$4; CORPUS=${5:-}
# run 179: a 6th argument sets the container's limit in seconds (the long MTP dump); KD_TL in the environment as before
[ -n "${6:-}" ] && KD_TL=$6
# run 187: a 7th argument names a seccomp profile on intel-host for the container (m3-step.sh's STEP_SECCOMP; the io_uring
# profile lets libavx's n-gram gather run as it does in the service)
SECC=${7:-}
W=$HOME/code/radiance-intel
D=/mnt/fast-storage/models/test/radiance-intel
R=$W/results-m3/card01
ssh intel-host mkdir -p "$D/m3/$STEP" || exit 1
ssh intel-host cp "$D/m3/$REF/flashnext-card1.args" "$D/m3/$REF/gl-replay.jsonl" "$D/m3/$STEP/" || exit 1
if [ -n "$CORPUS" ]; then ssh intel-host tee "$D/m3/$STEP/gl-replay.jsonl" < "$CORPUS" > /dev/null || exit 1; fi
ssh intel-host tee "$D/m3/$STEP/kld-gl.sh" < "$W/m3/intel-host/kld-gl.sh" > /dev/null || exit 1
timeout 1900 bash "$W/m3/cool43.sh" 1800 || exit 1
timeout 300 "$W/intel-host-engine-guard.sh" --precheck 40 || exit 1
deadline=$(( $(date +%s) + 1800 ))
"$W/intel-host-engine-guard.sh" --hot 80 "ri-m3-$STEP" 1000 > "$W/logs/guard-$STEP.txt" 2>&1 &
guard=$!
timeout $(( ${KD_TL:-1200} + 300 )) ssh intel-host env ${SECC:+STEP_SECCOMP=$SECC} STEP_ENGINE=1 STEP_CARD=both STEP_CPUS=2 STEP_NOTE7_CPUS2=1 STEP_DEADLINE=$deadline \
    bash "$D/m3-step.sh" "$STEP" "${KD_TL:-1200}" "$IMG" bash /out/kld-gl.sh "$MODE" > "$W/logs/step-$STEP.txt" 2>&1
grep 'rc=' "$W/logs/step-$STEP.txt" | tail -1
wait $guard
tail -1 "$W/logs/guard-$STEP.txt"
sleep 20
ssh intel-host bash "$D/ri-klog2.sh" "$STEP"
mkdir -p "$R/$STEP"
ssh intel-host cat "$D/out-$STEP.txt" > "$R/$STEP-out.txt"
ssh intel-host cat "$D/m3/$STEP/kld-$MODE.json" > "$R/$STEP/kld-$MODE.json"
grep -E "=== |rc=" "$R/$STEP-out.txt" | cut -c1-120
python3 -I -c "import json,sys; d=json.load(open(sys.argv[1]))['all']; k=d['kld']; print('positions %d: mean KL %.3g, median %.3g, p99 %.3g, max %.3g, top-1 %.1f %%, ppl ratio %.4f' % (d['positions'], k['mean'], k['median'], k['p99'], k['max'], 100*d['top1_agreement'], d['ppl']['ratio']))" "$R/$STEP/kld-$MODE.json"
