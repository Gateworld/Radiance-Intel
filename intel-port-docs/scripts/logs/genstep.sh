#!/bin/bash
# SPDX-License-Identifier: Apache-2.0
# Part of the Intel port (run 284, owner note 228 item 1; logs/run283-mtp-v2-gen-design.md section 6): one MTP v2
# generation part under the step rules (kdstep.sh / mstep.sh's form): stage m3/STEP on intel-host with a reference engine
# step's flashnext-card1.args (the production-like config), the prompt set, the client and gen-serve.sh; cool to <= 43
# C; precheck 40 GiB; the guard at --hot 80 (it also stops under 8 GiB; the design's 20 GiB stop is the precheck's
# margin); m3-step.sh on both cards, --cpus 2, libavx's io_uring seccomp profile as the engine steps, the container's
# limit 3,000 s (<= 50 min: the client takes new documents for MAX_MINUTES); the kernel log 20 s after; docs-partN.jsonl,
# the client's log and the serve log's tail fetched to results-m3/card01/STEP/.
#   bash logs/genstep.sh STEP IMAGE_ID PART [IDS=-] [CONCURRENCY=3] [MAX_MINUTES=45] [REF=M285] [PROMPTS=logs/mtp-v2/prompts.jsonl] [FORM=svc]
# IDS: "lo-hi" selects the part's documents by id, "-" (default) takes every document of PROMPTS not yet done.
# FORM: gen-serve.sh's engine form -- svc (the staged args), hr4g (headroom 4096), notape, hr4gnotape (after GEN0's stall).
# RESUME (10th): a local docs-partN.jsonl from an earlier step of the same part, staged as /out/docs-partN.jsonl so the
# client skips its documents and appends the rest (the fetched file then holds the whole part).
set -u
STEP=$1; IMG=$2; PART=$3; IDS=${4:--}; CONC=${5:-3}; MAXMIN=${6:-45}; REF=${7:-M285}
W=$HOME/code/radiance-intel
PROMPTS=${8:-$W/logs/mtp-v2/prompts.jsonl}; FORM=${9:-svc}; RESUME=${10:-}
D=/mnt/fast-storage/models/test/radiance-intel
R=$W/results-m3/card01
ssh intel-host mkdir -p "$D/m3/$STEP" || exit 1
ssh intel-host cp "$D/m3/$REF/flashnext-card1.args" "$D/m3/$STEP/" || exit 1
ssh intel-host tee "$D/m3/$STEP/prompts.jsonl" < "$PROMPTS" > /dev/null || exit 1
ssh intel-host tee "$D/m3/$STEP/mtp_v2_gen.py" < "$W/m3/tools/mtp_v2_gen.py" > /dev/null || exit 1
ssh intel-host tee "$D/m3/$STEP/gen-serve.sh" < "$W/m3/intel-host/gen-serve.sh" > /dev/null || exit 1
if [ -n "$RESUME" ]; then
  ssh intel-host tee "$D/m3/$STEP/docs-part$PART.jsonl" < "$RESUME" > /dev/null || exit 1
  echo "resume: $(grep -c '"id"' "$RESUME") documents staged from $RESUME"
fi
timeout 1900 bash "$W/m3/cool43.sh" 1800 || exit 1
timeout 300 "$W/intel-host-engine-guard.sh" --precheck 40 || exit 1
deadline=$(( $(date +%s) + 3300 ))
"$W/intel-host-engine-guard.sh" --hot 80 "ri-m3-$STEP" 1000 > "$W/logs/guard-$STEP.txt" 2>&1 &
guard=$!
SECC="STEP_SECCOMP=$D/seccomp-iouring.json"
timeout 3400 ssh intel-host env STEP_ENGINE=1 STEP_CARD=both STEP_CPUS=2 STEP_NOTE7_CPUS2=1 STEP_DEADLINE=$deadline $SECC \
    bash "$D/m3-step.sh" "$STEP" 3000 "$IMG" bash /out/gen-serve.sh "$PART" "$IDS" "$CONC" "$MAXMIN" "$IMG" "$FORM" > "$W/logs/step-$STEP.txt" 2>&1
grep 'rc=' "$W/logs/step-$STEP.txt" | tail -1
wait $guard
tail -1 "$W/logs/guard-$STEP.txt"
sleep 20
ssh intel-host bash "$D/ri-klog2.sh" "$STEP"
mkdir -p "$R/$STEP"
ssh intel-host cat "$D/out-$STEP.txt" > "$R/$STEP-out.txt"
ssh intel-host cat "$D/m3/$STEP/docs-part$PART.jsonl" > "$R/$STEP/docs-part$PART.jsonl"
ssh intel-host cat "$D/m3/$STEP/gen-part$PART.log" > "$R/$STEP/gen-part$PART.log"
ssh intel-host tail -40 "$D/m3/$STEP/serve.log" > "$R/$STEP/serve-tail.log"
grep -E "^===|RESULT|up=|rc=|exit" "$R/$STEP-out.txt" | cut -c1-200
