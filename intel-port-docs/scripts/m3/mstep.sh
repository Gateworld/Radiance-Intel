#!/bin/bash
# SPDX-License-Identifier: Apache-2.0
# Part of the Intel port of radiance (milestone 3, run 90). One M-pair half on both cards, the way runs 88-90 ran them
# by hand: stage m3/STEP on intel-host from a reference step's card files and today's m3-tps.sh, cool to the guard's
# threshold, precheck, start the heat guard (--hot 80, owner note 161), run `m3-tps.sh CFG 2` on IMAGE, check the kernel log 20 s
# after the step (a reset can be logged seconds after a kill), fetch out-STEP.txt and the two runs' SSE and
# serve.log to results-m3/card01/STEP/, and compare both runs' text with the reference step's.
#   bash m3/mstep.sh STEP IMAGE_ID CFG REF_STEP REF_CFG [MODEL_PATH]
# run 189: MODEL_PATH /a/<file> serves a model on intel-host's archive disk (m3-step.sh's STEP_ARCHIVE mount), e.g. the
# trained draft head's /a/flashnext-mtp1.rad; the MODEL environment form below stays for /d files.
set -u
STEP=$1; IMG=$2; CFG=$3; REF=$4; REFCFG=$5; MPATH=${6:-}
ARCH=()
case "$MPATH" in
  "") ;;
  /a/*.rad) ARCH=(STEP_ARCHIVE=1) ;;
  *) echo "mstep: MODEL_PATH must be /a/<file>.rad (got $MPATH)"; exit 2 ;;
esac
W=$HOME/code/radiance-intel
D=/mnt/fast-storage/models/test/radiance-intel
R=$W/results-m3/card01
ssh intel-host mkdir "$D/m3/$STEP" || exit 1
ssh intel-host cp -r "$D/m3/$REF/card" "$D/m3/$REF/flashnext-card1.args" "$D/m3/$STEP/" || exit 1
# run 172 (owner note 165 item 3): the 20K prompt body beside the card files, for a `-lanes-` config (B111's copy
# of the prefill steps' body; harmless for every other config)
ssh intel-host cp "$D/m3/${LANES_REF:-B111}/card/prefill-20k.body" "$D/m3/$STEP/card/" || true
scp -q "$W/m3/intel-host/m3-tps.sh" "intel-host:$D/m3/$STEP/m3-tps.sh" || exit 1
# run 116: MODEL=<file in D> serves another container (the step's own copy of the args names it)
if [ -n "${MODEL:-}" ]; then
  ssh intel-host sed -i "s#/d/qwen3.8-next-flash-fp8-iq4r-moe.rad#/d/$MODEL#" "$D/m3/$STEP/flashnext-card1.args" || exit 1
  ssh intel-host grep -c "/d/$MODEL" "$D/m3/$STEP/flashnext-card1.args" || exit 1
fi
if [ -n "$MPATH" ]; then
  ssh intel-host sed -i "s#/d/qwen3.8-next-flash-fp8-iq4r-moe.rad#$MPATH#" "$D/m3/$STEP/flashnext-card1.args" || exit 1
  ssh intel-host grep -c -F "$MPATH" "$D/m3/$STEP/flashnext-card1.args" || exit 1   # (one word: no space for ssh to split)
fi
# run 114: cool43.sh, not the guard's --cool (which returned at 47 C before M187): loads start at <= 43 C (note 121)
timeout 1900 bash "$W/m3/cool43.sh" 1800 || exit 1
timeout 300 "$W/intel-host-engine-guard.sh" --precheck 40 || exit 1
deadline=$(( $(date +%s) + 1800 ))
"$W/intel-host-engine-guard.sh" --hot 80 "ri-m3-$STEP" 1000 > "$W/logs/guard-$STEP.txt" 2>&1 &
guard=$!
# run 169 (owner notes 166 / 174): an ...avx config runs under the io_uring seccomp profile (podman's default plus the
# three io_uring syscalls, owner note 45; staged at $D/seccomp-iouring.json), which libavx's n-gram gather needs
# owner note 175 (00:27, 11-04): libavx is the step default, so every engine step runs under the profile
SECC="STEP_SECCOMP=$D/seccomp-iouring.json"
# run 172 (owner note 161 allows 2 or 4 CPUs a step): a `-cpu4-` config runs the container at 4 CPUs, for the
# lanes steps' question whether a prompt read beside decoders is starved on the host side
case "$CFG-" in *-cpu4-*) CPUS=4 ;; *) CPUS=2 ;; esac
timeout 1100 ssh intel-host env STEP_ENGINE=1 STEP_CARD=both STEP_CPUS=$CPUS STEP_NOTE7_CPUS2=1 STEP_DEADLINE=$deadline $SECC "${ARCH[@]}" \
    bash "$D/m3-step.sh" "$STEP" 900 "$IMG" bash /out/m3-tps.sh "$CFG" 2 > "$W/logs/step-$STEP.txt" 2>&1
grep 'rc=' "$W/logs/step-$STEP.txt" | tail -1
wait $guard
tail -1 "$W/logs/guard-$STEP.txt"
sleep 20
ssh intel-host bash "$D/ri-klog2.sh" "$STEP"
mkdir -p "$R/$STEP/$CFG"
scp -q "intel-host:$D/out-$STEP.txt" "$R/$STEP-out.txt"
scp -q "intel-host:$D/m3/$STEP/$CFG/tps-1.sse" "intel-host:$D/m3/$STEP/$CFG/tps-2.sse" "intel-host:$D/m3/$STEP/$CFG/serve.log" "$R/$STEP/$CFG/"
# run 172: a `-lanes-` config writes lanes-A/ and lanes-B/ (the decoders' SSE, the prompt's .http and .time) instead
case "$CFG-" in *-lanes-*) ssh intel-host tar -C "$D/m3/$STEP/$CFG" -cf - lanes-A lanes-B serve.log | tar -C "$R/$STEP/$CFG" -xf - ;;
                *-conc-*) ssh intel-host tar -C "$D/m3/$STEP/$CFG" -cf - conc-1 conc-3 conc-6 serve.log | tar -C "$R/$STEP/$CFG" -xf -
                          grep -E "conc [136]:|conc warm" "$R/$STEP-out.txt" | cut -c1-220 ;; esac
# run 174 (owner note 165 item 4): a `-rdump-` config writes route/route-*.bin (the router's rows, m3/route_union.py)
case "$CFG-" in *-rdump-*) ssh intel-host tar -C "$D/m3/$STEP/$CFG" -cf - route | tar -C "$R/$STEP/$CFG" -xf - ;; esac
grep -E 'up=|run [12]:' "$R/$STEP-out.txt"
grep 'step decode' "$R/$STEP-out.txt" | tail -1
grep 'driver takes' "$R/$STEP/$CFG/serve.log" | cut -c1-160
for k in 1 2; do
  python3 -I "$W/m3/tools/sse-text.py" "$R/$STEP/$CFG/tps-$k.sse" "$R/$REF/$REFCFG/tps-$k.sse" | tail -1
done
