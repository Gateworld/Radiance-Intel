#!/bin/bash
# SPDX-License-Identifier: Apache-2.0
# Part of the Intel port: a hand-run rotation step under the step rules (owner note 161: start <= 43 C, the guard at
# --hot 80): the step dir is staged already (card/, flashnext-card1.args, m2-heat.sh); cool, precheck, guard,
# m3-step.sh with both cards, the kernel log 20 s after, fetch serve.log, print the rotation's medians.
#   bash logs/hstep.sh STEP IMAGE_ID CFG [minutes=10] [MODEL_PATH]
# run 189: MODEL_PATH /a/<file>.rad serves a model on intel-host's archive disk (m3-step.sh's STEP_ARCHIVE mount): the
# staged args' --model is pointed at it first.
set -u
STEP=$1; IMG=$2; CFG=$3; MIN=${4:-10}; MPATH=${5:-}
W=$HOME/code/radiance-intel
D=/mnt/fast-storage/models/test/radiance-intel
R=$W/results-m3/card01
ARCH=()
case "$MPATH" in
  "") ;;
  /a/*.rad)
    ARCH=(STEP_ARCHIVE=1)
    ssh intel-host sed -i "s#/d/qwen3.8-next-flash-fp8-iq4r-moe.rad#$MPATH#" "$D/m3/$STEP/flashnext-card1.args" || exit 1
    ssh intel-host grep -c -F "$MPATH" "$D/m3/$STEP/flashnext-card1.args" || exit 1 ;;
  *) echo "hstep: MODEL_PATH must be /a/<file>.rad (got $MPATH)"; exit 2 ;;
esac
# run 177: the container's limit from the replay's length -- the engine's ~2-minute start and the summary on top
# (1,200 s for the 10-minute form, as before; 20 minutes -> 1,800 s)
TL=$(( MIN * 60 + 600 )); [ "$TL" -lt 1200 ] && TL=1200
timeout 1900 bash "$W/m3/cool43.sh" 1800 || exit 1
timeout 300 "$W/intel-host-engine-guard.sh" --precheck 40 || exit 1
deadline=$(( $(date +%s) + 1800 ))
"$W/intel-host-engine-guard.sh" --hot 80 "ri-m3-$STEP" 1000 > "$W/logs/guard-$STEP.txt" 2>&1 &
guard=$!
# owner note 175 (run 169): libavx is the step default, so the engine runs under note 45's io_uring seccomp profile
SECC="STEP_SECCOMP=$D/seccomp-iouring.json"
timeout $(( TL + 300 )) ssh intel-host env STEP_ENGINE=1 STEP_CARD=both STEP_CPUS=2 STEP_NOTE7_CPUS2=1 STEP_DEADLINE=$deadline $SECC "${ARCH[@]}" \
    bash "$D/m3-step.sh" "$STEP" "$TL" "$IMG" bash /out/m2-heat.sh "$CFG" "$MIN" > "$W/logs/step-$STEP.txt" 2>&1
grep 'rc=' "$W/logs/step-$STEP.txt" | tail -1
wait $guard
tail -1 "$W/logs/guard-$STEP.txt"
sleep 20
ssh intel-host bash "$D/ri-klog2.sh" "$STEP"
mkdir -p "$R/$STEP"
ssh intel-host cat "$D/out-$STEP.txt" > "$R/$STEP-out.txt"
ssh intel-host cat "$D/m3/$STEP/serve.log" > "$R/$STEP/serve.log"
grep -E 'requests|median|heat stop|exit' "$R/$STEP-out.txt" | tail -4
python3 -I "$W/m3/heat_median.py" "$R/$STEP/serve.log"
