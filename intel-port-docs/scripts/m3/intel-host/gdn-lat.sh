#!/bin/bash
# SPDX-License-Identifier: Apache-2.0
# Part of the Intel port of radiance (milestone 3, run 283; briefs/2026-11-11-gdn-split-latency.md, KG17). Inside the
# step's container (one card): xe_gdn_split_bench from the step folder against the step folder's libxe.so, the VS 2
# split (RADIANCE_XE_GDN_SPLIT=2) with the brief's EXACT load-chain changes RADIANCE_XE_GDN_LAT=0 (today's split),
# 1 (L1: the anchor slot selected from registers), 3 (L1 + L2: the replay rows loaded up front), 7 (+ N1: the norm
# folded into the head's last arriving slice, no norm launch) -- both state forms, one T a process (RI_GDN_T), the
# launch profiler's device times, 200 launches; the fused kernel (SPLIT 0) as the reference. The gate: every hash
# equal to SPLIT 0's -- the per-call hashes here and the anchor-seq schedule (SEQ_REPS repetitions) at every LAT.
# The coordinator's addition (03:2x 11-11): N1 is a cross-workgroup hand-off (fence -> counter -> the last arriver reads
# the other slice's rows), so a race is probabilistic and 20 reps can pass by luck: STRESS reps of anchor-seq at LAT 7
# (and SPLIT 0 as its reference), every per-call hash equal and the count of compared calls printed; N1 is adopted only
# with that line clean.
#   bash /out/gdn-lat.sh [launches=200] [LATS=0,1,3,7] [SEQ_REPS=20] [STRESS=500]
L=/out/libxe.so
B=/out/xe_gdn_split_bench
chmod +x $B
LN=${1:-200}
LATS=$(echo "${2:-0,1,3,7}" | tr ',' ' ')
SEQ=${3:-20}
STRESS=${4:-500}
echo "=== gdn-lat: launches $LN, LATS [$LATS], SEQ_REPS $SEQ, STRESS $STRESS"
run() {   # run NAME FORM T SPLIT LAT
  local name=$1 form=$2 T=$3 v=$4 lat=$5 f=/out/lat-$2-T$3-$1.txt
  echo "=== $form $name RADIANCE_XE_GDN_SPLIT=$v RADIANCE_XE_GDN_LAT=$lat T=$T ($(date -u +%H:%M:%S))"
  RI_GDN_T=$T RADIANCE_XE_GDN_NT=0 RADIANCE_XE_GDN_SPLIT=$v RADIANCE_XE_GDN_LAT=$lat RADIANCE_XE_LAUNCH_PROFILE=1 \
    RADIANCE_L0_QUEUE_PROFILING=1 timeout 120 $B $L $LN $form > $f 2>&1
  grep -E "RESULT|refused|rror|recurrent_update|ru_sg|split_norm|launch profile" $f | cut -c1-170
}
for form in per_candidate anchor; do
for T in 1 4; do
  run fused $form $T 0 0
  for lat in $LATS; do run L$lat $form $T 2 $lat; done
  ref=$(grep "RESULT gdn_split T $T hashes" /out/lat-$form-T$T-fused.txt)
  for lat in $LATS; do
    h=$(grep "RESULT gdn_split T $T hashes" /out/lat-$form-T$T-L$lat.txt)
    if [ -n "$h" ] && [ "$h" = "$ref" ]; then echo "RESULT hashes $form T $T LAT $lat = SPLIT 0"
    else echo "RESULT hashes $form T $T LAT $lat DIFFER from SPLIT 0"; fi
  done
done
done
if [ "$SEQ" -gt 0 ] 2>/dev/null; then
  echo "=== anchor-seq x$SEQ SPLIT 0 ($(date -u +%H:%M:%S))"
  RADIANCE_XE_GDN_NT=0 RADIANCE_XE_GDN_SPLIT=0 timeout 300 $B $L 1 anchor-seq "$SEQ" 2>&1 \
    | grep -E "RESULT|refused|rror" | cut -c1-200 > /out/seq-lat-0.txt
  grep -c "RESULT anchor-seq" /out/seq-lat-0.txt
  for lat in $LATS; do
    echo "=== anchor-seq x$SEQ SPLIT 2 LAT $lat ($(date -u +%H:%M:%S))"
    RADIANCE_XE_GDN_NT=0 RADIANCE_XE_GDN_SPLIT=2 RADIANCE_XE_GDN_LAT=$lat timeout 300 $B $L 1 anchor-seq "$SEQ" 2>&1 \
      | grep -E "RESULT|refused|rror" | cut -c1-200 > /out/seq-lat-$lat.txt
    if cmp -s <(grep "RESULT anchor-seq" /out/seq-lat-0.txt) <(grep "RESULT anchor-seq" /out/seq-lat-$lat.txt); then
      echo "RESULT anchor-seq LAT $lat: every hash = SPLIT 0 ($(grep -c 'RESULT anchor-seq' /out/seq-lat-$lat.txt) calls)"
    else
      echo "RESULT anchor-seq LAT $lat: HASHES DIFFER from SPLIT 0 -- first difference:"
      diff <(grep "RESULT anchor-seq" /out/seq-lat-0.txt) <(grep "RESULT anchor-seq" /out/seq-lat-$lat.txt) | head -4 | cut -c1-200
    fi
  done
fi
# the stress LAT: 7 when listed (the cross-workgroup hand-off), else the last LAT of the list (run 284: brief C's LAT 8)
SL=""; echo " $LATS " | grep -q " 7 " && SL=7; [ -z "$SL" ] && SL=${LATS##* }
if [ "$STRESS" -gt 0 ] 2>/dev/null && [ "$SL" != 0 ]; then
  echo "=== STRESS anchor-seq x$STRESS SPLIT 0 ($(date -u +%H:%M:%S))"
  RADIANCE_XE_GDN_NT=0 RADIANCE_XE_GDN_SPLIT=0 timeout 150 $B $L 1 anchor-seq "$STRESS" 2>&1 \
    | grep -E "RESULT|refused|rror" | cut -c1-200 > /out/stress-0.txt
  echo "=== STRESS anchor-seq x$STRESS SPLIT 2 LAT $SL ($(date -u +%H:%M:%S))"
  RADIANCE_XE_GDN_NT=0 RADIANCE_XE_GDN_SPLIT=2 RADIANCE_XE_GDN_LAT=$SL timeout 150 $B $L 1 anchor-seq "$STRESS" 2>&1 \
    | grep -E "RESULT|refused|rror" | cut -c1-200 > /out/stress-$SL.txt
  n0=$(grep -c "RESULT anchor-seq" /out/stress-0.txt); n7=$(grep -c "RESULT anchor-seq" /out/stress-$SL.txt)
  if [ "$n0" -gt 0 ] && cmp -s <(grep "RESULT anchor-seq" /out/stress-0.txt) <(grep "RESULT anchor-seq" /out/stress-$SL.txt); then
    echo "RESULT STRESS LAT $SL: every per-call hash = SPLIT 0 over $n7 of $n0 calls ($(date -u +%H:%M:%S))"
  else
    echo "RESULT STRESS LAT $SL: HASHES DIFFER or the counts differ ($n7 vs $n0 calls) -- first difference:"
    diff <(grep "RESULT anchor-seq" /out/stress-0.txt) <(grep "RESULT anchor-seq" /out/stress-$SL.txt) | head -4 | cut -c1-200
  fi
fi
