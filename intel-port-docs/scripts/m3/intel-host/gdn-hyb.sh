#!/bin/bash
# SPDX-License-Identifier: Apache-2.0
# Part of the Intel port of radiance (milestone 3, run 284, owner note 230: the VS 4 HYBRID slice-group). Inside the step's
# container (one card): xe_gdn_split_bench from the step folder against the step folder's libxe.so -- the fused kernel
# (RADIANCE_XE_GDN_SPLIT=0), the VS 2 split + L1 (SPLIT=2, the default) and the hybrid (RADIANCE_XE_GDN_HYBRID=4) -- both
# state forms, one T a process (RI_GDN_T), the launch profiler's device times, LAUNCHES launches; every per-call hash
# compared with the fused kernel's; then the anchor-seq schedule x SEQ_REPS and x STRESS for the hybrid against the fused
# kernel (every per-call hash equal, the counts printed). The gates (note 230): hashes = SPLIT 0 everywhere; >= 10 %
# better than the split + L1 at T 4 -> the M-pair; zeinfo 0 spills (desk).
#   bash /out/gdn-hyb.sh [launches=5000] [SEQ_REPS=20] [STRESS=500]
L=/out/libxe.so
B=/out/xe_gdn_split_bench
chmod +x $B
LN=${1:-5000}; SEQ=${2:-20}; STRESS=${3:-500}
echo "=== gdn-hyb: launches $LN, SEQ_REPS $SEQ, STRESS $STRESS"
run() {   # run NAME FORM T SPLIT HYBRID
  local name=$1 form=$2 T=$3 v=$4 h=$5 f=/out/hyb-$2-T$3-$1.txt
  echo "=== $form $name RADIANCE_XE_GDN_SPLIT=$v RADIANCE_XE_GDN_HYBRID=$h T=$T ($(date -u +%H:%M:%S))"
  RI_GDN_T=$T RADIANCE_XE_GDN_NT=0 RADIANCE_XE_GDN_SPLIT=$v RADIANCE_XE_GDN_HYBRID=$h RADIANCE_XE_LAUNCH_PROFILE=1 \
    RADIANCE_L0_QUEUE_PROFILING=1 timeout 180 $B $L $LN $form > $f 2>&1
  grep -E "RESULT|refused|rror|recurrent_update|ru_sg|split_norm|launch profile" $f | cut -c1-170
}
for form in per_candidate anchor; do
for T in 1 4; do
  run fused $form $T 0 0
  run split $form $T 2 0
  run hyb $form $T 0 4
  ref=$(grep "RESULT gdn_split T $T hashes" /out/hyb-$form-T$T-fused.txt)
  for name in split hyb; do
    h=$(grep "RESULT gdn_split T $T hashes" /out/hyb-$form-T$T-$name.txt)
    if [ -n "$h" ] && [ "$h" = "$ref" ]; then echo "RESULT hashes $form T $T $name = SPLIT 0"
    else echo "RESULT hashes $form T $T $name DIFFER from SPLIT 0"; fi
  done
done
done
seq_run() {   # seq_run NAME REPS SPLIT HYBRID OUT
  RADIANCE_XE_GDN_NT=0 RADIANCE_XE_GDN_SPLIT=$3 RADIANCE_XE_GDN_HYBRID=$4 timeout 300 $B $L 1 anchor-seq "$2" 2>&1 \
    | grep -E "RESULT|refused|rror" | cut -c1-200 > $5
}
for reps in $SEQ $STRESS; do
  [ "$reps" -gt 0 ] 2>/dev/null || continue
  echo "=== anchor-seq x$reps fused vs hybrid ($(date -u +%H:%M:%S))"
  seq_run fused $reps 0 0 /out/hseq-0-$reps.txt
  seq_run hyb $reps 0 4 /out/hseq-h-$reps.txt
  n0=$(grep -c "RESULT anchor-seq" /out/hseq-0-$reps.txt); nh=$(grep -c "RESULT anchor-seq" /out/hseq-h-$reps.txt)
  if [ "$n0" -gt 0 ] && cmp -s <(grep "RESULT anchor-seq" /out/hseq-0-$reps.txt) <(grep "RESULT anchor-seq" /out/hseq-h-$reps.txt); then
    echo "RESULT anchor-seq x$reps hybrid: every per-call hash = SPLIT 0 over $nh of $n0 calls"
  else
    echo "RESULT anchor-seq x$reps hybrid: HASHES DIFFER or counts differ ($nh vs $n0) -- first difference:"
    diff <(grep "RESULT anchor-seq" /out/hseq-0-$reps.txt) <(grep "RESULT anchor-seq" /out/hseq-h-$reps.txt) | head -4 | cut -c1-200
  fi
done
