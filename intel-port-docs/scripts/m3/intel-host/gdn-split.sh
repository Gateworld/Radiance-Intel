#!/bin/bash
# SPDX-License-Identifier: Apache-2.0
# Part of the Intel port of radiance (milestone 3, run 112, owner note 138). Inside the step's container (one card):
# xe_gdn_split_bench from the step folder against the step folder's libxe.so, with RADIANCE_XE_GDN_SPLIT unset, 2 and 4.
# Every output hash must be equal across the three; the times a call are the microbench.
#   bash /out/gdn-split.sh [launches=200] [SPLITS=0,2,4] [SEQ_REPS=0]
# run 282 (owner note 224 item 2; logs/run281-gdn-stage1-plan.md): SPLITS names the RADIANCE_XE_GDN_SPLIT values to run,
# COMMA-separated (a quoted space list does not survive the ssh hop: KG14 ran SPLIT 0 alone with SEQ_REPS "2"); the new
# kernel's 8 / 16 beside 0 2 4. SEQ_REPS > 0 adds the `anchor-seq` schedule (one state, six calls hashed after each,
# SEQ_REPS repetitions) at every SPLIT -- every per-call hash must equal SPLIT 0's.
L=/out/libxe.so
B=/out/xe_gdn_split_bench
chmod +x $B
SPLITS=$(echo "${2:-0,2,4}" | tr ',' ' ')
SEQ=${3:-0}
echo "=== gdn-split: launches ${1:-200}, SPLITS [$SPLITS], SEQ_REPS $SEQ"
# run 113 (note 138's rest): both state forms, per_candidate (MTP off) and anchor (MTP on)
for form in per_candidate anchor; do
for v in $SPLITS; do
for T in 1 4; do   # run 283: one T a process, so the launch profiler's exit summary is that T's (the gate is at T 4)
  echo "=== $form RADIANCE_XE_GDN_SPLIT=$v T=$T ($(date -u +%H:%M:%S))"
  # RADIANCE_XE_GDN_NT=0 as the m3 configs serve (gnt0): the non-temporal state store holds the chain
  RI_GDN_T=$T RADIANCE_XE_GDN_NT=0 RADIANCE_XE_GDN_SPLIT=$v RADIANCE_XE_LAUNCH_PROFILE=1 RADIANCE_L0_QUEUE_PROFILING=1 timeout 120 $B $L ${1:-200} $form 2>&1 \
    | grep -E "RESULT|refused|rror|recurrent_update|ru_sg|split_norm|launch profile" | cut -c1-170
done
done
done
if [ "$SEQ" -gt 0 ] 2>/dev/null; then
for v in $SPLITS; do
  echo "=== anchor-seq x$SEQ RADIANCE_XE_GDN_SPLIT=$v ($(date -u +%H:%M:%S))"
  RADIANCE_XE_GDN_NT=0 RADIANCE_XE_GDN_SPLIT=$v timeout 300 $B $L 1 anchor-seq "$SEQ" 2>&1 \
    | grep -E "RESULT|refused|rror" | cut -c1-200 > /out/seq-$v.txt
  grep -c "RESULT anchor-seq" /out/seq-$v.txt
  tail -2 /out/seq-$v.txt
done
# the gate: every per-call hash line equal to SPLIT 0's (the rep / call / T / nacc prefix included)
for v in $SPLITS; do
  [ "$v" = 0 ] && continue
  if cmp -s <(grep "RESULT anchor-seq" /out/seq-0.txt) <(grep "RESULT anchor-seq" /out/seq-$v.txt); then
    echo "RESULT anchor-seq SPLIT=$v: every hash = SPLIT 0 ($(grep -c 'RESULT anchor-seq' /out/seq-$v.txt) calls)"
  else
    echo "RESULT anchor-seq SPLIT=$v: HASHES DIFFER from SPLIT 0 -- first difference:"
    diff <(grep "RESULT anchor-seq" /out/seq-0.txt) <(grep "RESULT anchor-seq" /out/seq-$v.txt) | head -4 | cut -c1-200
  fi
done
fi
