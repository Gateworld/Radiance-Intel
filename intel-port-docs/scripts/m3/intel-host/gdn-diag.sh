#!/bin/bash
# SPDX-License-Identifier: Apache-2.0
# Part of the Intel port of radiance (milestone 3, run 284: the coordinator's diagnostic sweep of the GDN slice-group
# kernel, stage 1 of owner note 224 item 2). Inside the step's container (one card): xe_gdn_split_bench from the step
# folder against the step folder's libxe.so -- a BENCH BUILD configured with XE_GDN_SG_DIAG=ON -- in the anchor form
# (nacc 2), one T a process (RI_GDN_T), with the launch profiler's device times a kernel. The references SPLIT 0 (the
# fused kernel) and SPLIT 2 (the VS 2 split), then SPLIT 8's V0 (today's slice-group kernel) and V1-V6
# (RADIANCE_XE_GDN_SG_DIAG=1..6, TIMING ONLY: 1 no last-arriver tail, 2 no skh / sgb parks, 3 no sbad barrier, 4 no
# per-token global stores, 5 q / k from global x without the q | k conv, 6 GC 1 = one value head a workgroup). The
# per-call hashes are compared with SPLIT 0's: V0, V3 and V6 should match; V1, V2, V4 and V5 differ by construction.
#   bash /out/gdn-diag.sh [launches=200] [T list=1,4]
L=/out/libxe.so
B=/out/xe_gdn_split_bench
chmod +x $B
LN=${1:-200}
TS=$(echo "${2:-1,4}" | tr ',' ' ')
echo "=== gdn-diag: launches $LN, T [$TS], anchor form (nacc 2)"
run() {   # run NAME T SPLIT DIAG
  local name=$1 T=$2 v=$3 d=$4 f=/out/diag-T$2-$1.txt
  echo "=== anchor $name RADIANCE_XE_GDN_SPLIT=$v RADIANCE_XE_GDN_SG_DIAG=$d T=$T ($(date -u +%H:%M:%S))"
  # RADIANCE_XE_GDN_NT=0 as the m3 configs serve (gnt0)
  RI_GDN_T=$T RADIANCE_XE_GDN_NT=0 RADIANCE_XE_GDN_SPLIT=$v RADIANCE_XE_GDN_SG_DIAG=$d RADIANCE_XE_LAUNCH_PROFILE=1 \
    RADIANCE_L0_QUEUE_PROFILING=1 timeout 120 $B $L $LN anchor > $f 2>&1
  grep -E "RESULT|refused|rror|recurrent_update|ru_sg|split_norm|launch profile" $f | cut -c1-170
}
for T in $TS; do
  run fused $T 0 0
  run vs2 $T 2 0
  for d in 0 1 2 3 4 5 6; do run V$d $T 8 $d; done
  # the hashes against SPLIT 0's (the fused kernel's)
  ref=$(grep "RESULT gdn_split T $T hashes" /out/diag-T$T-fused.txt)
  for name in vs2 V0 V1 V2 V3 V4 V5 V6; do
    h=$(grep "RESULT gdn_split T $T hashes" /out/diag-T$T-$name.txt)
    if [ -n "$h" ] && [ "$h" = "$ref" ]; then echo "RESULT hashes T $T $name = SPLIT 0"
    else echo "RESULT hashes T $T $name DIFFER from SPLIT 0"; fi
  done
done
