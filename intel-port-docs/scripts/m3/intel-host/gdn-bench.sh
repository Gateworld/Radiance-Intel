#!/bin/bash
# SPDX-License-Identifier: Apache-2.0
# Part of the Intel port of radiance (milestone 3, run 101, owner note 108 item 1). The decode GDN update's device time
# at one rank's shape (gdn_decode_bench: the fused conv + recurrent form and the recurrent form alone), at T = 1 (MTP
# off) and T = 4 (the 4-row verify), with the image's libxe.so; with SCALE=1 also over the head and sequence counts
# (whether the time follows the work or a fixed chain). Extra environment (a kernel form's flag) as arguments.
#   m3-step.sh STEP 600 <image> bash /out/gdn-bench.sh [VAR=value ...]
set -u
cp /out/xe_gdn_decode_bench /tmp/b && chmod +x /tmp/b || { echo "gdn-bench: no bench binary in /out"; exit 2; }
L=/opt/ri/radiance_home/kernels/libxe.so
rc=0
for t in 1 4; do
  echo "=== BENCH_T=$t $* ($(date -u +%H:%M:%S))"
  env BENCH_T=$t "$@" timeout 300 /tmp/b $L 200 5 || rc=1
done
if [ "${SCALE:-0}" = 1 ]; then
  for c in "BENCH_H=8 BENCH_HG=4" "BENCH_H=32 BENCH_HG=16" "BENCH_N=2" "BENCH_N=4" "BENCH_T=2" "BENCH_T=8"; do
    echo "=== $c $* ($(date -u +%H:%M:%S))"
    env $c "$@" timeout 300 /tmp/b $L 200 5 || rc=1
  done
fi
# DIAG=1: the kernel's timing diagnostic (RADIANCE_XE_GDN_DIAG bits: 1 norm, 2 quantiser, 4 state store, 16 l2 row
# sums, 32 dot / output row sums), at T = 1 and 4 -- what each part of the chain costs (outputs then wrong)
if [ "${DIAG:-0}" = 1 ]; then
  for d in 1 2 3 4 16 32 48 7 55; do
    for t in 1 4; do
      echo "=== RADIANCE_XE_GDN_DIAG=$d BENCH_T=$t ($(date -u +%H:%M:%S))"
      env RADIANCE_XE_GDN_DIAG=$d BENCH_T=$t "$@" timeout 300 /tmp/b $L 200 5 | grep 'RESULT gdn_decode conv' || rc=1
    done
  done
fi
# NT=1: the state's non-temporal hint forced off and on (RADIANCE_XE_GDN_NT; no result changes: checksums compared)
if [ "${NT:-0}" = 1 ]; then
  for v in 0 1; do
    for t in 1 4; do
      echo "=== RADIANCE_XE_GDN_NT=$v BENCH_T=$t ($(date -u +%H:%M:%S))"
      env RADIANCE_XE_GDN_NT=$v BENCH_T=$t "$@" timeout 300 /tmp/b $L 200 5 | grep -E 'RESULT gdn_decode (conv|recurrent)|CHECKSUM' || rc=1
    done
  done
fi
echo "gdn-bench: done rc=$rc ($(date -u +%H:%M:%S))"
exit $rc
