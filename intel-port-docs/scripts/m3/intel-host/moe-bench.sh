#!/bin/bash
# SPDX-License-Identifier: Apache-2.0
# Part of the Intel port of radiance (milestone 3, run 100, owner note 104). The planes expert GEMV's own device time at
# Flash-Next's decode shapes (gate_up and down, 10 routed experts a launch), then its element-wise check, on one card,
# with the image's libxe.so. Extra environment (a kernel form's flag) is passed as arguments: VAR=value ...
#   m3-step.sh STEP 900 <image> bash /out/moe-bench.sh [VAR=value ...]
set -u
cp /out/xe_moe_planes_bench /tmp/b && chmod +x /tmp/b || { echo "moe-bench: no bench binary in /out"; exit 2; }
L=/opt/ri/radiance_home/kernels/libxe.so
rc=0
echo "=== timing, one rank's shapes (BENCH_TP2=1) $* ($(date -u +%H:%M:%S))"
env BENCH_TP2=1 "$@" timeout 600 /tmp/b $L 512 48 5 || rc=1
echo "=== timing, the whole model's shapes $* ($(date -u +%H:%M:%S))"
env "$@" timeout 600 /tmp/b $L 512 48 5 || rc=1
for v in 1 2; do
  echo "=== CHECK=$v $* ($(date -u +%H:%M:%S))"
  env CHECK=$v "$@" timeout 600 /tmp/b $L || rc=1
done
echo "moe-bench: done rc=$rc ($(date -u +%H:%M:%S))"
exit $rc
