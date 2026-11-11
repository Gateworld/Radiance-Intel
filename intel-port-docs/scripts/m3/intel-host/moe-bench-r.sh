#!/bin/bash
# SPDX-License-Identifier: Apache-2.0
# Part of the Intel port of radiance (milestone 3, run 100, owner note 104 item 1). The planes expert GEMV's device time
# at one rank's decode shapes (BENCH_TP2=1: gate_up 512 x 2560, down 2560 x 256) in the shipped form and with the
# exact variants: 8-lane teams at K <= 256 (RADIANCE_XE_MOE_LPR8=1) and more rows a team for the small K alone
# (RADIANCE_XE_MOE_V2R_SMALLK). The CHECKSUM lines of equal-bits variants must match the shipped one's. Then the
# element-wise check at the variants. The bench binary comes from /out (amd-host's build).
#   m3-step.sh STEP 900 <image> bash /out/moe-bench-r.sh
set -u
cp /out/xe_moe_planes_bench /tmp/b && chmod +x /tmp/b || { echo "moe-bench-r: no bench binary in /out"; exit 2; }
L=/opt/ri/radiance_home/kernels/libxe.so
rc=0
run() {   # run LABEL VAR=value ...
  local lab=$1; shift
  echo "=== $lab: $* ($(date -u +%H:%M:%S))"
  env BENCH_TP2=1 "$@" timeout 300 /tmp/b $L 512 48 5 || rc=1
}
run shipped RADIANCE_XE_MOE_LPR8=0
run h2 RADIANCE_XE_MOE_H2=1
run lpr8r4 RADIANCE_XE_MOE_LPR8=1 RADIANCE_XE_MOE_V2R_SMALLK=4
run lpr8r4h2 RADIANCE_XE_MOE_LPR8=1 RADIANCE_XE_MOE_V2R_SMALLK=4 RADIANCE_XE_MOE_H2=1
run shipped2 RADIANCE_XE_MOE_LPR8=0
run h2b RADIANCE_XE_MOE_H2=1
for v in "RADIANCE_XE_MOE_H2=1" "RADIANCE_XE_MOE_LPR8=1 RADIANCE_XE_MOE_V2R_SMALLK=4 RADIANCE_XE_MOE_H2=1"; do
  echo "=== CHECK=1 BENCH_TP2=1 $v ($(date -u +%H:%M:%S))"
  env CHECK=1 BENCH_TP2=1 RADIANCE_XE_MOE_MX=0 $v timeout 300 /tmp/b $L || rc=1
done
echo "moe-bench-r: done rc=$rc ($(date -u +%H:%M:%S))"
exit $rc
