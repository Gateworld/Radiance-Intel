#!/bin/bash
# SPDX-License-Identifier: Apache-2.0
# Part of the Intel port of radiance (milestone 3, run 110, owner note 129 item 4). Inside the step's container (one
# card): xe_router_bench from the step folder against the step folder's libxe.so (else the image's).
#   bash /out/router-bench.sh
# run 279 (owner note 216 item 3, the router redesign's bench first): the sort forms, each twice; the output hashes of
# every form at M 1 / 4 / 8 must equal the base's --
#   base       today's kernel (the small sort: three barriers, LDS atomics, emulated DPP scans)
#   scan=1     RB4a: the small sort with the sub-group's native inclusive scan (RADIANCE_XE_SORT_SCAN=1)
#   direct=2   RB4: the direct form with the slot list in registers, the loops unrolled over 64 (RADIANCE_XE_SORT_DIRECT=2,
#              T <= 64: the 8-row case falls back to the small sort)
#   direct=1   the run-110 direct form (T <= 16: the 1-row case only), for the record
L=/opt/ri/radiance_home/kernels/libxe.so
[ -f /out/libxe.so ] && L=/out/libxe.so && echo "libxe.so from /out"
chmod +x /out/xe_router_bench
run() {
  echo "=== $1 ($(date -u +%H:%M:%S))"
  env "${@:2}" timeout 120 /out/xe_router_bench "$L" 400 5 2>&1 | grep -E "^RESULT|rror|refused"
}
for pass in 1 2; do
  run "base pass $pass" RADIANCE_XE_SORT_DIRECT=0
  run "scan=1 pass $pass" RADIANCE_XE_SORT_SCAN=1
  run "direct=2 pass $pass" RADIANCE_XE_SORT_DIRECT=2
  run "direct=1 pass $pass" RADIANCE_XE_SORT_DIRECT=1
done
