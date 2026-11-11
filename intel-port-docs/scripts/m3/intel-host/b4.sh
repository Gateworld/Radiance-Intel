#!/bin/bash
# SPDX-License-Identifier: Apache-2.0
# Part of the Intel port of radiance (milestone 3, run 50, stage B item B4). The planes bench's element-wise
# check of a prefill-sized routing on the card, four ways in one container: CHECK=1 (equal classes) and 2
# (the odd class narrower), each with the matrix-unit form and with v2m (RADIANCE_XE_MOE_MX=0). The bench
# binary comes from /out (amd-host's build of the same sources as the image's libxe.so), copied to /tmp
# because the mount may be noexec.
#   m3-step.sh B4 1200 <image> bash /out/b4.sh
set -u
cp /out/xe_moe_planes_bench /tmp/b && chmod +x /tmp/b || { echo "b4: no bench binary in /out"; exit 2; }
L=/opt/ri/radiance_home/kernels/libxe.so
ls -la $L
rc=0
for v in 1 2; do
  for mx in 1 0; do
    echo "=== CHECK=$v RADIANCE_XE_MOE_MX=$mx ($(date -u +%H:%M:%S))"
    env CHECK=$v RADIANCE_XE_MOE_MX=$mx timeout 600 /tmp/b $L || rc=1
  done
done
echo "b4: done rc=$rc ($(date -u +%H:%M:%S))"
exit $rc
