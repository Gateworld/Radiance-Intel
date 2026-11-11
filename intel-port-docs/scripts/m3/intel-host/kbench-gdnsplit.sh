#!/bin/bash
# SPDX-License-Identifier: Apache-2.0
# Part of the Intel port of radiance (milestone 3, run 112, owner note 138). Inside the step's container (one card,
# STEP_ENGINE=1 for /d): the decode GDN fold (gdn_conv_recurrent_update) against libref on the real container's
# weights at the decode and verify rows (1-4 tokens), with RADIANCE_XE_GDN_SPLIT unset / 2 / 4, under libxe's launch
# profiler (each case's launches timed). The split is meant to give the fused fold's bytes, so every report must be the
# same file (md5). The step folder's libxe.so.
#   bash /out/kbench-gdnsplit.sh
M=/d/qwen3.8-next-flash-fp8-iq4r-moe.rad
H=/opt/ri/radiance_home
[ -f /out/libxe.so ] && cp /out/libxe.so $H/kernels/libxe.so && echo "libxe.so from /out"
for v in 0 2 4; do
  echo "=== RADIANCE_XE_GDN_SPLIT=$v ($(date -u +%H:%M:%S))"
  env RADIANCE_XE_GDN_SPLIT=$v RADIANCE_XE_LAUNCH_PROFILE=1 RADIANCE_L0_QUEUE_PROFILING=1 timeout 300 rad-kbench -m $M \
    --home $H --kernels libxe:libref --op gdn_conv_recurrent_update --max-ctx 4096 --max-tok 4 \
    --report /out/report-gdnsplit$v.md > /out/kb-gdnsplit$v.log 2>&1
  echo "rc=$? $(grep -m1 -o -E '[0-9]+ case\(s\) checked, [0-9]+ passed, [0-9]+ failed.*' /out/report-gdnsplit$v.md)"
  grep -E "us  ZN|launch profile" /out/kb-gdnsplit$v.log | grep -E "recurrent_update|split_norm|launch profile" | head -8 | cut -c1-150
done
md5sum /out/report-gdnsplit*.md
