#!/bin/bash
# SPDX-License-Identifier: Apache-2.0
# Part of the Intel port of radiance (milestone 3, run 101, owner note 108 item 1). Inside the step's container (one
# card, STEP_ENGINE=1 for /d): the decode GDN update (gdn_conv_recurrent_update, the fused conv + recurrent update +
# gated norm + quantiser) against libref on the real container's weights, at the decode rows, under libxe's launch
# profiler (each case's launches timed on the queue); then rad-kbench's own timer on the same op, which segfaulted on
# hc_read in run 94 and is tried here last. Extra environment (a kernel form's flag) as arguments: VAR=value ...
#   bash /out/kbench-gdn.sh [VAR=value ...]
M=/d/qwen3.8-next-flash-fp8-iq4r-moe.rad
H=/opt/ri/radiance_home
echo "=== check + launch profile $* ($(date -u +%H:%M:%S))"
env RADIANCE_XE_LAUNCH_PROFILE=1 RADIANCE_L0_QUEUE_PROFILING=1 "$@" rad-kbench -m $M --home $H --kernels libxe:libref \
  --op gdn_conv_recurrent_update --max-ctx 4096 --max-tok 4 --report /out/report-gdn.md 2>&1 \
  | grep -E 'checked|launch profile|recurrent|conv_|rc=|rel_l2'
echo "rc=${PIPESTATUS[0]}"
echo "=== rad-kbench --bench --timing-only $* ($(date -u +%H:%M:%S))"
env "$@" timeout 300 rad-kbench -m $M --home $H --kernels libxe --op gdn_conv_recurrent_update --max-ctx 4096 \
  --max-tok 4 --bench --timing-only --bench-iters 200 2>&1 | grep -v -E '^I ' | tail -30
echo "rc=${PIPESTATUS[0]}"
