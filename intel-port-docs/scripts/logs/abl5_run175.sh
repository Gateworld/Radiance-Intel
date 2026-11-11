#!/bin/bash
# SPDX-License-Identifier: Apache-2.0
# Part of the Intel port (run 175, owner note 167 item 1): five profiled decode steps on one image -- the control and
# the format C run kernel's four ablations (RADIANCE_XE_I4I8_RUN_ABL=1-4, wrong numbers by design) -- for the expert
# kernels' microseconds a launch.
#   bash logs/abl5_run175.sh IMAGE_ID
set -u
IMG=$1
W=$HOME/code/radiance-intel
cd "$W" || exit 1
run() {
  timeout 1500 bash m3/mstep.sh "$1" "$IMG" "svcmtpx-gnt0-hc8-rp-gnx-$2" M262 svcmtpx-gnt0-hc8-rp-gnx > "logs/mstep-$1.txt" 2>&1
  echo "$1 rc=$?"
  grep -E "klog2" "logs/mstep-$1.txt" | cut -c1-140
  python3 -I logs/prof_step.py "results-m3/card01/$1/svcmtpx-gnt0-hc8-rp-gnx-$2/serve.log" logits_i8 1 moe_i4i8 | cut -c1-120
}
run P91 prof
run P92 abl1-prof
run P93 abl2-prof
run P94 abl3-prof
run P95 abl4-prof
date -u
