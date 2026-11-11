#!/bin/bash
# SPDX-License-Identifier: Apache-2.0
# Part of the Intel port (run 175, owner note 176 item 4b): three profiled decode steps on one image -- the control,
# the trunk W8A8 on the matrix units at 4 rows (RADIANCE_XE_GEMM_DPAS_ROWS=1) and hc on the matrix units from 2 rows
# (RADIANCE_XE_HC_PF_DPAS=2) -- for the device ms a rank-step and the kernels each form runs.
#   bash logs/prof3_run175.sh IMAGE_ID
set -u
IMG=$1
W=$HOME/code/radiance-intel
cd "$W" || exit 1
run() {
  timeout 1500 bash m3/mstep.sh "$1" "$IMG" "svcmtpx-gnt0-hc8-rp-gnx-$2" M262 svcmtpx-gnt0-hc8-rp-gnx > "logs/mstep-$1.txt" 2>&1
  echo "$1 rc=$?"
  grep -E "klog2|run [12]:" "logs/mstep-$1.txt" | cut -c1-140
}
run P88 prof
run P89 rows-prof
run P90 hcsr-prof
date -u
