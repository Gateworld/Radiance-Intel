#!/bin/bash
# SPDX-License-Identifier: Apache-2.0
# Part of the Intel port of radiance (milestone 3, run 102, owner note 111). The lane layer's cross-lane butterfly on
# the card: libxe's emulated DPP / permlanex16 (select_from_group over a computed lane) against SYCL's
# permute_group_by_xor, the same 32-lane sum, timed (xe_lane_xor_bench copied into the step folder).
#   m3-step.sh STEP 600 <image> bash /out/lane-bench.sh [iters] [groups] [runs]
set -u
cp /out/xe_lane_xor_bench /tmp/b && chmod +x /tmp/b || { echo "lane-bench: no bench binary in /out"; exit 2; }
timeout 300 /tmp/b "${1:-4096}" "${2:-4096}" "${3:-5}"
echo "rc=$?"
