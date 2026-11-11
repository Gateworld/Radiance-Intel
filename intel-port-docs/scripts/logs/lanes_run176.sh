#!/bin/bash
# SPDX-License-Identifier: Apache-2.0
# Part of the Intel port (run 176, owner note 176 item 4c): run 172's long-prompt lanes re-measured on today's
# defaults (the attention op, the bandwidth all-reduce, the record barrier, the warm-up pass): two decoders and a
# third user's 20K prompt (m3-tps.sh `-lanes-`), the scheduler's knobs as run 172 tried them; each step's decoders'
# worst gap and the prompt's time by logs/lanes_gaps.py.
#   bash logs/lanes_run176.sh IMAGE_ID
set -u
IMG=$1
W=$HOME/code/radiance-intel
cd "$W" || exit 1
run() {
  timeout 1500 bash m3/mstep.sh "$1" "$IMG" "svcmtpx-gnt0-hc8-rp-gnx-$2" M262 svcmtpx-gnt0-hc8-rp-gnx > "logs/mstep-$1.txt" 2>&1
  echo "$1 ($2) rc=$?"
  grep -E "klog2" "logs/mstep-$1.txt" | cut -c1-120
  python3 -I logs/lanes_gaps.py "results-m3/card01/$1/svcmtpx-gnt0-hc8-rp-gnx-$2" 2>&1 | tail -8 | cut -c1-200
}
run L15 lanes
run L16 lanes-pi8
run L17 lanes-bpc1k-pi8
run L18 lanes-bpc2k-pi8
date -u
