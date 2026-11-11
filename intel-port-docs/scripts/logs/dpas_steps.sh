#!/bin/bash
# SPDX-License-Identifier: Apache-2.0
# Part of the Intel port: note 168 item 2 (run 164) -- the matrix-unit expert form at prefill checked on the cards:
# B65 the run form (today's) with the 7K and its logprob form, B66 the same with RADIANCE_XE_I4I8_DPAS=1, B67 the
# matrix-unit form under the launch profiler; then the two logprob answers compared bit for bit.
#   bash logs/dpas_steps.sh IMAGE_ID
set -u
IMG=$1
W=$HOME/code/radiance-intel
D=/mnt/fast-storage/models/test/radiance-intel
timeout 2700 bash "$W/logs/bstep.sh" B65 "$IMG" nomtpcappfmjm64tr B58 "7k 7klp" > "$W/logs/bstep-B65.txt" 2>&1
timeout 2700 bash "$W/logs/bstep.sh" B66 "$IMG" nomtpcappfmjm64trdpas B58 "7k 7klp" > "$W/logs/bstep-B66.txt" 2>&1
timeout 2700 bash "$W/logs/bstep.sh" B67 "$IMG" nomtpcappfmjm64trdpasprof B58 "7k" > "$W/logs/bstep-B67.txt" 2>&1
grep -h -E "prefill-7k|klog2|rc=" "$W/logs/bstep-B65.txt" "$W/logs/bstep-B66.txt" "$W/logs/bstep-B67.txt" | cut -c1-120
mkdir -p "$W/results-m3/card01/B65" "$W/results-m3/card01/B66"
ssh intel-host cat "$D/m3/B65/resp/prefill-7klp-1.http" > "$W/results-m3/card01/B65/prefill-7klp-1.http"
ssh intel-host cat "$D/m3/B66/resp/prefill-7klp-1.http" > "$W/results-m3/card01/B66/prefill-7klp-1.http"
python3 -I "$W/logs/lp_cmp.py" "$W/results-m3/card01/B65/prefill-7klp-1.http" "$W/results-m3/card01/B66/prefill-7klp-1.http"
grep -A6 "xe launch profile (exit)" "$W/results-m3/card01/B67/serve.log" | cut -c1-150
