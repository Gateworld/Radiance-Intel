#!/bin/bash
# SPDX-License-Identifier: Apache-2.0
# Part of the Intel port: note 168 item 2 (run 164) -- the bit-exact matrix-unit form checked on the cards: B68 the
# form with the 7K and its logprob form, B69 the form under the launch profiler; the logprob answer compared with
# B65's (the run form on image 186, unchanged since).
#   bash logs/dpas_steps2.sh IMAGE_ID
set -u
IMG=$1
W=$HOME/code/radiance-intel
D=/mnt/fast-storage/models/test/radiance-intel
timeout 2700 bash "$W/logs/bstep.sh" B68 "$IMG" nomtpcappfmjm64trdpas B58 "7k 7klp" > "$W/logs/bstep-B68.txt" 2>&1
timeout 2700 bash "$W/logs/bstep.sh" B69 "$IMG" nomtpcappfmjm64trdpasprof B58 "7k" > "$W/logs/bstep-B69.txt" 2>&1
grep -h -E "prefill-7k|klog2|rc=" "$W/logs/bstep-B68.txt" "$W/logs/bstep-B69.txt" | cut -c1-120
mkdir -p "$W/results-m3/card01/B68"
ssh intel-host cat "$D/m3/B68/resp/prefill-7klp-1.http" > "$W/results-m3/card01/B68/prefill-7klp-1.http"
python3 -I "$W/logs/lp_cmp.py" "$W/results-m3/card01/B65/prefill-7klp-1.http" "$W/results-m3/card01/B68/prefill-7klp-1.http"
grep -A6 "xe launch profile (exit)" "$W/results-m3/card01/B69/serve.log" | cut -c1-150
