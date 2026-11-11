#!/bin/bash
# SPDX-License-Identifier: Apache-2.0
# Part of the Intel port: note 168 item 1 (run 167) -- the attention port's first engine checks on image N: B97 the
# 7K / 20K / 7K-logprob prefills with attn_qsa_fmha on (`nomtpcappfmjm64trfmha`), then the KL replay of the form
# against the exact per-row path (kld-gl.sh fmha, staged from KDP1's args + corpus), then the 7K logprob answer
# against the exact path's (B66) for the shift.
#   bash logs/fmha_steps.sh IMAGE_ID
set -u
IMG=$1
W=$HOME/code/radiance-intel
D=/mnt/fast-storage/models/test/radiance-intel
#   bash logs/fmha_steps.sh IMAGE_ID [BSTEP=B98] [KSTEP=KFM1]
BS=${2:-B98}; KS=${3:-KFM1}
timeout 2700 bash "$W/logs/bstep.sh" "$BS" "$IMG" nomtpcappfmjm64trfmha B58 "7k 20k 7klp" > "$W/logs/bstep-$BS.txt" 2>&1
grep -h -E "prefill-|klog2|rc=" "$W/logs/bstep-$BS.txt" | cut -c1-120
grep -m3 -E "refused op|DEVICE_LOST|FAULT" "$W/results-m3/card01/$BS/serve.log" | cut -c1-160
grep -q "klog2 $BS: FAULT" "$W/logs/bstep-$BS.txt" && { echo "fault: the replay is not run"; exit 1; }
mkdir -p "$W/results-m3/card01/$BS"
ssh intel-host cat "$D/m3/$BS/resp/prefill-7klp-1.http" > "$W/results-m3/card01/$BS/prefill-7klp-1.http"
python3 -I "$W/logs/lp_cmp.py" "$W/results-m3/card01/B66/prefill-7klp-1.http" "$W/results-m3/card01/$BS/prefill-7klp-1.http" | tail -3
timeout 2400 bash "$W/logs/kdstep.sh" "$KS" "$IMG" fmha KDP1 > "$W/logs/kdstep-$KS.txt" 2>&1
tail -3 "$W/logs/kdstep-$KS.txt" | cut -c1-200
