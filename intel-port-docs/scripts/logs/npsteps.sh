#!/bin/bash
# SPDX-License-Identifier: Apache-2.0
# Part of the Intel port: note 165 item 1 (run 162) -- the gate's single-prompt steps with no step issued ahead:
# the story prompt without the gate (the pipelining's cost), then the story and CE2 at the gate's threshold.
#   bash logs/npsteps.sh IMAGE_ID STEP_A STEP_B STEP_C TOKEN    e.g. ... M247 M248 M249 dg2
set -u
IMG=$1; A=$2; B=$3; C=$4; TOK=$5
W=$HOME/code/radiance-intel
timeout 3600 bash "$W/m3/mstep.sh" "$A" "$IMG" "svcmtpx-gnt0-hc8-rp-gnx-nopipe" M229 svcmtpx-gnt0-hc8-rp-gnx > "$W/logs/mstep-$A.txt" 2>&1
timeout 3600 bash "$W/m3/mstep.sh" "$B" "$IMG" "svcmtpx-gnt0-hc8-rp-gnx-nopipe-$TOK" M229 svcmtpx-gnt0-hc8-rp-gnx > "$W/logs/mstep-$B.txt" 2>&1
timeout 3600 bash "$W/m3/mstep.sh" "$C" "$IMG" "svcmtpx-gnt0-hc8-rp-gnx-nopipe-$TOK" M236 svcmtpx-gnt0-hc8-rp-gnx > "$W/logs/mstep-$C.txt" 2>&1
grep -hE "run 2:|klog2|identical|differ|peak Tctl" "$W/logs/mstep-$A.txt" "$W/logs/mstep-$B.txt" "$W/logs/mstep-$C.txt" | cut -c1-150
grep -h "draft gate:" "$W/results-m3/card01/$B/svcmtpx-gnt0-hc8-rp-gnx-nopipe-$TOK/serve.log" | tail -2 | cut -c1-230
grep -h "draft gate:" "$W/results-m3/card01/$C/svcmtpx-gnt0-hc8-rp-gnx-nopipe-$TOK/serve.log" | tail -2 | cut -c1-230
