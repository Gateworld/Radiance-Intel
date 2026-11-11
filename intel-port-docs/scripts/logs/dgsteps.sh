#!/bin/bash
# SPDX-License-Identifier: Apache-2.0
# Part of the Intel port: note 165 item 1 (run 162) -- the draft gate's single-prompt steps, two prompts per config:
# the story prompt (REF M229) and the C ring-buffer edit CE2 (REF M236; its card/ holds the body). Each config runs
# the M-pair's MTP-on form with the gate token appended; texts must match the references.
#   bash logs/dgsteps.sh IMAGE_ID STEP_A STEP_B TOKEN     e.g. bash logs/dgsteps.sh e6ab5e5796fe M241 M242 dgs
set -u
IMG=$1; A=$2; B=$3; TOK=$4
W=$HOME/code/radiance-intel
timeout 3600 bash "$W/m3/mstep.sh" "$A" "$IMG" "svcmtpx-gnt0-hc8-rp-gnx-$TOK" M229 svcmtpx-gnt0-hc8-rp-gnx > "$W/logs/mstep-$A.txt" 2>&1
timeout 3600 bash "$W/m3/mstep.sh" "$B" "$IMG" "svcmtpx-gnt0-hc8-rp-gnx-$TOK" M236 svcmtpx-gnt0-hc8-rp-gnx > "$W/logs/mstep-$B.txt" 2>&1
grep -hE "run 2:|klog2|identical|differ|peak Tctl" "$W/logs/mstep-$A.txt" "$W/logs/mstep-$B.txt" | cut -c1-150
grep -h "draft gate" "$W/results-m3/card01/$A/svcmtpx-gnt0-hc8-rp-gnx-$TOK/serve.log" "$W/results-m3/card01/$B/svcmtpx-gnt0-hc8-rp-gnx-$TOK/serve.log" | tail -6 | cut -c1-230
