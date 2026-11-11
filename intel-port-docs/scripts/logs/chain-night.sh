#!/bin/bash
# SPDX-License-Identifier: Apache-2.0
# Part of the Intel port: tonight's card chain on one image (the supervisor, run 202): the hc merge's profile pair and
# M-pair, its 10-minute rotation, the probes, then the 7K profile. Each step's own wrapper cools, prechecks, guards and
# reads klog2; the chain stops at the first step whose kernel log is not clean (the fault rule: look before going on).
#   bash logs/chain-night.sh IMAGE_ID
set -u
IMG=$1
W=$HOME/code/radiance-intel
cd "$W" || exit 1
B=svcmtpx-gnt0-hc8-rp-gnx
ok() { grep -q "klog2 $1: clean" "$2" || { echo "chain: $1's kernel log is not clean -- stopping ($(date -u +%H:%M:%S))"; exit 1; }; echo "chain: $1 done, klog clean ($(date -u +%H:%M:%S))"; }
bash m3/mstep.sh M280 "$IMG" "$B-hcab-prof" M262 "$B" > logs/mstep-M280.txt 2>&1; ok M280 logs/mstep-M280.txt
bash m3/mstep.sh M281 "$IMG" "$B-prof" M262 "$B" > logs/mstep-M281.txt 2>&1; ok M281 logs/mstep-M281.txt
bash m3/mstep.sh M282 "$IMG" "$B-hcab" M262 "$B" > logs/mstep-M282.txt 2>&1; ok M282 logs/mstep-M282.txt
bash logs/hstep.sh H133 "$IMG" bestrxhcab 10 > logs/hstep-H133.txt 2>&1; ok H133 logs/hstep-H133.txt
bash logs/pstep.sh P97 "$IMG" > logs/pstep-P97.txt 2>&1; ok P97 logs/pstep-P97.txt
bash logs/bstep.sh B150 "$IMG" nomtpcappfmjm64trfmhaprof B135 "7k" > logs/bstep-B150.txt 2>&1; ok B150 logs/bstep-B150.txt
echo "chain: all steps done ($(date -u +%H:%M:%S))"
