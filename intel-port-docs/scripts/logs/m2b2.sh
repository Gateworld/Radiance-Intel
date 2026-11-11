#!/bin/bash
# SPDX-License-Identifier: Apache-2.0
# Part of the Intel port: note 163 item 2 (run 161) -- the two-card exchange probe (round 2's m2b_exchange, image
# ri-tune:6 on intel-host) run again on the new PCIe link, under the heat rules: start <= 43 C, the guard at --hot 80.
# Output: results-m3/card01/M2B2-out.txt (+ the kernel log check inside out-m2b.txt's tail).
set -u
W=$HOME/code/radiance-intel
D=/mnt/fast-storage/models/test/radiance-intel
timeout 300 "$W/intel-host-engine-guard.sh" --precheck 1 || exit 1
timeout 600 bash "$W/m3/cool43.sh" 540 || exit 1
"$W/intel-host-engine-guard.sh" --hot 80 ri-m2b 90 > "$W/logs/guard-M2B2.txt" 2>&1 &
guard=$!
timeout 400 ssh intel-host bash "$D/m2b-run.sh" > "$W/results-m3/card01/M2B2-out.txt" 2>&1
echo "m2b rc=$?"
wait $guard
tail -1 "$W/logs/guard-M2B2.txt"
tail -16 "$W/results-m3/card01/M2B2-out.txt"
