#!/bin/bash
# SPDX-License-Identifier: Apache-2.0
# Part of the Intel port of radiance (milestone 3, run 280, owner note 219 B.0 / note 222): where the LL all-reduce's
# ~17-19 us a launch go. Inside the step's container (BOTH cards, no engine, no model), from the step folder:
#   1. xe_clock_probe: the device clock's tick length (OpReadClockKHR), ns a tick from two loop lengths against the host
#      clock (their difference cancels the launch overhead);
#   2. the correctness cases once with the stamps ON (XE_AR_STAMPS=1; 20 iterations): the stamps must change no result;
#   3. for the decode message sizes (2560 x rows bf16, rows 1 / 2 / 4 = 5 / 10 / 20 KB), AR_BENCH (2,000 launches in a
#      graph, three repetitions) with the stamps OFF (today's kernel: the launch pitch) and ON (the pitch again, so the
#      measurement's own cost shows, plus the STAMPS table: median / p90 of each phase for block 0, the last block and
#      every block, in us at the measured tick).
# PROTO 5 is forced (the form decode runs: M398's ar_kernel<0,0,5> at 19.2 us; the default 6 takes form 5 below 256 KiB
# anyway). The fused HW form is not benched: decode does not run it (hc_write is fused with the norm instead).
#   bash /out/ar-stamps.sh [N=2000]
set -u
N=${1:-2000}
L=/out/libxe.so; T=/out/xe_allreduce_test; C=/out/xe_clock_probe
chmod +x $T $C
export XE_AR_SPIN=${XE_AR_SPIN:-20000000}
export AR_BOUND_X=${AR_BOUND_X:-4}      # the instance's bound 4x the message, as the engine's band bound (ar-bench.sh)
export RADIANCE_XE_AR_PROTO=5
echo "=== clock probe ($(date -u +%H:%M:%S))"
timeout 120 $C 4000000 2>&1 | tee /out/clock.txt
# ns a tick = (host us of the 4x loop - host us of the 1x loop) * 1000 / (ticks of the 4x loop - ticks of the 1x loop)
TICK=$(awk '/^RESULT loop/ {i++; d[i]=$9; h[i]=$12} END {if (i>=3 && d[3]>d[2]) printf "%.4f", (h[3]-h[2])*1000/(d[3]-d[2])}' /out/clock.txt)
echo "RESULT tick_ns=${TICK:-unknown}"
echo "=== correctness with the stamps on ($(date -u +%H:%M:%S))"
XE_AR_STAMPS=1 timeout 300 $T $L 20 2>&1 | grep -E "RESULT|stalled|differ|wrong|STALL"
for rows in 1 2 4; do
  echo "=== rows $rows: AR_BENCH $N, stamps OFF ($(date -u +%H:%M:%S))"
  AR_BENCH=$N AR_ROWS=$rows timeout 300 $T $L 2>&1 | grep -E "RESULT|STALL"
  echo "=== rows $rows: AR_BENCH $N, stamps ON"
  XE_AR_STAMPS=1 AR_STAMPS=1 AR_TICK_NS=${TICK:-0} AR_BENCH=$N AR_ROWS=$rows timeout 300 $T $L 2>&1 | grep -E "RESULT|STAMPS|STALL|^  "
done
echo "done ($(date -u +%H:%M:%S))"
