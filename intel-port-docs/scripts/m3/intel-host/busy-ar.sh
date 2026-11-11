#!/bin/bash
# SPDX-License-Identifier: Apache-2.0
# Part of the Intel port of radiance (milestone 3, run 87, owner notes 81-82). The two-rank all_reduce test with one
# host core kept busy by a spinning shell loop: if the exchange drops from ~1.2 ms to ~0.1 ms, the cross-device
# write latency follows the host's idle states (the SoC fabric asleep while the host waits), not the engine.
#   bash /out/busy-ar.sh [iters=50]
( while :; do :; done ) &
spin=$!
/out/xe_allreduce_test /opt/ri/radiance_home/kernels/libxe.so "${1:-50}"
rc=$?
kill $spin
exit $rc
