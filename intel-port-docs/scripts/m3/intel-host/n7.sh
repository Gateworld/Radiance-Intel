#!/bin/bash
# SPDX-License-Identifier: Apache-2.0
# Part of the Intel port of radiance (milestone 3, stage A item 1). Inside ri-m3rt on one card, v2 adapter
# (m3-step.sh): N6 showed v2 + the native barrier is timing-dependent (246 wrong of 2000, then 0). Each
# ordering mode at 2000 rounds, twice, with the failing rounds' distribution (first, last, per 256-round
# block: ring-event reuse begins at play 256) and the early-completion counters.
N=${1:-2000}
for m in ${MODES:-inorder natbar nowait host}; do
  for k in 1 2; do
    echo "=== RADIANCE_L0_NTAPE_SYNC=$m run $k"
    RADIANCE_L0_NTAPE_SYNC=$m timeout 600 /opt/ri/tests/xe_native_tape_test $N 2>&1 | grep -E '^  (3|6)|^      first|^native_tape|FAIL|refused'
    echo "exit ${PIPESTATUS[0]}"
  done
done
