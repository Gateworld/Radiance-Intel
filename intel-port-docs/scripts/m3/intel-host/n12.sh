#!/bin/bash
# SPDX-License-Identifier: Apache-2.0
# Part of the Intel port of radiance (milestone 3, stage A item 1). Inside ri-m3rt on one card, v2 adapter
# (m3-step.sh): N7's two open questions -- (1) the 2000-round order race that survives the `done`-event
# slot reuse (what value does a wrong fold see?), (2) the lifetime test's hang (where, and does the host
# oracle hang too?).
N=${1:-2000}
for k in 1 2; do
  echo "=== native_tape $N rounds, run $k"
  timeout 600 /opt/ri/tests/xe_native_tape_test $N 2>&1 | grep -E '^  [0-9]|^native_tape|FAIL|refused'
  echo "exit ${PIPESTATUS[0]}"
done
echo "=== tape_lifetime 40 16, the host oracle, verbose"
RI_VERBOSE=1 RADIANCE_L0_NTAPE_SYNC=host timeout 300 /opt/ri/tests/xe_tape_lifetime_test 200 16 2>&1 | grep -vE "^I l0 tape" | tail -8
echo "exit ${PIPESTATUS[0]}"
echo "=== tape_lifetime 40 16, the design, verbose"
RI_VERBOSE=1 timeout 300 /opt/ri/tests/xe_tape_lifetime_test 200 16 2>&1 | grep -vE "^I l0 tape" | tail -8
echo "exit ${PIPESTATUS[0]}"
echo "=== tape_bench 2000 20 dev"
timeout 300 /opt/ri/tests/xe_tape_bench 2000 20 dev 2>&1 | grep -vE '^I l0 tape [0-9]+: play'
echo "exit ${PIPESTATUS[0]}"
