#!/bin/bash
# SPDX-License-Identifier: Apache-2.0
# Part of the Intel port of radiance (milestone 3, stage A item 1). Inside ri-m3rt on one card, with the v2
# Level Zero adapter (UR_LOADER_USE_LEVEL_ZERO_V2=1, set by m3-step.sh): the confirmation of N5's finding --
# the native tape test at 2000 rounds twice (the default ordering: a native barrier after the play), then the
# tape lifetime test and the tape bench on the native path (graveyard, per-request buffers, the per-launch figure).
echo "UR_LOADER_USE_LEVEL_ZERO_V2=${UR_LOADER_USE_LEVEL_ZERO_V2:-unset}"
for k in 1 2; do
  echo "=== native_tape 2000 rounds, run $k"
  timeout 600 /opt/ri/tests/xe_native_tape_test 2000 2>&1 | grep -E '^  [0-9]|^native_tape|FAIL|refused|l0 tape:'
  echo "exit ${PIPESTATUS[0]}"
done
echo "=== native_tape, the host oracle, 500 rounds"
RADIANCE_L0_NTAPE_SYNC=host timeout 300 /opt/ri/tests/xe_native_tape_test 500 2>&1 | grep -E '^  3a|^native_tape'
echo "exit ${PIPESTATUS[0]}"
echo "=== tape_lifetime 200 16"
timeout 900 /opt/ri/tests/xe_tape_lifetime_test 200 16 2>&1 | grep -vE '^I l0 tape [0-9]+: play'
echo "exit ${PIPESTATUS[0]}"
echo "=== tape_bench 2000 20 dev"
timeout 300 /opt/ri/tests/xe_tape_bench 2000 20 dev 2>&1 | grep -vE '^I l0 tape [0-9]+: play'
echo "exit ${PIPESTATUS[0]}"
echo "=== tape_bench 2000 20 host"
timeout 300 /opt/ri/tests/xe_tape_bench 2000 20 host 2>&1 | grep -vE '^I l0 tape [0-9]+: play'
echo "exit ${PIPESTATUS[0]}"
