#!/bin/bash
# SPDX-License-Identifier: Apache-2.0
# Part of the Intel port of radiance (milestone 3, stage A item 1). Inside ri-m3rt on one card: the native
# tape test's case 3 under the SYCL runtime's own in-order knobs (the Unified Runtime Level Zero adapter:
# UR_L0_USE_DRIVER_INORDER_LISTS=1 makes the runtime's immediate lists driver-in-order; the v2 adapter uses
# counter-based events), each with the backend's play-ordering modes, plus the early-completion counters
# (N4: the runtime's next command does not wait for the appended lists with any device-side mechanism).
N=${1:-200}
echo "adapters: $(ls /opt/ri/lib | grep -i ur_adapter | tr '\n' ' ')"
run() {   # run LABEL [ENV=VAL ...]
  local label=$1; shift
  for m in ${MODES:-events natbar}; do
    echo "=== $label RADIANCE_L0_NTAPE_SYNC=$m"
    env "$@" RADIANCE_L0_NTAPE_SYNC=$m timeout 240 /opt/ri/tests/xe_native_tape_test $N 2>&1 | grep -E '^  (3|6)|^native_tape|FAIL|refused|UR_|error'
    echo "exit ${PIPESTATUS[0]}"
  done
}
run default
run driver-inorder UR_L0_USE_DRIVER_INORDER_LISTS=1
run v2-adapter UR_LOADER_USE_LEVEL_ZERO_V2=1
run v2-inorder UR_LOADER_USE_LEVEL_ZERO_V2=1 UR_L0_USE_DRIVER_INORDER_LISTS=1
