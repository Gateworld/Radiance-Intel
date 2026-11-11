#!/bin/bash
# SPDX-License-Identifier: Apache-2.0
# Part of the Intel port of radiance (milestone 3, stage A item 1). Inside ri-m3rt on one card: the native
# tape test under each play-ordering mechanism (RADIANCE_L0_NTAPE_SYNC, core/device/l0.cpp n_play_lists),
# to find which side of the order against the SYCL runtime's own commands fails (N2 attempt 3: 246 of 400
# plain -> replay -> plain rounds wrong).
N=${1:-200}
for m in ${MODES:-events natbar depkernel host}; do
  echo "=== RADIANCE_L0_NTAPE_SYNC=$m"
  RADIANCE_L0_NTAPE_SYNC=$m timeout 240 /opt/ri/tests/xe_native_tape_test $N 2>&1 | grep -E '^  [0-9]|^native_tape|FAIL|refused'
  echo "exit ${PIPESTATUS[0]}"
done
