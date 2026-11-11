#!/bin/bash
# SPDX-License-Identifier: Apache-2.0
# Part of the Intel port of radiance (milestone 3, run 115, owner note 142 item 1). Inside the step's container (card1,
# no model): xe_gemv_i8_test's shift check -- a column's output against the same column launched k columns later
# (1 and 4 rows, k 1 / 3 / 8 / 37) -- with the GEMV's float steps as served (RADIANCE_XE_GEMV_NX=1) and without.
#   bash /out/gemv-shift.sh
chmod +x /out/xe_gemv_i8_test
for nx in 1 0; do
  echo "=== RADIANCE_XE_GEMV_NX=$nx ($(date -u +%H:%M:%S))"
  env RI_SHIFT=1 RADIANCE_XE_GEMV_NX=$nx timeout 300 /out/xe_gemv_i8_test 2>&1 | grep -E "shift|gemv_i8_test:"
done
