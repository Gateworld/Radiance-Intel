#!/bin/bash
# SPDX-License-Identifier: Apache-2.0
# Part of the Intel port of radiance (milestone 3, run 110, owner note 129 item 3). Inside the step's container (one
# card): xe_gemv_i8_test from the step folder (its small cases: N <= 200, the host check is tiny) with the trunk GEMV
# as it is (NX) and with RADIANCE_XE_GEMV_G4 / _G4_ONE = 1, 2, 3; each case prints how many y elements differ from the
# GEMV's documented order (emulated on the host) and a hash of y.
#   bash /out/gemv-bits.sh [full]   (full: also Flash-Next's trunk shapes, timed; the 256-row cases' host check is ~2 s)
T=/out/xe_gemv_i8_test
chmod +x $T
for v in 0 1 0 1; do
  echo "=== RADIANCE_XE_GEMV_NX=1 RADIANCE_XE_GEMV_RT=$v ($(date -u +%H:%M:%S))"
  RADIANCE_XE_GEMV_NX=1 RADIANCE_XE_GEMV_RT=$v timeout 300 $T $1 2>&1 | grep -E "M |ms,|test"
done
true
