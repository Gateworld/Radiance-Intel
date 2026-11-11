#!/bin/bash
# SPDX-License-Identifier: Apache-2.0
# Part of the Intel port of radiance (milestone 3, runs 114-115, owner note 141 item 1). Inside the step's container
# (card1, no model): xe_gemv_i8_test full from the step folder with the GEMV's default form, then each
# RADIANCE_XE_GEMV_GL value given (default 1 2), with bf16 weight scales and with f32 ones (RI_BS_F32=1, run 115) --
# each case's y hash must equal the default's (the same bits), and the timed cases give the times.
#   bash /out/gemv-gl.sh [GL values...]
chmod +x /out/xe_gemv_i8_test
export RI_CHECK_ALL=1   # run 115: every column at M <= 8 against the GEMV's source order, emulated
vals=${*:-1 2}
for f in 0 1; do
  for gl in 0 $vals; do
    echo "=== RI_BS_F32=$f RADIANCE_XE_GEMV_GL=$gl ($(date -u +%H:%M:%S))"
    env RI_BS_F32=$f RADIANCE_XE_GEMV_GL=$gl timeout 300 /out/xe_gemv_i8_test full 2>&1 | grep -E "M +[0-9]+ N|ms,|gemv_i8_test:"
  done
done
