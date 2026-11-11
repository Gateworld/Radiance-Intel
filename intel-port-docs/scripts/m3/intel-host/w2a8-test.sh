#!/bin/bash
# SPDX-License-Identifier: Apache-2.0
# Part of the Intel port of radiance (milestone 3, run 123, owner note 151 C). Inside the step's container (card1):
# xe_gemv_w2a8_test (the MTP draft head's GEMV, compiled into the test) in its full mode -- the small cases, then the
# served shape (N 248,320, K 2560, M 1 and 4) timed, each with the rows form (RADIANCE_XE_W2A8_R=2 / 4) compared byte
# for byte with the one-row form and timed beside it.
#   bash /out/w2a8-test.sh
chmod +x /out/xe_gemv_w2a8_test
timeout 300 /out/xe_gemv_w2a8_test full
