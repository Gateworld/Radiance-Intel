#!/bin/bash
# SPDX-License-Identifier: Apache-2.0
# Part of the Intel port of radiance (milestone 3, run 114, owner notes 131 / 134 item 3). Inside the step's container
# (card1): format B's grouped GEMM test (xe_moe_i4i8_test) against the step folder's libxe.so.
#   bash /out/i4i8-test.sh
chmod +x /out/xe_moe_i4i8_test
timeout 300 /out/xe_moe_i4i8_test /out/libxe.so
