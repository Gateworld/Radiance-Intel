#!/bin/bash
# SPDX-License-Identifier: Apache-2.0
# Part of the Intel port of radiance (milestone 3, run 111, owner note 135 item 3). Inside the step's container (one
# card): xe_gram_accum_test from the step folder against the step folder's libxe.so -- the small cases first, then the
# expert shapes (`full`; the host reference is ~0.5 G multiply-adds, owner note 3's small check).
#   bash /out/gram-test.sh
L=/out/libxe.so
T=/out/xe_gram_accum_test
chmod +x $T
echo "=== small ($(date -u +%H:%M:%S))"
timeout 120 $T $L 2>&1 | tail -8
echo "rc=${PIPESTATUS[0]}"
echo "=== full ($(date -u +%H:%M:%S))"
timeout 300 $T $L full 2>&1 | tail -10
echo "rc=${PIPESTATUS[0]} ($(date -u +%H:%M:%S))"
