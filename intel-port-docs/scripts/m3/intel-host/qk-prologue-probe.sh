#!/bin/bash
# SPDX-License-Identifier: Apache-2.0
# Part of the Intel port of radiance (milestone 3, run 273, the decode step's pair 3). Inside the step's container
# (card1 alone): xe_qk_prologue_test from the step folder against the image's libxe.so -- the gated attention's four
# prologue rows (rmsnorm q over the [q|gate] column view, rmsnorm k, rope q, rope k) against the qk_norm_rope_gate row
# listed under RADIANCE_XE_QK_PROLOGUE=1, q and k byte for byte over three shapes (RESULT lines). No engine, no model:
# deterministic data, seconds.
#   bash /out/qk-prologue-probe.sh
P=/out/xe_qk_prologue_test
chmod +x $P
echo "=== qk_prologue_test ($(date -u +%H:%M:%S))"
RADIANCE_XE_QK_PROLOGUE=1 RI_LIBXE=${RI_LIBXE:-/opt/ri/radiance_home/kernels/libxe.so} timeout 300 $P 2>&1 | grep -E "^qk_prologue|^RESULT|^PASS|^FAIL|rror|refu"
echo "rc=${PIPESTATUS[0]} ($(date -u +%H:%M:%S))"
