#!/bin/bash
# SPDX-License-Identifier: Apache-2.0
# Part of the Intel port of radiance (milestone 3, run 273, the decode step's pairs 1 and 2). Inside the step's container
# (card1 alone): the two paired launches' row checks against the image's libxe.so, byte for byte on the card --
#   pair 2: xe_router_quant_test (router_topk_scatter + had_quant_act_i8g vs router_topk_scatter_quant at M 1 / 4 / 8 /
#           12; RESULT lines, PASS / FAIL);
#   pair 1: xe_gate_quant_test RI_GQ=4 (the two gated quantisers) and RI_GQ=5 (the gated_quant_pair row), the two
#           output files compared (the CPU test's form, here on the GPU device).
# No engine, no model: deterministic data, seconds.
#   bash /out/rq-probe.sh
L=${RI_LIBXE:-/opt/ri/radiance_home/kernels/libxe.so}
chmod +x /out/xe_router_quant_test /out/xe_gate_quant_test
echo "=== router_quant_test ($(date -u +%H:%M:%S))"
RADIANCE_XE_ROUTER_QUANT_PAIR=1 RI_LIBXE=$L timeout 300 /out/xe_router_quant_test 2>&1 | grep -E "^router_quant|^RESULT|^PASS|^FAIL|rror|refu"
echo "rc=${PIPESTATUS[0]} ($(date -u +%H:%M:%S))"
echo "=== gate_quant_test, pair 1 ($(date -u +%H:%M:%S))"
RI_GQ=4 timeout 300 /out/xe_gate_quant_test $L /tmp/gq4.bin 2>&1 | grep -E "^gate_quant|rror|refu"
RADIANCE_XE_GATED_QUANT_PAIR=1 RI_GQ=5 timeout 300 /out/xe_gate_quant_test $L /tmp/gq5.bin 2>&1 | grep -E "^gate_quant|rror|refu"
if cmp /tmp/gq4.bin /tmp/gq5.bin; then echo "PASS gated_quant_pair (card): bit-identical to the two gated quantisers"; echo "rc=0"
else echo "FAIL gated_quant_pair (card): the two output files differ"; echo "rc=1"; fi
