#!/bin/bash
# SPDX-License-Identifier: Apache-2.0
# Part of the Intel port of radiance (milestone 3, run 114, note 143 d). Inside the step's container (card1): the
# tiny models' card answers for the host comparison -- today's w4nl tiny model natively (the baseline: does it match
# amd-host's host reference?), then the format B tiny model natively and under the op oracle for its two new ops.
#   bash /out/ti2.sh
mkdir -p /out/w4nl /out/fb
env MODEL=/opt/ri/tiny/tiny-q4-m0.rad bash /out/m3-e2e.sh 64 native
mv /out/e2e-native.http /out/e2e-native-2.http /out/e2e-native.log /out/w4nl/
env MODEL=/out/tiny-q4-m0-w4g64i8h.rad bash /out/m3-e2e.sh 64 oracle moe_gemm_q,had_quant_act_i8g 0.03
