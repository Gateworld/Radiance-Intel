#!/bin/bash
# SPDX-License-Identifier: Apache-2.0
# Part of the Intel port of radiance (milestone 3, run 115, owner note 146 item 3). Inside localhost/ri-m1build:3: the
# one-layer convert's shape on the tiny model -- the served recipe with layer 0's routed experts in format B by a
# --quant rule ahead of it, --reuse of the tiny w4nl container -- then --plan-only and the convert itself.
#   podman run ... localhost/ri-m1build:3 bash /w/m3/tiny-l0.sh
set -u
T=/w/tiny; B=/w/build-m1l0
export RADIANCE_HOME=$B/radiance_home CALIB=$T/calib OMP_NUM_THREADS=8
Q='blk.0.ffn_*_exps.*.weight=gptq:codes=i4,group=64,scale=bf16,transform=fwht128,cd=3,calib=$CALIB'
R=/w/src/data/recipes/qwen4exp-w4nl64-i8-hc8m.recipe
$B/bin/rad-convert $T/ckpt-m0 --recipe $R --quant "$Q" --reuse $T/tiny-q4-m0.rad --plan-only > $T/convert-m0-l0-plan.log 2>&1
echo "plan rc=$?"
$B/bin/rad-convert $T/ckpt-m0 --recipe $R --quant "$Q" --reuse $T/tiny-q4-m0.rad -o $T/tiny-q4-m0-l0b.rad \
    > $T/convert-m0-l0.log 2>&1
echo "convert rc=$?"
