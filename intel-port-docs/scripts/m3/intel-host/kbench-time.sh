#!/bin/bash
# SPDX-License-Identifier: Apache-2.0
# Part of the Intel port of radiance (milestone 3, run 101). Inside the step's container (one card, STEP_ENGINE=1 for
# /d): rad-kbench's own timer (--bench --timing-only, libref loaded for the schemas) on one op at the decode rows;
# run 94 saw --bench segfault on hc_read, so this tries it per op. Extra environment as arguments after the op.
#   bash /out/kbench-time.sh OP [MAXTOK=4] [VAR=value ...]
M=/d/qwen3.8-next-flash-fp8-iq4r-moe.rad
H=/opt/ri/radiance_home
op=${1:?op}; mt=${2:-4}; shift 2 2>/dev/null || shift
echo "=== rad-kbench --bench --timing-only --op $op --max-tok $mt $* ($(date -u +%H:%M:%S))"
env "$@" timeout 400 rad-kbench -m $M --home $H --kernels libxe:libref --op "$op" --max-ctx 4096 --max-tok "$mt" \
  --bench --timing-only --bench-iters 200 --report /out/report-time-$op.md 2>&1 | grep -v -E '^(I|E) ' | tail -40
echo "rc=${PIPESTATUS[0]}"
