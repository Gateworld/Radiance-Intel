#!/bin/bash
# SPDX-License-Identifier: Apache-2.0
# Part of the Intel port of radiance (milestone 3, run 206: the format C level-table count, the supervisor's desk item).
# Inside a step's container (no GPU work): rad-info's recipe and its per-tensor listing of the served container, read
# only, written to /out for the host-side count (m3/intel-host/radtab.py).
#   bash /out/radinfo.sh [MODEL=/d/qwen3.8-next-flash-fp8-iq4r-moe.rad]
M=${1:-/d/qwen3.8-next-flash-fp8-iq4r-moe.rad}
ls /opt/ri/bin
B=/opt/ri/bin/rad-info
echo "=== rad-info --recipe $M"
timeout 120 $B --recipe "$M" > /out/recipe.txt 2>&1; echo "recipe rc=$?"
echo "=== rad-info -v $M"
timeout 300 $B -v "$M" > /out/radinfo-v.txt 2>&1; echo "list rc=$?"
wc -l /out/recipe.txt /out/radinfo-v.txt
echo "rc=0"
