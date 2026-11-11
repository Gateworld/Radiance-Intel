#!/bin/bash
# SPDX-License-Identifier: Apache-2.0
# Part of the Intel port of radiance (milestone 3, run 112, owner note 140): the calibration recording's two halves
# as one directory, by hard links (same filesystem, no copy). libquant sums gram.r<r>.<stem>.bin over every r < 16
# and checks only the magic, K and domain, so half A keeps ranks 0 / 1 and half B's become 2 / 3.
#   calib-merge.sh A_DIR B_DIR OUT_DIR
set -eu
A=$1; B=$2; O=$3
mkdir -p "$O"
cd "$A"
for f in gram.r0.* gram.r1.*; do ln "$A/$f" "$O/$f"; done
cd "$B"
for f in gram.r0.*; do ln "$B/$f" "$O/gram.r2.${f#gram.r0.}"; done
for f in gram.r1.*; do ln "$B/$f" "$O/gram.r3.${f#gram.r1.}"; done
echo "calib-merge: $(ls "$O" | wc -l) files in $O ($(ls "$A" | wc -l) from A, $(ls "$B" | wc -l) from B)"
