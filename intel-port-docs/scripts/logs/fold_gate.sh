#!/bin/bash
# SPDX-License-Identifier: Apache-2.0
# Part of the Intel port (run 283; the L1 fold's exactness gate). Two gdnlat.sh steps compared on the FUSED kernel
# (SPLIT 0), which carries no switch: the per-call hash lines that follow every "=== <form> fused" header in both
# steps' outputs (both state forms, T 1 and T 4), and the anchor-seq schedule's 120 per-call hash lines (REF's fetched
# copy against NEW's, read from intel-host's step folder). Every line must be equal: then the folded kernel is the pre-fold
# kernel bit for bit on the bench's inputs.
#   bash logs/fold_gate.sh REF_STEP NEW_STEP      e.g. bash logs/fold_gate.sh KG17 KG18
set -u
REF=$1; NEW=$2
W=$HOME/code/radiance-intel
R=$W/results-m3/card01
D=/mnt/fast-storage/models/test/radiance-intel
fused_lines() { awk '/^=== .* fused /{want=1; next} want && /^RESULT gdn_split T . hashes/{print; want=0}' "$1"; }
echo "=== per-call hashes of the fused kernel: $REF vs $NEW"
fused_lines "$R/$REF-out.txt" > /tmp/fold-ref.txt
fused_lines "$R/$NEW-out.txt" > /tmp/fold-new.txt
wc -l < /tmp/fold-ref.txt; wc -l < /tmp/fold-new.txt
if cmp -s /tmp/fold-ref.txt /tmp/fold-new.txt && [ -s /tmp/fold-ref.txt ]; then echo "RESULT fold gate per-call: EQUAL ($(wc -l < /tmp/fold-ref.txt) lines)"
else echo "RESULT fold gate per-call: DIFFER"; diff /tmp/fold-ref.txt /tmp/fold-new.txt | head -6; fi
echo "=== anchor-seq x20 of the fused kernel: $REF-seq-fused.txt vs $NEW's seq-lat-0.txt"
[ -s "$R/$REF-seq-fused.txt" ] || { echo "no $R/$REF-seq-fused.txt"; exit 1; }
ssh intel-host cat "$D/m3/$NEW/seq-lat-0.txt" > "$R/$NEW-seq-fused.txt" || exit 1
n1=$(grep -c "RESULT anchor-seq" "$R/$REF-seq-fused.txt"); n2=$(grep -c "RESULT anchor-seq" "$R/$NEW-seq-fused.txt")
if [ "$n1" -gt 0 ] && cmp -s <(grep "RESULT anchor-seq" "$R/$REF-seq-fused.txt") <(grep "RESULT anchor-seq" "$R/$NEW-seq-fused.txt"); then
  echo "RESULT fold gate anchor-seq: EQUAL ($n2 of $n1 calls)"
else
  echo "RESULT fold gate anchor-seq: DIFFER ($n2 vs $n1 calls)"; diff <(grep "RESULT anchor-seq" "$R/$REF-seq-fused.txt") <(grep "RESULT anchor-seq" "$R/$NEW-seq-fused.txt") | head -6
fi
