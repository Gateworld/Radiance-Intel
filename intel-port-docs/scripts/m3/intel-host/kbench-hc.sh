#!/bin/bash
# SPDX-License-Identifier: Apache-2.0
# Part of the Intel port of radiance (milestone 3, run 94, owner note 66). Inside the step's container (one card,
# STEP_ENGINE=1 for /d): hc_read against libref on the real container's weights, with the current kernels and with
# hc_read V2 (RADIANCE_XE_HC_V2=1). A bit-identical V2 reports exactly V1's per-case errors. No --bench: it segfaults
# on this backend (run 94).
#   bash /out/kbench-hc.sh        the V2 levels at the decode rows (up to 4)
#   bash /out/kbench-hc.sh pf     run 99, owner note 102: the per-row kernels and the matrix-unit prefill form
#                                 (RADIANCE_XE_HC_PF_DPAS=1, from 64 rows) at up to 512 rows
M=/d/qwen3.8-next-flash-fp8-iq4r-moe.rad
H=/opt/ri/radiance_home
#   bash /out/kbench-hc.sh sr     run 100, owner note 105 item 3: the per-row kernels and the verify-rows form
#                                 (RADIANCE_XE_HC_PF_DPAS=2, from 2 rows) at up to 8 rows
if [ "$1" = sr ]; then
  for p in 0 2; do
    echo "=== RADIANCE_XE_HC_PF_DPAS=$p"
    RADIANCE_XE_HC_PF_DPAS=$p rad-kbench -m $M --home $H --kernels libxe:libref --op hc_read --max-ctx 4096 \
      --max-tok 8 --report /out/report-sr$p.md
    echo "rc=$?"
  done
  exit 0
fi
#   bash /out/kbench-hc.sh v7     run 102, owner note 114: level 5 against level 7 (the down product four rows at a
#                                 time) at up to 7 rows (the V2 path's rows: groups of four and their 1-3 row tails);
#                                 level 7 is meant to give level 5's bits, so its report should be level 5's (md5)
#   bash /out/kbench-hc.sh v8     run 103, owner note 119: level 5 against level 8 (the column-split four-row form), the same
if [ "$1" = v7 ] || [ "$1" = v8 ]; then
  for v in 5 ${1#v}; do
    echo "=== RADIANCE_XE_HC_V2=$v"
    RADIANCE_XE_HC_V2=$v rad-kbench -m $M --home $H --kernels libxe:libref --op hc_read --max-ctx 4096 --max-tok 7 \
      --report /out/report-v$v.md
    echo "rc=$?"
  done
  md5sum /out/report-v5.md /out/report-v${1#v}.md
  exit 0
fi
#   bash /out/kbench-hc.sh rg     run 115, owner note 142 item 2: level 8 against level 8 with its columns and inject gains in
#                                 one launch (RADIANCE_XE_HC_V2_RG=1), meant to be level 8's bits (md5), with the step
#                                 folder's libxe.so when there is one
if [ "$1" = rg ]; then
  [ -f /out/libxe.so ] && cp /out/libxe.so $H/kernels/libxe.so && echo "libxe.so from /out"
  for g in 0 1; do
    echo "=== RADIANCE_XE_HC_V2=8 RADIANCE_XE_HC_V2_RG=$g"
    RADIANCE_XE_HC_V2=8 RADIANCE_XE_HC_V2_RG=$g rad-kbench -m $M --home $H --kernels libxe:libref --op hc_read \
      --max-ctx 4096 --max-tok 7 --report /out/report-rg$g.md
    echo "rc=$?"
  done
  md5sum /out/report-rg0.md /out/report-rg1.md
  exit 0
fi
if [ "$1" = pf ]; then
  for p in 0 1; do
    echo "=== RADIANCE_XE_HC_PF_DPAS=$p"
    RADIANCE_XE_HC_PF_DPAS=$p rad-kbench -m $M --home $H --kernels libxe:libref --op hc_read --max-ctx 4096 \
      --max-tok 512 --report /out/report-pf$p.md
    echo "rc=$?"
  done
  exit 0
fi
for v in 0 1 2 3 4 5; do   # 2: also hc_up_mix's staged W_up rows; 3: the inject gains apart (run 94, step 2); 4: N in LDS, codes by hc_quant_rows (run 95)
  echo "=== RADIANCE_XE_HC_V2=$v"
  RADIANCE_XE_HC_V2=$v rad-kbench -m $M --home $H --kernels libxe:libref --op hc_read --max-ctx 4096 --max-tok 4 \
    --report /out/report-v$v.md
  echo "rc=$?"
done
