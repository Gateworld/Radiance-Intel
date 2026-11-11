#!/bin/bash
# SPDX-License-Identifier: Apache-2.0
# Part of the Intel port of radiance (milestone 3, run 108, owner note 109 item 5). Inside the step's container (one
# card, STEP_ENGINE=1 for /d): the chunk scan against libref with RADIANCE_XE_GDN_CS_JM at each v block given, with
# rad-kbench's stderr kept, so the form's own line ("I gdn chunk scan: the matrix-unit form ..." or the "W ..." fallback)
# shows which form ran.
#   bash /out/kbench-jm.sh MAXTOK BLOCK...   (BLOCK 0 = the emulated form)
M=/d/qwen3.8-next-flash-fp8-iq4r-moe.rad
H=/opt/ri/radiance_home
mt=${1:?maxtok}; shift
for b in "$@"; do
  # d = the variable unset (the library's default, run 109)
  if [ "$b" = d ]; then ev=(); else ev=(RADIANCE_XE_GDN_CS_JM=$b); fi
  echo "=== gdn_chunk_scan --max-tok $mt ${ev[*]:-(no flag)} ($(date -u +%H:%M:%S))"
  env "${ev[@]}" timeout 300 rad-kbench -m $M --home $H --kernels libxe:libref --op gdn_chunk_scan \
    --max-ctx 4096 --max-tok "$mt" --report /out/r-jm$b-$mt.md > /out/kb-jm$b-$mt.log 2>&1
  echo "rc=$? $(grep -m1 -o -E '[0-9]+ case\(s\) checked, [0-9]+ passed, [0-9]+ failed.*' /out/r-jm$b-$mt.md)"
  grep -E 'gdn chunk scan|Segmentation|abort|error' /out/kb-jm$b-$mt.log | head -5
done
md5sum /out/r-jm*-$mt.md
