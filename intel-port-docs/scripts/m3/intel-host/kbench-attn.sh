#!/bin/bash
# SPDX-License-Identifier: Apache-2.0
# Part of the Intel port of radiance (milestone 3, run 109). Inside the step's container (one card, STEP_ENGINE=1 for
# /d): attn_paged against libref at prefill rows, with RADIANCE_XE_ATTN_JM unset / =1, kept small (64 rows, a
# context of at most 1,024: owner note 3's host-reference rule), stderr kept.
#   bash /out/kbench-attn.sh [MAXTOK=64] [MAXCTX=1024]
M=/d/qwen3.8-next-flash-fp8-iq4r-moe.rad
H=/opt/ri/radiance_home
mt=${1:-64}; mc=${2:-1024}
# a libxe.so newer than the image, from the step folder; it replaces the container's copy only
[ -f /out/libxe.so ] && cp /out/libxe.so $H/kernels/libxe.so && echo "libxe.so from /out"
# FORMS: 0 the split kernel, 1 RADIANCE_XE_ATTN_JM=1 (the dense prefill form), 2 RADIANCE_XE_ATTN_MX=1 (the per-row form)
for v in ${FORMS:-0 3}; do
  # 3 (run 110): RADIANCE_XE_ATTN_VX=1, the split kernel's 8-byte cache loads and native butterflies (bit for bit)
  case $v in 1) ev=(RADIANCE_XE_ATTN_JM=1) ;; 2) ev=(RADIANCE_XE_ATTN_MX=1) ;; 3) ev=(RADIANCE_XE_ATTN_VX=1) ;; *) ev=() ;; esac
  echo "=== attn_paged --max-tok $mt --max-ctx $mc ${ev[*]:-(split kernel)} ($(date -u +%H:%M:%S))"
  env "${ev[@]}" timeout 300 rad-kbench -m $M --home $H --kernels libxe:libref --op attn_paged \
    --max-ctx "$mc" --max-tok "$mt" --report /out/r-attn$v.md > /out/kb-attn$v.log 2>&1
  echo "rc=$? $(grep -m1 -o -E '[0-9]+ case\(s\) checked, [0-9]+ passed, [0-9]+ failed.*' /out/r-attn$v.md)"
done
md5sum /out/r-attn*.md
