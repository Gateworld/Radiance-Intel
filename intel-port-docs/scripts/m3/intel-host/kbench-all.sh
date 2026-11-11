#!/bin/bash
# SPDX-License-Identifier: Apache-2.0
# Part of the Intel port of radiance (milestone 3, run 102, owner note 111). Inside the step's container (one card,
# STEP_ENGINE=1 for /d): a list of ops against libref on the real container's weights, one rad-kbench a op (every op
# in one run heat-stopped intel-host's CPU at 23 s: KA1), at up to MAXTOK rows -- run on two images (a library-wide form
# off and on) and the reports compared op by op (md5). A form that keeps every value reports the same per-case errors.
#   bash /out/kbench-all.sh TAG [MAXTOK=4] [OPS=comma list] [VAR=value ...]   (the environment for a flagged form)
M=/d/qwen3.8-next-flash-fp8-iq4r-moe.rad
H=/opt/ri/radiance_home
TAG=${1:?tag}; MT=${2:-4}
OPS=${3:-hc_read,moe_gemm_q,router_topk_scatter,row_topk,qsa_score,qsa_select,quant_act_fp8,gated_quant_fp8,rmsnorm,attn_paged,logits_gemm}
shift 3 2>/dev/null
for op in ${OPS//,/ }; do
  echo "=== $op --max-tok $MT $* ($(date -u +%H:%M:%S))"
  env "$@" timeout 300 rad-kbench -m $M --home $H --kernels libxe:libref --op $op --max-ctx 4096 --max-tok "$MT" \
    --report /out/r-$TAG-$op.md > /dev/null 2>&1
  echo "rc=$? $(grep -m1 -o -E '[0-9]+ case\(s\) checked, [0-9]+ passed, [0-9]+ failed' /out/r-$TAG-$op.md)"
done
md5sum /out/r-$TAG-*.md
