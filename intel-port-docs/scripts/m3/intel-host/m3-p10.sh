#!/bin/bash
# SPDX-License-Identifier: Apache-2.0
# Part of the Intel port of radiance (milestone 3, run 41: owner note 40's prompt 10, time-boxed to one run). Inside
# ri-m3rt on both cards (D/m3-step.sh, STEP_ENGINE=1): for each variant, serve the model with the service args
# changed as the variant says, send /out/card/raw10.body once (prompt 10 as the card rendered it in milestone 1:
# 462 tokens, max_tokens 2, temperature 0, logprobs 5), keep the whole response in /out/p10/VARIANT.http, print the
# first token's top logprobs, and stop the server. The host reference cannot run on amd-host (milestone 1 report
# section 11.2: libref's expert layout is not readable in place), so the comparison is across the port's own
# paths: which switch moves the first token toward production's 'The' (-0.050 on 1.11.0; the card said 'We'
# -0.318 in milestone 1).
#   m3-p10.sh base,kvfp16,noplanes,moev1,nomtp
H=/opt/ri/radiance_home; P=18421
VARS=${1:-base}; VARS=${VARS//,/ }
post_raw() {   # post_raw PATH BODYFILE OUT: the whole response (status, headers, body) to OUT
  local body; body=$(cat "$2")
  exec 5<>/dev/tcp/127.0.0.1/$P || return 1
  printf 'POST %s HTTP/1.1\r\nHost: l\r\nContent-Type: application/json\r\nContent-Length: %d\r\nConnection: close\r\n\r\n%s' \
    "$1" "${#body}" "$body" >&5
  cat <&5 > "$3"; exec 5<&-
}
mkdir -p /out/p10
echo "cpu.max: $(cat /sys/fs/cgroup/cpu.max 2>/dev/null)"
for V in $VARS; do
  ARGS=$(grep -v '^#' /out/flashnext-card1.args | tr '\n' ' ')
  ARGS=${ARGS/--gpu-headroom-mib 2048/--gpu-headroom-mib 4096}     # the service setting (run 39)
  ENVV=(RADIANCE_SPEC_MAX_SEQS=1)
  case $V in
    base) ;;
    kvfp16)   ARGS=${ARGS/--kv-cache-dtype fp8/--kv-cache-dtype fp16} ;;      # the FP8 K/V conversions
    kvbf16)   ARGS=${ARGS/--kv-cache-dtype fp8/--kv-cache-dtype bf16} ;;
    noplanes) ENVV+=(RADIANCE_XE_NO_PLANES=1) ;;                               # the expert planes GEMM off
    moev1)    ENVV+=(RADIANCE_XE_MOE_V1=1) ;;                                   # the v1 expert path
    nomtp)    ARGS=${ARGS/--num-speculative-tokens 3/--num-speculative-tokens 0} ;;
    chunk)    ARGS=${ARGS/--max-num-batched-tokens 4096/--max-num-batched-tokens 1024} ;;
    # P10b (run 41): bf16 K/V matched production's top token at -0.0045 (fp8: -0.47, '用户' second at -1.19), so the
    # fp8 K/V path is the divergence; these split it further -- the other fp8 layout, and bf16 with the fallback expert path
    kvfp8v4)        ARGS=${ARGS/--kv-cache-dtype fp8/--kv-cache-dtype fp8_v4} ;;
    kvbf16noplanes) ARGS=${ARGS/--kv-cache-dtype fp8/--kv-cache-dtype bf16}; ENVV+=(RADIANCE_XE_NO_PLANES=1) ;;
    *) echo "unknown variant $V"; continue ;;
  esac
  t0=$(date +%s)
  env "${ENVV[@]}" timeout 900 /opt/ri/bin/radiance $ARGS --radiance-home $H -v > /out/p10/$V.serve.log 2>&1 &
  pid=$!; up=0
  for i in $(seq 400); do
    kill -0 $pid 2>/dev/null || break
    if exec 3<>/dev/tcp/127.0.0.1/$P 2>/dev/null; then
      printf 'GET /health HTTP/1.1\r\nHost: l\r\nConnection: close\r\n\r\n' >&3
      head -1 <&3 | grep -q " 200" && up=1; exec 3<&-
      [ $up = 1 ] && break
    fi
    sleep 1
  done 2>/dev/null
  echo "$V up=$up after $(( $(date +%s) - t0 )) s"
  if [ $up = 1 ]; then
    post_raw /v1/completions /out/card/raw10.body /out/p10/$V.http
    echo "$V: $(head -1 /out/p10/$V.http | tr -d '\r') $(grep -o '"top_logprobs":\[{[^}]*}' /out/p10/$V.http | head -1 | cut -c1-320)"
    echo "$V text: $(grep -o '"text":"[^"]*"' /out/p10/$V.http | head -1)"
  fi
  kill $pid 2>/dev/null; wait $pid; echo "$V server exit $?"
  grep -E "^E " /out/p10/$V.serve.log | head -3
done
