#!/bin/bash
# SPDX-License-Identifier: Apache-2.0
# Part of the Intel port of radiance (milestone 3, run 60; owner note 49: the agreement table with the matrix-unit
# expert form on). Inside ri-m3rt on both cards, started by D/m3-step.sh with STEP_ENGINE=1 and STEP_CARD=both:
# ONE model load at the service setting (as m2-heat.sh's mtp4g: MTP on, the gate, 4 GiB headroom; mtp4gmx adds
# RADIANCE_XE_MOE_MX=1, the matrix-unit expert form -- the service default since owner note 49, a switch on
# images up to :57; mtp4gv2m sets =0, the GEMV form), then the /out/card/NN-*.path + .body requests ONCE each,
# in order (the agreement card: greedy, <= 256 tokens, the logprobs asked for), each answer to
# /out/resp/NN-name.http, one line a request to /out/requests.txt ("<epoch> <name> <wall s> <http status>"),
# /stats before and after. Compared with production's answers on amd-host (m3/agree_table.py).
#   m3-agree.sh CFG
H=/opt/ri/radiance_home; P=18421
CFG=${1:-mtp4gmx}
get() {
  exec 4<>/dev/tcp/127.0.0.1/$P || return
  printf 'GET %s HTTP/1.1\r\nHost: l\r\nConnection: close\r\n\r\n' "$1" >&4
  local line; while IFS= read -r line <&4; do [ "$line" = $'\r' ] && break; done
  cat <&4; exec 4<&-
}
post_raw() {   # post_raw PATH BODYFILE OUT -> prints the status line
  local body; body=$(cat "$2")
  exec 5<>/dev/tcp/127.0.0.1/$P || return 1
  printf 'POST %s HTTP/1.1\r\nHost: l\r\nContent-Type: application/json\r\nContent-Length: %d\r\nConnection: close\r\n\r\n%s' \
    "$1" "${#body}" "$body" >&5
  cat <&5 > "$3"; exec 5<&-
  head -1 "$3" | tr -d '\r'
}
ARGS=$(grep -v '^#' /out/flashnext-card1.args | tr '\n' ' ')
ENVV=()
case $CFG in
  mtp4g) ;;
  mtp4gmx) ENVV=(RADIANCE_XE_MOE_MX=1) ;;
  # run 103 (owner note 120 item 2): the same with the all-reduce's LL form, against mtp4gmx on the same image --
  # whether any served answer depended on the shipped form's large-message race (the prompts' prefill all-reduces)
  mtp4gmxar4) ENVV=(RADIANCE_XE_MOE_MX=1 RADIANCE_XE_AR_PROTO=4) ;;
  mtp4gv2m) ENVV=(RADIANCE_XE_MOE_MX=0) ;;
  # run 108 (owner note 109 item 5): the GDN chunk scan on the matrix units (v block 64), LL by the image's default
  mtp4gmxjm64) ENVV=(RADIANCE_XE_MOE_MX=1 RADIANCE_XE_GDN_CS_JM=64) ;;
  # ... with the expert matrix form's compact grid (823c79d: no wrong-expert blocks past the regular runs)
  mtp4gmxcxjm64) ENVV=(RADIANCE_XE_MOE_MX=1 RADIANCE_XE_MOE_MX_COMPACT=1 RADIANCE_XE_GDN_CS_JM=64) ;;
  # run 109 (owner note 126): the engine's defaults (COMPACT and the jm64 chunk scan built in) + the m3 default STAGE
  mtp4gm3) ENVV=(RADIANCE_XE_MOE_MX=1 RADIANCE_XE_MOE_MX_STAGE=1) ;;
  # run 110 (owner note 130): the same with the prefill attention on the matrix units
  mtp4gm3am) ENVV=(RADIANCE_XE_MOE_MX=1 RADIANCE_XE_MOE_MX_STAGE=1 RADIANCE_XE_ATTN_MX=1) ;;
  # run 115 (owner note 141 item 1): the same with the multi-row trunk GEMV a group a lane (exact to the GEMV's source
  # order; the GEMV itself strays from it in a few outputs a launch, so not its bits)
  mtp4gm3gl) ENVV=(RADIANCE_XE_MOE_MX=1 RADIANCE_XE_MOE_MX_STAGE=1 RADIANCE_XE_GEMV_GL=1) ;;
  # run 116: the defaults with format B's launch geometry traced (a format B container's args in the step folder)
  mtp4gm3t) ENVV=(RADIANCE_XE_MOE_MX=1 RADIANCE_XE_MOE_MX_STAGE=1 RADIANCE_XE_I4I8_TRACE=1) ;;
  *) echo "unknown config $CFG"; exit 2 ;;
esac
ARGS=${ARGS/--gpu-headroom-mib 2048/--gpu-headroom-mib 4096}
# a libxe.so newer than the image, from the step folder (run 108); it replaces the container's copy only
[ -f /out/libxe.so ] && cp /out/libxe.so $H/kernels/libxe.so && echo "libxe.so from /out"
echo "$CFG: env ${ENVV[*]:-(none)}; args $ARGS"
t0=$(date +%s)
env "${ENVV[@]}" timeout 3000 /opt/ri/bin/radiance $ARGS --radiance-home $H -v > /out/serve.log 2>&1 &
pid=$!; up=0
for i in $(seq 2400); do
  kill -0 $pid 2>/dev/null || break
  if exec 3<>/dev/tcp/127.0.0.1/$P 2>/dev/null; then
    printf 'GET /health HTTP/1.1\r\nHost: l\r\nConnection: close\r\n\r\n' >&3
    head -1 <&3 | grep -q " 200" && up=1; exec 3<&-
    [ $up = 1 ] && break
  fi
  sleep 1
done 2>/dev/null
echo "$CFG up=$up after $(( $(date +%s) - t0 )) s"
if [ $up = 1 ]; then
  mkdir -p /out/resp
  get /stats > /out/stats-0.txt
  for b in /out/card/[0-9][0-9]-*.body; do
    name=$(basename "$b" .body); pth=$(cat "${b%.body}.path")
    w0=$EPOCHREALTIME
    st=$(post_raw "$pth" "$b" /out/resp/$name.http)
    w1=$EPOCHREALTIME
    echo "$(date +%s) $name $(echo "$w0 $w1" | awk '{printf "%.2f", $2-$1}') $st" | tee -a /out/requests.txt
  done
  get /stats > /out/stats-final.txt
  grep -E "req .* done" /out/serve.log | tail -11
fi
kill $pid 2>/dev/null; wait $pid; echo "$CFG server exit $?"
grep -E "^E |heat engine:|GPU|fault" /out/serve.log | tail -6
