#!/bin/bash
# SPDX-License-Identifier: Apache-2.0
# Part of the Intel port of radiance (milestone 2, stage 2 step 4): first-token time at ~7K and ~20K prompt
# tokens. Inside ri-m2rt on both cards, started by D/m2-step.sh with STEP_ENGINE=1: ONE model load (CFG as
# in m2-tps.sh), then /out/card/prefill-7k.body and prefill-20k.body (max_tokens 1), each REPS times, wall
# time a request to /out/prefill.txt ("<name> <k> <wall s> <status>"); the server's own "ttft=" and
# "prompt=N tok" lines come from /out/serve.log.
#   m2-prefill.sh CFG [REPS=2]
H=/opt/ri/radiance_home; P=18421
CFG=${1:-nomtp}; REPS=${2:-2}
post_raw() {
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
  mtp) ;;
  nomtp) ARGS=${ARGS/--num-speculative-tokens 3/--num-speculative-tokens 0} ;;
  plain) ENVV=(RADIANCE_L0_TAPE=0) ;;
  plainnomtp) ENVV=(RADIANCE_L0_TAPE=0); ARGS=${ARGS/--num-speculative-tokens 3/--num-speculative-tokens 0} ;;
  *) echo "unknown config $CFG"; exit 2 ;;
esac
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
  for name in ${PREFILL_NAMES:-7k 7kb}; do   # milestone 3 run 48: PREFILL_NAMES="7klp" posts the logprobs body only
    for k in $(seq "$REPS"); do
      w0=$EPOCHREALTIME
      st=$(post_raw /v1/completions /out/card/prefill-$name.body /out/resp/prefill-$name-$k.http)
      w1=$EPOCHREALTIME
      line="prefill-$name $k $(echo "$w0 $w1" | awk '{printf "%.2f", $2-$1}') $st"
      echo "$line" | tee -a /out/prefill.txt
    done
  done
  grep -E "req .* done" /out/serve.log | grep -oE "prompt=[0-9]+ .*ttft=[0-9.]+s" | tail -4
fi
kill $pid 2>/dev/null; wait $pid; echo "$CFG server exit $?"
grep -E "^E |step prefill|prefill" /out/serve.log | tail -4
