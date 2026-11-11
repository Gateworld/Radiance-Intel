#!/bin/bash
# SPDX-License-Identifier: Apache-2.0
# Part of the Intel port of radiance (milestone 2, stage 1 step 1): the per-op profile of decode steps
# on the real model. Inside ri-m1rt on card1, started by D/m2-step.sh with STEP_ENGINE=1. For each
# config: serve with --profile-ops (the engine's own per-op device timer, spec §16: an event pair
# around every launch, collected by one sync a step; tapes are off under it by design, ctx.cpp),
# RADIANCE_PROFILE_EVERY=64 so the cumulative table is printed at steps 64, 128, 192, 256 and at
# exit; send /out/card/tps.body once (256 greedy tokens, ignore_eos). Consecutive dumps differ by
# exactly 64 decode steps, with no prefill in between.
#   m2-prof.sh nomtp,mtp
#   nomtp: --num-speculative-tokens 0 (one pass a step, ~1,600 launches); mtp: as the args file
# Per config, /out/CFG/: serve.log (-v; the tables are in it), tps.http, stats-{before,after}.json.
H=/opt/ri/radiance_home; P=18421
CFGS=${1:-nomtp}; CFGS=${CFGS//,/ }
EVERY=${PROFILE_EVERY:-16}   # milestone 3 P2: 64-token requests, a table every 16 steps
get() {   # path out
  exec 4<>/dev/tcp/127.0.0.1/$P || return
  printf 'GET %s HTTP/1.1\r\nHost: l\r\nConnection: close\r\n\r\n' "$1" >&4
  local line; while IFS= read -r line <&4; do [ "$line" = $'\r' ] && break; done
  cat <&4 > "$2"; exec 4<&-
}
post() {  # path body-file out
  local body; body=$(cat "$2")
  timeout 1500 bash -c 'exec 3<>/dev/tcp/127.0.0.1/'$P' || exit 1
    printf "POST %s HTTP/1.1\r\nHost: l\r\nContent-Type: application/json\r\nContent-Length: %d\r\nConnection: close\r\n\r\n%s" \
      "$0" "${#1}" "$1" >&3
    cat <&3' "$1" "$body" > "$3"
  echo "$(date +%T) $(basename "$3"): $(head -1 "$3" | tr -d '\r') $(grep -o '"predicted_per_second":[0-9.]*' "$3")"
}
echo "cpu.max: $(cat /sys/fs/cgroup/cpu.max 2>/dev/null)"
for CFG in $CFGS; do
  O=/out/$CFG; mkdir -p $O
  ARGS=$(grep -v '^#' /out/flashnext-card1.args | tr '\n' ' ')
  case $CFG in
    mtp) ;;
    nomtp) ARGS=${ARGS/--num-speculative-tokens 3/--num-speculative-tokens 0} ;;
    *) echo "unknown config $CFG"; continue ;;
  esac
  t0=$(date +%s)
  RADIANCE_PROFILE_EVERY=$EVERY timeout 3000 /opt/ri/bin/radiance $ARGS --profile-ops --radiance-home $H -v > $O/serve.log 2>&1 &
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
    get /stats $O/stats-before.json
    u0=$(awk '$1=="usage_usec"{print $2}' /sys/fs/cgroup/cpu.stat); w0=$EPOCHREALTIME
    post /v1/completions "${PROF_BODY:-/out/card/tps.body}" $O/tps.http   # milestone 3 stage B: PROF_BODY=/out/card/prefill-7k.body profiles a prefill
    u1=$(awk '$1=="usage_usec"{print $2}' /sys/fs/cgroup/cpu.stat); w1=$EPOCHREALTIME
    get /stats $O/stats-after.json
    echo "$CFG: container CPU $(( (u1 - u0) / 1000 )) ms over $(echo "$w0 $w1" | awk '{printf "%.1f", $2-$1}') s"
  fi
  kill -INT $pid 2>/dev/null; wait $pid; echo "$CFG server exit $?"
  echo "$CFG: $(grep -c '^per-op timing' $O/serve.log) profile tables; $(grep -c 'step decode' $O/serve.log) step-decode lines"
  grep -E "^E |per-op timing: the step issued|not timed" $O/serve.log | tail -5
done
