#!/bin/bash
# SPDX-License-Identifier: Apache-2.0
# Part of the Intel port of radiance (milestone 1 step 7; milestone 2 stage 1: tok/s after each tuned kernel).
# Inside ri-m1rt on card1, started by D/m1-step.sh with STEP_ENGINE=1. For each config: serve the
# production model, send /out/card/tps.body REPS times as a streamed request, then stop the server.
#   m2-tps.sh mtp,nomtp,plain,v1,v1nomtp [REPS=3]   (milestone 3: graph = RADIANCE_L0_TAPE=graph (the SYCL-graph tapes, the A/B), default = native command lists; milestone 2: v1 = RADIANCE_XE_MOE_V1=1, the old expert GEMV, the A/B)
#   mtp: /out/flashnext-card1.args; nomtp: --num-speculative-tokens 0; plain: RADIANCE_L0_TAPE=0
# Per config, /out/CFG/: serve.log (-v), and per run K:
#   tps-K.sse    one line per SSE line: "<epoch s.us> <cgroup usage_usec> <the line>"
#   stats-K.txt  /stats every STATS_PERIOD (0.5) s during the run: "T <epoch s.us> <usage_usec>" then
#                the body
# Only shell builtins in the per-chunk path (EPOCHREALTIME, read), so the sampling costs little.
H=/opt/ri/radiance_home; P=18421
CFGS=${1:-mtp}; CFGS=${CFGS//,/ }; REPS=${2:-3}   # "mtp,nomtp,plain": ssh flattens quoted spaces
CG=/sys/fs/cgroup/cpu.stat
usage() { local k v; while read -r k v; do [ "$k" = usage_usec ] && { echo "$v"; return; }; done < $CG; echo 0; }
get_stats() {   # one GET /stats, body to stdout
  exec 4<>/dev/tcp/127.0.0.1/$P || return
  printf 'GET /stats HTTP/1.1\r\nHost: l\r\nConnection: close\r\n\r\n' >&4
  local line; while IFS= read -r line <&4; do [ "$line" = $'\r' ] && break; done
  cat <&4; exec 4<&-
}
sampler() {     # /stats every STATS_PERIOD s (0.5: it costs the capped engine CPU) until the flag file goes
  while [ -e "$1" ]; do echo "T $EPOCHREALTIME $(usage)"; get_stats; echo; sleep "${STATS_PERIOD:-0.5}"; done
}
stream_req() {  # body-file out.sse
  local body line; body=$(cat "$1")
  exec 3<>/dev/tcp/127.0.0.1/$P || return 1
  printf 'POST /v1/completions HTTP/1.1\r\nHost: l\r\nContent-Type: application/json\r\nContent-Length: %d\r\nConnection: close\r\n\r\n%s' \
    "${#body}" "$body" >&3
  while IFS= read -r line <&3; do echo "$EPOCHREALTIME $(usage) $line"; done > "$2"
  exec 3<&-
}
echo "cpu.stat: $(head -3 $CG | tr '\n' ' ')  cpu.max: $(cat /sys/fs/cgroup/cpu.max 2>/dev/null)"
BODY=/tmp/tps-stream.body
sed 's/}$/,"stream":true}/' /out/card/tps.body > $BODY
for CFG in $CFGS; do
  O=/out/$CFG; mkdir -p $O
  ARGS=$(grep -v '^#' /out/flashnext-card1.args | tr '\n' ' ')
  ENVV=()
  case $CFG in
    mtp) ;;
    nomtp) ARGS=${ARGS/--num-speculative-tokens 3/--num-speculative-tokens 0} ;;
    plain) ENVV=(RADIANCE_L0_TAPE=0) ;;
    graph) ENVV=(RADIANCE_L0_TAPE=graph RADIANCE_L0_TAPE_STRETCHES=128) ;;
    graphnomtp) ENVV=(RADIANCE_L0_TAPE=graph RADIANCE_L0_TAPE_STRETCHES=128); ARGS=${ARGS/--num-speculative-tokens 3/--num-speculative-tokens 0} ;;
    v1) ENVV=(RADIANCE_XE_MOE_V1=1) ;;
    v1nomtp) ENVV=(RADIANCE_XE_MOE_V1=1); ARGS=${ARGS/--num-speculative-tokens 3/--num-speculative-tokens 0} ;;
    plainnomtp) ENVV=(RADIANCE_L0_TAPE=0); ARGS=${ARGS/--num-speculative-tokens 3/--num-speculative-tokens 0} ;;
    # owner note 24: the compute runtime's debug keys (NEOReadDebugKeys=1 enables them in a release build,
    # per its FAQ); LogAllocationSummaryReport prints every allocation type's bytes by memory pool at exit;
    # nomtphost adds ForceSystemMemoryPlacement = bits (type-1) of commandBuffer 3, indirectObjectHeap 9,
    # internalHeap 11, linearStream 16, surfaceStateHeap 27 = 0x4008504: command lists and their heaps
    # in system memory; scratch (23) and kernel ISA (14) stay in VRAM.
    nomtplog)  ENVV=(NEOReadDebugKeys=1 PrintDebugSettings=1 LogAllocationSummaryReport=1)
               ARGS=${ARGS/--num-speculative-tokens 3/--num-speculative-tokens 0} ;;
    plainnomtplog) ENVV=(RADIANCE_L0_TAPE=0 NEOReadDebugKeys=1 PrintDebugSettings=1 LogAllocationSummaryReport=1)
               ARGS=${ARGS/--num-speculative-tokens 3/--num-speculative-tokens 0} ;;
    rtkernels) ENVV=(RADIANCE_L0_NTAPE_OWN_KERNELS=0)
               ARGS=${ARGS/--num-speculative-tokens 3/--num-speculative-tokens 0} ;;
    mtpskipdraft) ENVV=(RADIANCE_TAPE_SKIP=-1,1,2,3) ;;
    mtpskiptrunk) ENVV=(RADIANCE_TAPE_SKIP=0) ;;
    mtpskipall) ENVV=(RADIANCE_TAPE_SKIP=-1,0,1,2,3) ;;
    mtpskiphist) ENVV=(RADIANCE_TAPE_SKIP=-1) ;;
    mtpskiprounds) ENVV=(RADIANCE_TAPE_SKIP=1,2,3) ;;
    # run 18: the all-reduce's poll budget at 2e7 (~6 s) so a stalled collective reports itself in seconds
    mtpspin) ENVV=(XE_AR_SPIN=20000000) ;;
    # run 20: half the all-reduce's blocks a launch, so two launches overlapping on a card stay resident
    # (M10/M12: both compute engines 100 % busy with every rank's collectives spinning for the peer's)
    mtpblocks32) ENVV=(XE_AR_BLOCKS=32) ;;
    # run 27: the tape profiler (device timestamps around every replayed command; a per-kernel table in serve.log
    # every 8192 harvested lists)
    profnomtp) ENVV=(RADIANCE_L0_NTAPE_PROFILE=1); ARGS=${ARGS/--num-speculative-tokens 3/--num-speculative-tokens 0} ;;
    profmtp) ENVV=(RADIANCE_L0_NTAPE_PROFILE=1) ;;
    # run 27: the op oracle on the big model (libref on the host checks every issue of the named op at 2e-2;
    # the tiny model has no draft head, so mtp_enter can only be checked here)
    oraclemtp) ENVV=(RADIANCE_OP_ORACLE=${ORACLE_OP:-mtp_enter} RADIANCE_OP_ORACLE_MAX=${ORACLE_MAX:-12} RADIANCE_OP_ORACLE_TOL=0.02) ;;
    # (the args already list --kernels libxe:libref, which the oracle needs)
    nomtphost) ENVV=(NEOReadDebugKeys=1 PrintDebugSettings=1 LogAllocationSummaryReport=1 ForceSystemMemoryPlacement=67142916)
               ARGS=${ARGS/--num-speculative-tokens 3/--num-speculative-tokens 0} ;;
    # run 11, T4: the full mask aborted the driver at startup (linear_stream.h:84 -- a command stream's
    # buffer); the narrow mask places the command buffers alone (type 3 -> bit 2 = 4) in system memory.
    nomtphostcb) ENVV=(NEOReadDebugKeys=1 PrintDebugSettings=1 ForceSystemMemoryPlacement=4)
               ARGS=${ARGS/--num-speculative-tokens 3/--num-speculative-tokens 0} ;;
    *) echo "unknown config $CFG"; continue ;;
  esac
  t0=$(date +%s)
  env "${ENVV[@]}" timeout 3000 /opt/ri/bin/radiance $ARGS --radiance-home $H -v > $O/serve.log 2>&1 &
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
    for k in $(seq "$REPS"); do
      flag=/tmp/sampling-$CFG-$k; touch $flag
      sampler $flag > $O/stats-$k.txt 2>/dev/null &
      spid=$!
      u0=$(usage); w0=$EPOCHREALTIME
      timeout 900 bash -c "$(declare -f usage stream_req); CG=$CG; P=$P; stream_req $BODY $O/tps-$k.sse"
      u1=$(usage); w1=$EPOCHREALTIME
      rm -f $flag; wait $spid
      echo "$CFG run $k: $(grep -c '^[0-9.]* [0-9]* data: {' $O/tps-$k.sse) chunks, $(grep -o '"predicted_per_second":[0-9.]*' $O/tps-$k.sse | tail -1)," \
           "container CPU $(( (u1 - u0) / 1000 )) ms over $(echo "$w0 $w1" | awk '{printf "%.1f", $2-$1}') s"
    done
  fi
  kill $pid 2>/dev/null; wait $pid; echo "$CFG server exit $?"
  grep -E "^E |heat engine:|step decode" $O/serve.log | tail -8
done
