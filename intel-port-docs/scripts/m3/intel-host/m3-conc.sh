#!/bin/bash
# SPDX-License-Identifier: Apache-2.0
# Part of the Intel port of radiance (milestone 2, stage 2 step 3; milestone 3 run 30: the oraclemtp config). Inside ri-m2rt on both cards, started by
# D/m2-step.sh with STEP_ENGINE=1: ONE model load (CFG as in m2-tps.sh: mtp, nomtp, plain, plainnomtp), then
#   1. concurrency: for each N in CONC (default "1 3 6"), N streamed tps requests at once (the 256-token
#      story, ignore_eos), each to /out/conc/N/tps-K.sse with "<epoch> <usage_usec> <line>" lines; a
#      summary line per N: per-request tok/s, aggregate tok/s, wall, container CPU;
#   2. coherence: every /out/card/NN-*.path + .body (milestone 1's 10 requests and the medical one), one
#      at a time, greedy, to /out/coh/NN-*.http (status line, headers, body) -- summarize.py reads them;
#   3. /stats before and after each phase to /out/stats-*.txt.
#   m2-conc.sh CFG [CONC="1 3 6"]
H=/opt/ri/radiance_home; P=18421
CFG=${1:-plainnomtp}; CONC=${2:-1 3 6}
CG=/sys/fs/cgroup/cpu.stat
usage() { local k v; while read -r k v; do [ "$k" = usage_usec ] && { echo "$v"; return; }; done < $CG; echo 0; }
get() {   # get PATH -> body on stdout
  exec 4<>/dev/tcp/127.0.0.1/$P || return
  printf 'GET %s HTTP/1.1\r\nHost: l\r\nConnection: close\r\n\r\n' "$1" >&4
  local line; while IFS= read -r line <&4; do [ "$line" = $'\r' ] && break; done
  cat <&4; exec 4<&-
}
post_raw() {   # post_raw PATH BODYFILE OUT: the whole response (status, headers, body) to OUT
  local body; body=$(cat "$2")
  exec 5<>/dev/tcp/127.0.0.1/$P || return 1
  printf 'POST %s HTTP/1.1\r\nHost: l\r\nContent-Type: application/json\r\nContent-Length: %d\r\nConnection: close\r\n\r\n%s' \
    "$1" "${#body}" "$body" >&5
  cat <&5 > "$3"; exec 5<&-
}
stream_req() {  # stream_req BODYFILE OUT.sse
  local body line; body=$(cat "$1")
  exec 3<>/dev/tcp/127.0.0.1/$P || return 1
  printf 'POST /v1/completions HTTP/1.1\r\nHost: l\r\nContent-Type: application/json\r\nContent-Length: %d\r\nConnection: close\r\n\r\n%s' \
    "${#body}" "$body" >&3
  while IFS= read -r line <&3; do echo "$EPOCHREALTIME $(usage) $line"; done > "$2"
  exec 3<&-
}
echo "cpu.max: $(cat /sys/fs/cgroup/cpu.max 2>/dev/null)"
ARGS=$(grep -v '^#' /out/flashnext-card1.args | tr '\n' ' ')
ENVV=()
case $CFG in
  mtp) ;;
  nomtp) ARGS=${ARGS/--num-speculative-tokens 3/--num-speculative-tokens 0} ;;
  plain) ENVV=(RADIANCE_L0_TAPE=0) ;;
  plainnomtp) ENVV=(RADIANCE_L0_TAPE=0); ARGS=${ARGS/--num-speculative-tokens 3/--num-speculative-tokens 0} ;;
  # milestone 3, run 30: the op oracle on mtp_enter under concurrency; ORACLE_MAX a rank stays small so the
  # host reference's load ends before the heat guard acts
  # milestone 3, run 34: MTP on with the engine's adaptive draft window (drafts follow acceptance and the step cost)
  mtpadaptive) ARGS="$ARGS --draft-window adaptive" ;;
  # run 34: MTP on, drafting only while one sequence decodes (RADIANCE_SPEC_MAX_SEQS=1, the speculation gate)
  mtpgate) ENVV=(RADIANCE_SPEC_MAX_SEQS=1) ;;
  # run 35: the gate with a 4 GiB headroom (C5: at 2 GiB the VRAM floor sent every multi-sequence pass live)
  mtpgate4g) ENVV=(RADIANCE_SPEC_MAX_SEQS=1); ARGS=${ARGS/--gpu-headroom-mib 2048/--gpu-headroom-mib 4096} ;;
  # run 35: C6 stalled at the 3-user phase's first recordings (device 0's play never completed, no all-reduce
  # give-up): the history pass (draft pass -1) issued live, so its new multi-row tape is out of the step
  mtpgate4gskip) ENVV=(RADIANCE_SPEC_MAX_SEQS=1 RADIANCE_TAPE_SKIP=-1); ARGS=${ARGS/--gpu-headroom-mib 2048/--gpu-headroom-mib 4096} ;;
  # run 39: the committed service setting plus the kept-tape bound at 8 a rank (C9 at 6 users: 22 tapes filled the 4 GiB
  # headroom and the VRAM floor sent the passes live, 54.4 against C7's 72.4)
  mtpgate4gt8) ENVV=(RADIANCE_SPEC_MAX_SEQS=1 RADIANCE_TAPE_MAX=8); ARGS=${ARGS/--gpu-headroom-mib 2048/--gpu-headroom-mib 4096} ;;
  # run 40: the service setting with a 6 GiB headroom (C10: the bound of 8 did not keep the floor from firing at 6 users
  # -- the trunk tapes are ~0.8 GB each -- and cost the 3-user phase 10 tok/s of LRU churn)
  mtpgate6g) ENVV=(RADIANCE_SPEC_MAX_SEQS=1); ARGS=${ARGS/--gpu-headroom-mib 2048/--gpu-headroom-mib 6144} ;;
  # run 38 (owner note 42): the repro -- the history-only pass recorded again (the default since :39 issues it live)
  mtpgate4ghist) ENVV=(RADIANCE_SPEC_MAX_SEQS=1 RADIANCE_TAPE_HISTORY_ONLY=1); ARGS=${ARGS/--gpu-headroom-mib 2048/--gpu-headroom-mib 4096} ;;
  # the same with a 6 s all-reduce budget: a stalled handshake prints its give-up instead of hanging
  mtpgate4gspin) ENVV=(RADIANCE_SPEC_MAX_SEQS=1 XE_AR_SPIN=20000000); ARGS=${ARGS/--gpu-headroom-mib 2048/--gpu-headroom-mib 4096} ;;
  # run 108: today's settings (m2-heat.sh bestr: the candidate + gnt0 + hc8 + rp + the m3 defaults, headroom 1280,
  # LL by the image's default) with the MTP gate of the mtpgate runs, to refresh progress.tsv's aggregate row
  best) ENVV=(RADIANCE_SPEC_MAX_SEQS=1 RADIANCE_XE_MOE_MX=1 RADIANCE_TAPE_SHED=0 RADIANCE_TAPE_VRAM_FLOOR_MIB=256
              RADIANCE_L0_TAPE_EARLY=1 RADIANCE_HOST_POOL_PIECE_MIB=2048 RADIANCE_L0_EXPLICIT_RESIDENCY=1 NEOReadDebugKeys=1
              DisableIndirectAccess=1 OverrideCmdListCmdBufferSizeInKb=64 RADIANCE_VRAM_CAP_FROM_FREE=1
              RADIANCE_XE_GDN_NT=0 RADIANCE_XE_HC_V2=8 RADIANCE_XE_ROUTER_PAIR=1 RADIANCE_STALL_EXIT_S=60 RADIANCE_HOST_FIRST=1
              RADIANCE_XE_MOE_MX_STAGE=1)   # run 109 (owner note 126 item 5)
        ARGS=${ARGS/--gpu-headroom-mib 2048/--gpu-headroom-mib 1280}; ARGS=${ARGS/--host-pool-mib 16384/--host-pool-mib 8192}
        ARGS="$ARGS --checkpoint-slots 2" ;;
  oraclemtp) ENVV=(RADIANCE_OP_ORACLE=mtp_enter RADIANCE_OP_ORACLE_MAX=${ORACLE_MAX:-6} RADIANCE_OP_ORACLE_TOL=0.02) ;;
  *) echo "unknown config $CFG"; exit 2 ;;
esac
# concurrency needs the seats: --max-num-seqs 2 in the args -> 6, and a 16K context so six KV sequences
# (fp8, sharded) take ~0.8 GiB instead of 3.2 (one 64K sequence was 0.54 GiB a card in T0's budget) and
# the experts keep their VRAM share
ARGS=${ARGS/--max-num-seqs 2/--max-num-seqs 6}
ARGS=${ARGS/--max-model-len 65536/--max-model-len 16384}
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
  BODY=/tmp/tps-stream.body
  sed 's/}$/,"stream":true}/' /out/card/tps.body > $BODY
  # a warm-up request first (the first pass records nothing here, but the expert heat settles)
  if [ "$CFG" != oraclemtp ]; then
    stream_req $BODY /tmp/warm.sse
    echo "warm-up: $(grep -o '"predicted_per_second":[0-9.]*' /tmp/warm.sse | tail -1)"
  fi
  for N in $CONC; do
    O=/out/conc/$N; mkdir -p $O
    get /stats > $O/stats-before.txt
    u0=$(usage); w0=$EPOCHREALTIME
    # wait on the request jobs only: a bare `wait` also waits for the server (run 11, S2: hung after the
    # warm-up until the step's timeout)
    pids=""
    for k in $(seq "$N"); do stream_req $BODY $O/tps-$k.sse & pids="$pids $!"; done
    wait $pids
    u1=$(usage); w1=$EPOCHREALTIME
    get /stats > $O/stats-after.txt
    wall=$(echo "$w0 $w1" | awk '{printf "%.2f", $2-$1}')
    per=""; tot=0
    for k in $(seq "$N"); do
      r=$(grep -o '"predicted_per_second":[0-9.]*' $O/tps-$k.sse | tail -1 | cut -d: -f2)
      c=$(grep -c '^[0-9.]* [0-9]* data: {' $O/tps-$k.sse)
      per="$per ${r:-?}($c)"
    done
    echo "conc $N: wall ${wall}s, container CPU $(( (u1 - u0) / 1000 )) ms, per-request tok/s(chunks):$per"
  done
  [ "$CFG" = oraclemtp ] && echo "oracle: checks $(grep -c '^W oracle #' /out/serve.log), MISMATCH $(grep -c MISMATCH /out/serve.log), agree $(grep -c ' agrees ' /out/serve.log)"
  mkdir -p /out/coh
  [ "$CFG" = oraclemtp ] || [ "$CFG" = mtpadaptive ] || [ "$CFG" = mtpgate ] || [ "$CFG" = mtpgate4g ] || [ "$CFG" = mtpgate4gskip ] || [ "$CFG" = mtpgate4gspin ] || [ "$CFG" = mtpgate4ghist ] || [ "$CFG" = mtpgate4gt8 ] || [ "$CFG" = mtpgate6g ] || for pth in /out/card/[0-9][0-9]-*.path; do
    name=$(basename "$pth" .path)
    post_raw "$(cat "$pth")" "${pth%.path}.body" /out/coh/$name.http
    echo "coh $name: $(head -1 /out/coh/$name.http | tr -d '\r') $(wc -c < /out/coh/$name.http) bytes"
  done
  get /stats > /out/stats-final.txt
fi
kill $pid 2>/dev/null; wait $pid; echo "$CFG server exit $?"
grep -E "^E |heat engine:|step decode" /out/serve.log | tail -6
