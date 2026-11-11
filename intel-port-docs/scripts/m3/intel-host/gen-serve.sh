#!/bin/bash
# SPDX-License-Identifier: Apache-2.0
# Part of the Intel port of radiance (milestone 3, run 284, owner note 228 item 1; logs/run283-mtp-v2-gen-design.md
# section 6). Inside the generation step's container (image ri-m3gen:N = the runtime image + python3; both cards): serve
# the engine with the step's card args (the production-like config, as the M-steps), wait for /health, run the
# generation client against 127.0.0.1 (m2-heat.sh's form: the step container publishes no port), then stop the server.
#   bash /out/gen-serve.sh PART [IDS=] [CONCURRENCY=3] [MAX_MINUTES=45] [ENGINE_NOTE=] [FORM=svc]
# FORM (run 284, after GEN0's stall -- "l0 tape: appending failed out of device memory", 10 s barrier waits, the
# tool-call request never returned): svc = the args as staged (tapes on, headroom 2048); hr4g = the headroom raised to
# 4096 MiB (m2-heat.sh's 4g form); notape = RADIANCE_L0_TAPE=0 (no recorded passes); hr4gnotape = both.
H=/opt/ri/radiance_home; P=18421
PART=${1:?part}; IDS=${2:-}; CONC=${3:-3}; MAXMIN=${4:-45}; NOTE=${5:-}; FORM=${6:-svc}
[ "$IDS" = - ] && IDS=   # "-" = every document of the prompt set (an empty argument does not survive the step's hop)
ARGS=$(grep -v '^#' /out/flashnext-card1.args | tr '\n' ' ')
ENVV=()
case "$FORM" in
  hr4g) ARGS=${ARGS/--gpu-headroom-mib 2048/--gpu-headroom-mib 4096} ;;
  notape) ENVV=(RADIANCE_L0_TAPE=0) ;;
  hr4gnotape) ARGS=${ARGS/--gpu-headroom-mib 2048/--gpu-headroom-mib 4096}; ENVV=(RADIANCE_L0_TAPE=0) ;;
  svc) ;;
  *) echo "unknown FORM $FORM"; exit 2 ;;
esac
echo "=== gen-serve part $PART ids [$IDS] concurrency $CONC max $MAXMIN min ($(date -u +%H:%M:%S))"
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
echo "gen-serve: up=$up after $(( $(date +%s) - t0 )) s"
if [ $up = 1 ]; then
  # --timeout 300: a request that does not return in 5 min is given up (one retry), so one stuck document cannot hold a part
  python3 /out/mtp_v2_gen.py --prompts /out/prompts.jsonl --out /out/docs-part$PART.jsonl --port $P --timeout 300 \
    --concurrency "$CONC" --max-minutes "$MAXMIN" --part "$PART" --engine "$NOTE $FORM" ${IDS:+--ids "$IDS"} 2> /out/gen-part$PART.log
  echo "gen client rc=$?"; tail -3 /out/gen-part$PART.log
  grep -c '"id"' /out/docs-part$PART.jsonl 2>/dev/null
fi
kill $pid 2>/dev/null; wait $pid; echo "server exit $?"
grep -E "^E |GPU|fault" /out/serve.log | tail -4
