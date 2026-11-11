#!/bin/bash
# SPDX-License-Identifier: Apache-2.0
# Part of the Intel port of radiance (milestone 3, stage A item 1: owner note 28's acceptance). Inside
# ri-m3rt on one card (m3-step.sh mounts D/m3/<step> at /out): serve the tiny qwen4exp model on the
# container's loopback port and ask one greedy N-token completion (ignore_eos, logprobs) three times over:
# native command-list tapes (the default), SYCL-graph tapes (RADIANCE_L0_TAPE=graph) and plain launches
# (RADIANCE_L0_TAPE=0). One server per mode. HTTP through bash's /dev/tcp.
# -> /out/e2e-{native,graph,plain}.{log,http}, /out/e2e-times.txt; compare on amd-host with m1/cmp_tokens.py.
H=/opt/ri/radiance_home; P=18411; N=${1:-256}
# a libxe.so newer than the image, from the step folder (run 115, as m3-agree.sh's); it replaces the container's copy only
[ -f /out/libxe.so ] && cp /out/libxe.so $H/kernels/libxe.so && echo "libxe.so from /out"
# milestone 3 stage B: E2E_PROMPT=long gives a ~260-token prompt, so the prefill chunk is wide enough for the matrix-unit
# GEMM path (M >= 64) and leaves tail rows for the GEMV (both paths under the oracle)
if [ "${E2E_PROMPT:-}" = long ]; then
  PROMPT="The port of a serving engine to a new card is measured in small steps: every kernel is checked against the host reference before it is timed, every step on the card is guarded for heat and memory, and every result is written down before the next step starts. The weight formats stay as the production files store them, so the loader copies instead of rearranging, and the kernels read the stored rows in sixteen-byte chunks that line up with the groups the scales cover. When a kernel is replaced by a faster one, the old one stays beside it for the shapes the new one does not take, and a switch in the environment brings the old one back for a comparison. The first token of a long prompt is the measure that matters most to the people who type, and the decode rate is the measure that matters most to the people who read; both are reported with the prompt and the settings that produced them, so the numbers can be repeated later on the same two cards. This prompt is long on purpose."
else
  PROMPT="hello world, the"
fi
body() { echo "{\"prompt\":\"$PROMPT\",\"max_tokens\":$N,\"temperature\":0,\"ignore_eos\":true,\"logprobs\":2}"; }
get() {
  exec 3<>/dev/tcp/127.0.0.1/$P || return 1
  printf 'GET %s HTTP/1.1\r\nHost: l\r\nConnection: close\r\n\r\n' "$1" >&3
  head -1 <&3; exec 3<&-
}
post() {
  exec 3<>/dev/tcp/127.0.0.1/$P || return 1
  printf 'POST %s HTTP/1.1\r\nHost: l\r\nContent-Type: application/json\r\nContent-Length: %d\r\nConnection: close\r\n\r\n%s' \
      "$1" "${#2}" "$2" >&3
  cat <&3 > "$3"; exec 3<&-
}
run() {   # run TAG [env...]
  local tag=$1; shift
  # E2E_TP=2 serves the tiny model on both cards (owner note 58's two-rank check; the step needs STEP_CARD=both)
  env "$@" timeout 900 /opt/ri/bin/radiance --model ${MODEL:-/opt/ri/tiny/tiny-q4-m0.rad} --radiance-home $H \
      --tp ${E2E_TP:-1} \
      --kernels libxe:libref --host 127.0.0.1 --port $P --max-model-len 4096 --max-num-seqs 4 \
      --kv-cache-dtype fp8 --debug-accept-reference-kernels -v > /out/e2e-$tag.log 2>&1 &
  local pid=$! up=0 i
  for i in $(seq 600); do
    kill -0 $pid 2>/dev/null || break
    get /health 2>/dev/null | grep -q " 200" && { up=1; break; }
    sleep 1
  done
  echo "$tag: up=$up after ${i}s"
  if [ $up = 1 ]; then
    local t0=$(date +%s.%N)
    post /v1/completions "$(body)" /out/e2e-$tag.http; echo "$tag: request exit $?"
    local t1=$(date +%s.%N)
    echo "$tag: $N tokens in $(echo "$t1 - $t0" | awk '{print $1 - $3}') s (wall, prefill included)" | tee -a /out/e2e-times.txt
    # a second request: the recorded pass replays from the first (warm)
    t0=$(date +%s.%N)
    post /v1/completions "$(body)" /out/e2e-$tag-2.http; echo "$tag: request 2 exit $?"
    t1=$(date +%s.%N)
    echo "$tag: request 2, $N tokens in $(echo "$t1 - $t0" | awk '{print $1 - $3}') s" | tee -a /out/e2e-times.txt
    get /stats > /dev/null 2>&1
  fi
  kill $pid 2>/dev/null; wait $pid; echo "$tag: server exit $?"
  grep -E "l0 tape|passes|refused" /out/e2e-$tag.log | head -12
}
case "${2:-all}" in
  all)    run native; run graph RADIANCE_L0_TAPE=graph; run plain RADIANCE_L0_TAPE=0 ;;
  native) run native ;;
  graph)  run graph RADIANCE_L0_TAPE=graph ;;
  plain)  run plain RADIANCE_L0_TAPE=0 ;;
  # milestone 3, owner note 56: the hc mixing matrices expanded to FP16 at load -- the same answers and logprobs as
  # the E4M3 rows, bit for bit, since the expansion is exact (compare e2e-native*.http with e2e-hcf16*.http)
  hcf16)  run native; run hcf16 RADIANCE_XE_HC_F16=1 ;;
  # run 94, owner note 66: hc_read V2 (1) and with hc_up_mix's staged rows (2) -- bit-identical to native, so the
  # same answers and logprobs (compare e2e-native*.http with e2e-hcv2*.http / e2e-hcv3*.http)
  hcv2)   run native; run hcv2 RADIANCE_XE_HC_V2=1; run hcv3 RADIANCE_XE_HC_V2=2
          grep -h "hc_read V2" /out/e2e-hcv2.log /out/e2e-hcv3.log | head -2 ;;
  # run 98 (owner note 99): level 5 (the S8 decode, the gains with 8 loads in flight) against native
  hcv5)   run native; run hcv5 RADIANCE_XE_HC_V2=5
          grep -h "hc_read V2" /out/e2e-hcv5.log | head -1 ;;
  # run 99: level 7 (the 4-row column form) against native
  hcv7)   run native; run hcv7 RADIANCE_XE_HC_V2=7
          grep -h "hc_read V2" /out/e2e-hcv7.log | head -1 ;;
  # run 99 (owner note 102): the prefill hc on the matrix units (from 64 rows, so E2E_PROMPT=long) against native, then
  # under the op oracle for hc_read (a tolerance form: compare the answers, not the bits)
  pfm)    run native; run pfm RADIANCE_XE_HC_PF_DPAS=1
          grep -h "matrix units" /out/e2e-pfm.log | head -1
          run pfmoracle RADIANCE_XE_HC_PF_DPAS=1 RADIANCE_L0_TAPE=0 RADIANCE_OP_ORACLE=hc_read RADIANCE_OP_ORACLE_MAX=100000 \
              RADIANCE_OP_ORACLE_TOL="${4:-0.03}"
          echo "oracle: issues checked $(grep -c '^W oracle #' /out/e2e-pfmoracle.log), MISMATCH $(grep -c MISMATCH /out/e2e-pfmoracle.log), agrees $(grep -c ' agrees ' /out/e2e-pfmoracle.log)"
          grep MISMATCH /out/e2e-pfmoracle.log | head -5 ;;
  # run 114 (note 143 d): any ops under the op oracle, plain launches (MODEL= picks the model; $3 the ops, $4 the
  # tolerance), e.g. format B's moe_gemm_q,had_quant_act_i8g on the tiny w4g64i8h model
  oracle) run native
          run oracle RADIANCE_L0_TAPE=0 RADIANCE_OP_ORACLE="${3:-*}" RADIANCE_OP_ORACLE_MAX=100000 \
              RADIANCE_OP_ORACLE_TOL="${4:-0.03}"
          echo "oracle: issues checked $(grep -c '^W oracle #' /out/e2e-oracle.log), MISMATCH $(grep -c MISMATCH /out/e2e-oracle.log), agrees $(grep -c ' agrees ' /out/e2e-oracle.log)"
          grep MISMATCH /out/e2e-oracle.log | head -5 ;;
  # run 115 (owner note 142 item 2): hc_read level 8's columns and gains in one launch (RADIANCE_XE_HC_V2_RG=1)
  # against level 8 alone -- bit for bit, so the same answers and logprobs (compare e2e-hc8*.http with e2e-hcrg*.http)
  hcrg)   run hc8 RADIANCE_XE_HC_V2=8; run hcrg RADIANCE_XE_HC_V2=8 RADIANCE_XE_HC_V2_RG=1
          run hc8p RADIANCE_XE_HC_V2=8 RADIANCE_L0_TAPE=0; run hcrgp RADIANCE_XE_HC_V2=8 RADIANCE_XE_HC_V2_RG=1 RADIANCE_L0_TAPE=0 ;;
  # run 101: the trunk's rows form (note 105 item 1) against native, with tapes and without (M105 / M108's defect)
  rows)   run native; run rows RADIANCE_XE_GEMM_DPAS_ROWS=1; run rowsplain RADIANCE_XE_GEMM_DPAS_ROWS=1 RADIANCE_L0_TAPE=0 ;;
  # run 101 (owner note 107): the small two-card load with the two-card fix and the host pool bound first (run with
  # E2E_TP=2 and STEP_CARD=both); the startup lines name the pools
  tp2hf)  run tp2hf RADIANCE_HOST_FIRST=1 RADIANCE_HOST_POOL_PIECE_MIB=2048 RADIANCE_L0_EXPLICIT_RESIDENCY=1 \
              NEOReadDebugKeys=1 DisableIndirectAccess=1 RADIANCE_L0_VRAM_TRACE=1
          grep -h -E "host_pool|vram ledger" /out/e2e-tp2hf.log | head -12 ;;
  # run 100: note 104's exact expert forms (8-lane teams at K <= 256, 4 rows a team there) and note 105 item 3's hc at
  # 2-63 rows (the short prompt's prefill is a few rows) against native (compare e2e-native*.http with each)
  moe8)   run native; run moe8 RADIANCE_XE_MOE_LPR8=1 RADIANCE_XE_MOE_V2R_SMALLK=4; run hcsr RADIANCE_XE_HC_PF_DPAS=2
          grep -h "matrix units" /out/e2e-hcsr.log | head -1 ;;
  # run 101 (owner note 108): the GDN decode state store without the nontemporal hint, bit for bit -- the same answers
  # and logprobs as native (compare e2e-native*.http with e2e-gnt0*.http)
  gnt0)   run native; run gnt0 RADIANCE_XE_GDN_NT=0 ;;
  # run 101 (owner note 112): the fail-fast exit on a stuck bring-up, with an injected 60 s wait standing in for the
  # stuck bind -- expect up=0 and "server exit 75" within ~5 s, then a normal start with the exit armed (60 s)
  stallexit) run stallexit RADIANCE_STALL_EXIT_S=5 RADIANCE_STALL_EXIT_ANY=1 RADIANCE_DEBUG_STARTUP_WAIT_S=60
             grep -h -E "debug wait|stopped after|nothing has been logged" /out/e2e-stallexit.log | head -4
             run armed RADIANCE_STALL_EXIT_S=60 ;;
  # owner note 58: the first 6 tape appends fail as out of device memory (fake) and are retried after a drain --
  # the answers must equal the plain native run's (compare e2e-native*.http with e2e-fakeappend*.http)
  fakeappend) run native; run fakeappend RADIANCE_L0_FAKE_APPEND_FAIL=${E2E_FAKE:-3}
              grep -c "failed out of device memory" /out/e2e-fakeappend.log ;;
  # run 72 (E11): with more fakes than the retries cover, the recording's lost segment must fail the step (an
  # error to the client), never serve an answer computed without that segment's commands
  fakelost)   run fakelost RADIANCE_L0_FAKE_APPEND_FAIL=8
              grep -c "l0 tape lost" /out/e2e-fakelost.log; head -c 300 /out/e2e-fakelost.http; echo ;;
  # run 79, owner note 53 step 1: the per-layer host step (moe_host_experts, a zero partial nothing adds) with tapes
  # and without -- the answers must equal the plain native run's (compare e2e-native*.http with e2e-hostexp*.http)
  hostexp) run native; run hostexp RADIANCE_HOST_EXPERTS=1; run hostexpplain RADIANCE_HOST_EXPERTS=1 RADIANCE_L0_TAPE=0
           grep -h -E "moe_host_experts|host op|refused" /out/e2e-hostexp.log /out/e2e-hostexpplain.log | head -8 ;;
  # run 80, owner note 65: a recording's segment lists released as it goes (RADIANCE_L0_TAPE_EARLY=1), traced -- the
  # answers must equal the plain native run's (compare e2e-native*.http with e2e-early*.http)
  early) run native; run early RADIANCE_L0_TAPE_EARLY=1 RADIANCE_L0_VRAM_TRACE=1
         grep -h "vram trace" /out/e2e-early.log | tail -3 ;;
  # milestone 3, kernel 4: the op oracle on the card (libref on the host checks every issue of the ops in $3,
  # tolerance $4), the kernel-change gate when the CPU device's JIT cannot compile the kernel (run 24)
  oracle) run oracle RADIANCE_L0_TAPE=0 RADIANCE_OP_ORACLE="${3:-gdn_conv_recurrent_update}" RADIANCE_OP_ORACLE_MAX=100000 \
              RADIANCE_OP_ORACLE_TOL="${4:-0.02}"
          echo "oracle: issues checked $(grep -c '^W oracle #' /out/e2e-oracle.log), MISMATCH $(grep -c MISMATCH /out/e2e-oracle.log), agrees $(grep -c ' agrees ' /out/e2e-oracle.log)"
          grep MISMATCH /out/e2e-oracle.log | head -5 ;;
esac
